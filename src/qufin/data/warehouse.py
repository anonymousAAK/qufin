"""Parquet data warehouse with partitioned storage.

Stores market data in a ``asset_class/ticker/year.parquet`` hierarchy.
Supports predicate pushdown via PyArrow, automatic compaction, and
deduplication on (ticker, date) composite key.

Integrates with :mod:`qufin.data.cache` to check the warehouse before
making API calls.

Requires ``pyarrow`` (optional dependency).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    _HAS_PYARROW = False


def _require_pyarrow() -> None:
    if not _HAS_PYARROW:
        raise ImportError(
            "pyarrow is required for the data warehouse. "
            "Install it with: pip install pyarrow"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WarehouseConfig:
    """Configuration for the parquet data warehouse.

    Parameters
    ----------
    root_dir : Path
        Root directory for partitioned storage.
    compaction_threshold : int
        Merge small files when a partition has more than this many files.
    dedup_keys : list[str]
        Columns used for deduplication.
    date_column : str
        Name of the date/datetime column in DataFrames.
    """

    root_dir: Path = field(default_factory=lambda: Path.home() / ".qufin" / "warehouse")
    compaction_threshold: int = 10
    dedup_keys: list[str] = field(default_factory=lambda: ["ticker", "date"])
    date_column: str = "date"


# ---------------------------------------------------------------------------
# Warehouse
# ---------------------------------------------------------------------------


class ParquetWarehouse:
    """Partitioned Parquet data warehouse.

    Data is stored as::

        root_dir/
          asset_class/
            ticker/
              2023.parquet
              2024.parquet

    Parameters
    ----------
    config : WarehouseConfig | None
        Warehouse configuration. Defaults to WarehouseConfig().
    """

    def __init__(self, config: WarehouseConfig | None = None) -> None:
        _require_pyarrow()
        self.config = config or WarehouseConfig()
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _partition_dir(self, asset_class: str, ticker: str) -> Path:
        d = self.config.root_dir / asset_class / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _year_path(self, asset_class: str, ticker: str, year: int) -> Path:
        return self._partition_dir(asset_class, ticker) / f"{year}.parquet"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        df: pd.DataFrame,
        asset_class: str,
        ticker: str,
    ) -> list[Path]:
        """Write a DataFrame partitioned by year.

        The DataFrame must have a column matching ``config.date_column``.
        Data is deduplicated on (ticker, date) before writing.

        Parameters
        ----------
        df : pd.DataFrame
            Market data to store.
        asset_class : str
            Asset class partition key (e.g. "equity", "option").
        ticker : str
            Ticker symbol.

        Returns
        -------
        list[Path]
            Paths of written parquet files.
        """
        if df.empty:
            return []

        df = self._prepare_df(df, ticker)
        written: list[Path] = []

        # Group by year and write each partition
        for year, group in df.groupby(df[self.config.date_column].dt.year):
            path = self._year_path(asset_class, ticker, int(year))

            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, group], ignore_index=True)
                merged = self._deduplicate(merged)
            else:
                merged = group

            table = pa.Table.from_pandas(merged, preserve_index=False)
            pq.write_table(table, path)
            written.append(path)
            logger.debug("Wrote %d rows to %s", len(merged), path)

        return written

    def _prepare_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Ensure date column is datetime and ticker column exists."""
        df = df.copy()
        if self.config.date_column not in df.columns:
            # Try index
            if df.index.name == self.config.date_column or isinstance(
                df.index, pd.DatetimeIndex
            ):
                df = df.reset_index()
                if df.columns[0] != self.config.date_column:
                    df = df.rename(columns={df.columns[0]: self.config.date_column})
            else:
                raise ValueError(
                    f"DataFrame must have a '{self.config.date_column}' column "
                    f"or a DatetimeIndex."
                )
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        if "ticker" not in df.columns:
            df["ticker"] = ticker
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on configured dedup keys."""
        keys = [k for k in self.config.dedup_keys if k in df.columns]
        if not keys:
            return df
        return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Read with predicate pushdown
    # ------------------------------------------------------------------

    def read(
        self,
        asset_class: str,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read data with optional date filtering via predicate pushdown.

        Parameters
        ----------
        asset_class : str
            Asset class partition.
        ticker : str
            Ticker symbol.
        start_date, end_date : str | None
            ISO date strings for filtering.
        columns : list[str] | None
            Specific columns to read.

        Returns
        -------
        pd.DataFrame
            Filtered data.
        """
        part_dir = self.config.root_dir / asset_class / ticker
        if not part_dir.exists():
            return pd.DataFrame()

        # Determine which year files to read
        parquet_files = sorted(part_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        # Year-level pruning
        if start_date or end_date:
            start_year = pd.Timestamp(start_date).year if start_date else 0
            end_year = pd.Timestamp(end_date).year if end_date else 9999
            parquet_files = [
                f
                for f in parquet_files
                if start_year <= int(f.stem) <= end_year
            ]

        if not parquet_files:
            return pd.DataFrame()

        # Build PyArrow filter for predicate pushdown
        filters = self._build_filters(start_date, end_date)

        frames: list[pd.DataFrame] = []
        for pf in parquet_files:
            table = pq.read_table(pf, columns=columns, filters=filters)
            frames.append(table.to_pandas())

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return result

    def _build_filters(
        self, start_date: str | None, end_date: str | None
    ) -> list[tuple[str, str, object]] | None:
        """Build PyArrow filter expressions for predicate pushdown."""
        if not start_date and not end_date:
            return None
        filters: list[tuple[str, str, object]] = []
        dc = self.config.date_column
        if start_date:
            filters.append((dc, ">=", pd.Timestamp(start_date)))
        if end_date:
            filters.append((dc, "<=", pd.Timestamp(end_date)))
        return filters

    # ------------------------------------------------------------------
    # Query / check existence
    # ------------------------------------------------------------------

    def has_data(
        self,
        asset_class: str,
        ticker: str,
        year: int | None = None,
    ) -> bool:
        """Check whether data exists for a given partition."""
        if year is not None:
            return self._year_path(asset_class, ticker, year).exists()
        part_dir = self.config.root_dir / asset_class / ticker
        if not part_dir.exists():
            return False
        return any(part_dir.glob("*.parquet"))

    def list_tickers(self, asset_class: str) -> list[str]:
        """List all tickers stored under an asset class."""
        ac_dir = self.config.root_dir / asset_class
        if not ac_dir.exists():
            return []
        return sorted(d.name for d in ac_dir.iterdir() if d.is_dir())

    def list_years(self, asset_class: str, ticker: str) -> list[int]:
        """List available years for a ticker."""
        part_dir = self.config.root_dir / asset_class / ticker
        if not part_dir.exists():
            return []
        return sorted(int(f.stem) for f in part_dir.glob("*.parquet"))

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self, asset_class: str, ticker: str) -> int:
        """Merge small fragment files within a partition.

        If a year's directory contains multiple parquet fragment files
        (from incremental writes stored outside the year-naming scheme),
        this method consolidates them.

        Returns the number of files merged.
        """
        part_dir = self.config.root_dir / asset_class / ticker
        if not part_dir.exists():
            return 0

        all_files = list(part_dir.glob("*.parquet"))
        if len(all_files) <= self.config.compaction_threshold:
            return 0

        # Read all, deduplicate, re-partition by year
        frames = [pd.read_parquet(f) for f in all_files]
        combined = pd.concat(frames, ignore_index=True)
        combined = self._deduplicate(combined)
        combined[self.config.date_column] = pd.to_datetime(
            combined[self.config.date_column]
        )

        # Remove old files
        removed = 0
        for f in all_files:
            f.unlink()
            removed += 1

        # Write consolidated files
        for year, group in combined.groupby(
            combined[self.config.date_column].dt.year
        ):
            path = self._year_path(asset_class, ticker, int(year))
            table = pa.Table.from_pandas(group, preserve_index=False)
            pq.write_table(table, path)

        logger.info(
            "Compacted %s/%s: merged %d files", asset_class, ticker, removed
        )
        return removed

    def auto_compact(self, asset_class: str | None = None) -> int:
        """Run compaction on all partitions that exceed the threshold.

        Parameters
        ----------
        asset_class : str | None
            If given, only compact partitions under this asset class.

        Returns
        -------
        int
            Total number of files merged.
        """
        total = 0
        root = self.config.root_dir
        if asset_class:
            ac_dirs = [root / asset_class]
        else:
            ac_dirs = [d for d in root.iterdir() if d.is_dir()]

        for ac_dir in ac_dirs:
            if not ac_dir.exists():
                continue
            for ticker_dir in ac_dir.iterdir():
                if not ticker_dir.is_dir():
                    continue
                total += self.compact(ac_dir.name, ticker_dir.name)
        return total

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        asset_class: str,
        ticker: str,
        year: int | None = None,
    ) -> int:
        """Delete stored data.

        Parameters
        ----------
        asset_class : str
            Asset class partition.
        ticker : str
            Ticker to delete.
        year : int | None
            If given, delete only that year. Otherwise delete all years.

        Returns
        -------
        int
            Number of files deleted.
        """
        if year is not None:
            path = self._year_path(asset_class, ticker, year)
            if path.exists():
                path.unlink()
                return 1
            return 0

        part_dir = self.config.root_dir / asset_class / ticker
        if not part_dir.exists():
            return 0
        count = 0
        for f in part_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        # Clean up empty dirs
        with contextlib.suppress(OSError):
            part_dir.rmdir()
        return count


# ---------------------------------------------------------------------------
# Cache integration helper
# ---------------------------------------------------------------------------


def warehouse_or_fetch(
    warehouse: ParquetWarehouse,
    asset_class: str,
    ticker: str,
    start_date: str,
    end_date: str,
    fetch_fn: callable,  # type: ignore[valid-type]
) -> pd.DataFrame:
    """Check warehouse first; if data missing, call fetch_fn and store.

    Parameters
    ----------
    warehouse : ParquetWarehouse
        Warehouse instance.
    asset_class, ticker, start_date, end_date : str
        Query parameters.
    fetch_fn : callable
        ``fetch_fn(ticker, start_date, end_date) -> pd.DataFrame``

    Returns
    -------
    pd.DataFrame
        Market data.
    """
    df = warehouse.read(asset_class, ticker, start_date, end_date)
    if not df.empty:
        logger.debug("Warehouse hit for %s/%s", asset_class, ticker)
        return df

    logger.debug("Warehouse miss for %s/%s, fetching", asset_class, ticker)
    df = fetch_fn(ticker, start_date, end_date)
    if not df.empty:
        warehouse.write(df, asset_class, ticker)
    return df
