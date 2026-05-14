"""Tests for qufin.data.warehouse — Parquet data warehouse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qufin.data.warehouse import ParquetWarehouse, WarehouseConfig, warehouse_or_fetch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wh_dir(tmp_path: Path) -> Path:
    return tmp_path / "warehouse"


@pytest.fixture
def warehouse(wh_dir: Path) -> ParquetWarehouse:
    config = WarehouseConfig(root_dir=wh_dir)
    return ParquetWarehouse(config)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small sample DataFrame spanning 2023-2024."""
    dates = pd.date_range("2023-06-01", periods=5, freq="ME").tolist() + pd.date_range(
        "2024-01-01", periods=3, freq="ME"
    ).tolist()
    return pd.DataFrame(
        {
            "date": dates,
            "close": [100 + i for i in range(len(dates))],
            "volume": [1000 * (i + 1) for i in range(len(dates))],
        }
    )


# ---------------------------------------------------------------------------
# WarehouseConfig
# ---------------------------------------------------------------------------


class TestWarehouseConfig:
    def test_defaults(self):
        cfg = WarehouseConfig()
        assert cfg.compaction_threshold == 10
        assert cfg.dedup_keys == ["ticker", "date"]
        assert cfg.date_column == "date"

    def test_custom_root(self, tmp_path: Path):
        cfg = WarehouseConfig(root_dir=tmp_path / "custom")
        assert cfg.root_dir == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


class TestWrite:
    def test_write_creates_year_partitions(self, warehouse, sample_df):
        paths = warehouse.write(sample_df, "equity", "AAPL")
        assert len(paths) == 2  # 2023 and 2024
        for p in paths:
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_write_empty_df(self, warehouse):
        empty = pd.DataFrame(columns=["date", "close"])
        paths = warehouse.write(empty, "equity", "AAPL")
        assert paths == []

    def test_write_adds_ticker_column(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "MSFT")
        df = warehouse.read("equity", "MSFT")
        assert "ticker" in df.columns
        assert (df["ticker"] == "MSFT").all()

    def test_write_with_datetime_index(self, warehouse):
        dates = pd.date_range("2023-03-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [1, 2, 3]}, index=dates)
        df.index.name = "date"
        paths = warehouse.write(df, "equity", "TEST")
        assert len(paths) == 1

    def test_write_deduplicates_on_append(self, warehouse):
        df1 = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01", "2023-01-02"]), "close": [100, 101]}
        )
        df2 = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
                "close": [999, 102],
            }
        )
        warehouse.write(df1, "equity", "DEDUP")
        warehouse.write(df2, "equity", "DEDUP")
        result = warehouse.read("equity", "DEDUP")
        # Jan 2 should be deduplicated (keep last = 999)
        assert len(result) == 3
        jan2 = result[result["date"] == pd.Timestamp("2023-01-02")]
        assert jan2.iloc[0]["close"] == 999

    def test_write_missing_date_column_raises(self, warehouse):
        df = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(ValueError, match="date"):
            warehouse.write(df, "equity", "BAD")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_all(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        df = warehouse.read("equity", "AAPL")
        assert len(df) == len(sample_df)

    def test_read_with_date_filter(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        df = warehouse.read(
            "equity", "AAPL", start_date="2024-01-01", end_date="2024-12-31"
        )
        assert len(df) > 0
        assert all(df["date"] >= pd.Timestamp("2024-01-01"))

    def test_read_specific_columns(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        df = warehouse.read("equity", "AAPL", columns=["date", "close"])
        assert "volume" not in df.columns

    def test_read_nonexistent_ticker(self, warehouse):
        df = warehouse.read("equity", "MISSING")
        assert df.empty

    def test_read_nonexistent_asset_class(self, warehouse):
        df = warehouse.read("crypto", "BTC")
        assert df.empty

    def test_read_year_pruning(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        # Only 2024 data
        df = warehouse.read("equity", "AAPL", start_date="2024-01-01")
        assert len(df) > 0

    def test_read_no_matching_years(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        df = warehouse.read("equity", "AAPL", start_date="2030-01-01")
        assert df.empty


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQuery:
    def test_has_data_true(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        assert warehouse.has_data("equity", "AAPL") is True

    def test_has_data_false(self, warehouse):
        assert warehouse.has_data("equity", "NOPE") is False

    def test_has_data_specific_year(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        assert warehouse.has_data("equity", "AAPL", year=2023) is True
        assert warehouse.has_data("equity", "AAPL", year=2025) is False

    def test_list_tickers(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        warehouse.write(sample_df, "equity", "GOOG")
        tickers = warehouse.list_tickers("equity")
        assert tickers == ["AAPL", "GOOG"]

    def test_list_tickers_empty(self, warehouse):
        assert warehouse.list_tickers("equity") == []

    def test_list_years(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        years = warehouse.list_years("equity", "AAPL")
        assert years == [2023, 2024]


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    def test_compact_merges_fragments(self, warehouse, wh_dir):
        """Create many small fragment files and verify compaction merges them."""
        config = WarehouseConfig(root_dir=wh_dir, compaction_threshold=2)
        wh = ParquetWarehouse(config)

        # Create many fragment files manually
        part_dir = wh_dir / "equity" / "FRAG"
        part_dir.mkdir(parents=True)
        for i in range(5):
            df = pd.DataFrame(
                {
                    "date": [pd.Timestamp(f"2023-01-{i + 1:02d}")],
                    "close": [100 + i],
                    "ticker": ["FRAG"],
                }
            )
            df.to_parquet(part_dir / f"fragment_{i}.parquet")

        files_before = list(part_dir.glob("*.parquet"))
        assert len(files_before) == 5

        merged = wh.compact("equity", "FRAG")
        assert merged == 5

        files_after = list(part_dir.glob("*.parquet"))
        assert len(files_after) == 1  # all in 2023
        assert files_after[0].name == "2023.parquet"

    def test_compact_no_action_below_threshold(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        merged = warehouse.compact("equity", "AAPL")
        assert merged == 0  # only 2 files, threshold is 10

    def test_auto_compact(self, warehouse, wh_dir):
        config = WarehouseConfig(root_dir=wh_dir, compaction_threshold=1)
        wh = ParquetWarehouse(config)

        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "close": [1, 2, 3],
            }
        )
        wh.write(df, "equity", "AUTO")
        # Now we have 1 file for 2023, threshold=1 means compact fires
        # Need > threshold files, add another fragment
        part_dir = wh_dir / "equity" / "AUTO"
        extra = pd.DataFrame(
            {"date": [pd.Timestamp("2023-06-01")], "close": [50], "ticker": ["AUTO"]}
        )
        extra.to_parquet(part_dir / "extra.parquet")

        total = wh.auto_compact("equity")
        assert total > 0

    def test_compact_nonexistent_partition(self, warehouse):
        assert warehouse.compact("equity", "NOPE") == 0


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_all(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        count = warehouse.delete("equity", "AAPL")
        assert count == 2
        assert warehouse.has_data("equity", "AAPL") is False

    def test_delete_specific_year(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        count = warehouse.delete("equity", "AAPL", year=2023)
        assert count == 1
        assert warehouse.has_data("equity", "AAPL", year=2023) is False
        assert warehouse.has_data("equity", "AAPL", year=2024) is True

    def test_delete_nonexistent(self, warehouse):
        assert warehouse.delete("equity", "NOPE") == 0
        assert warehouse.delete("equity", "NOPE", year=2023) == 0


# ---------------------------------------------------------------------------
# warehouse_or_fetch
# ---------------------------------------------------------------------------


class TestWarehouseOrFetch:
    def test_cache_hit(self, warehouse, sample_df):
        warehouse.write(sample_df, "equity", "AAPL")
        fetch_mock = lambda t, s, e: pd.DataFrame()  # noqa: E731
        result = warehouse_or_fetch(
            warehouse, "equity", "AAPL", "2023-01-01", "2024-12-31", fetch_mock
        )
        assert not result.empty

    def test_cache_miss_fetches(self, warehouse):
        fetched = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "close": [10, 20, 30],
            }
        )
        fetch_fn = lambda t, s, e: fetched  # noqa: E731
        result = warehouse_or_fetch(
            warehouse, "equity", "NEW", "2023-01-01", "2023-12-31", fetch_fn
        )
        assert len(result) == 3
        # Verify it was stored
        assert warehouse.has_data("equity", "NEW")

    def test_cache_miss_empty_fetch(self, warehouse):
        fetch_fn = lambda t, s, e: pd.DataFrame()  # noqa: E731
        result = warehouse_or_fetch(
            warehouse, "equity", "EMPTY", "2023-01-01", "2023-12-31", fetch_fn
        )
        assert result.empty
