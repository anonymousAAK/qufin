# Data API

## Synthetic Generators

::: qufin.data.synthetic
    options:
      members:
        - gbm_paths
        - heston_paths
        - merton_jump_paths

## Macro Data (FRED)

::: qufin.data.macro
    options:
      members:
        - FREDProvider

## Bloomberg

::: qufin.data.bloomberg
    options:
      members:
        - BloombergDataSource
        - BloombergConfig
        - BloombergSession
        - normalize_fields
        - normalize_dataframe

## Refinitiv / LSEG

::: qufin.data.refinitiv
    options:
      members:
        - RefinitivDataSource
        - RefinitivConfig
        - TimeSeriesResult
        - SnapshotResult

## Real-Time Streaming

::: qufin.data.streaming
    options:
      members:
        - PriceStream
        - StreamConfig
        - PriceBuffer
        - LatencyMonitor
        - PortfolioTracker
        - RebalanceTrigger
        - RebalanceConfig

## Parquet Data Warehouse

::: qufin.data.warehouse
    options:
      members:
        - ParquetWarehouse
        - WarehouseConfig
        - warehouse_or_fetch

## Data Quality

::: qufin.data.quality
    options:
      members:
        - detect_gaps
        - detect_outliers
        - adjust_for_splits
        - adjust_for_dividends
        - compute_quality_score
        - compute_quality_scores
        - DataLineage
        - GapReport
        - OutlierReport
        - QualityScore
