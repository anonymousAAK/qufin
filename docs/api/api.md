# REST API

## Server

::: qufin.api.server
    options:
      members:
        - create_app
        - OptimizeRequest
        - PriceRequest
        - RiskRequest

## Job Queue

::: qufin.api.jobs
    options:
      members:
        - JobQueue
        - JobType
        - JobPriority
        - JobMeta

## Result Cache

::: qufin.api.cache
    options:
      members:
        - make_cache_key
        - CacheTTL
        - CacheStats
        - SQLiteCacheBackend
        - RedisCacheBackend
        - create_cache
