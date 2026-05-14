"""Plugin system for qufin: backend discovery, strategy registration, data sources.

Provides entry-point-based backend discovery, decorator-based strategy
registration, and a pluggable data source interface.

Entry Points
-------------
Third-party packages can register backends via the ``qufin.backends``
entry-point group in their ``pyproject.toml``::

    [project.entry-points."qufin.backends"]
    my_backend = "my_package.backend:MyBackend"

The backend class must inherit from :class:`qufin.backends.base.Backend`.

Strategy Plugins
-----------------
Custom optimization or pricing strategies can be registered with the
:func:`register_strategy` decorator::

    from qufin.plugins import register_strategy

    @register_strategy("my_optimizer", category="portfolio")
    def my_optimizer(returns, cov, **kwargs):
        # custom optimization logic
        return weights

Data Source Plugins
--------------------
Custom data sources implement :class:`DataSourcePlugin`::

    from qufin.plugins import DataSourcePlugin, register_data_source

    class MyDataSource(DataSourcePlugin):
        name = "my_source"

        def fetch(self, tickers, start, end, **kwargs):
            # return pandas DataFrame
            ...

        def validate(self):
            return True  # check credentials, connectivity, etc.

    register_data_source(MyDataSource())
"""

from __future__ import annotations

import importlib.metadata
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Entry-point group names
BACKEND_EP_GROUP = "qufin.backends"
STRATEGY_EP_GROUP = "qufin.strategies"
DATA_SOURCE_EP_GROUP = "qufin.data_sources"

# ------------------------------------------------------------------
# Plugin metadata
# ------------------------------------------------------------------


@dataclass
class PluginInfo:
    """Metadata describing a discovered plugin.

    Attributes
    ----------
    name : str
        Plugin name (entry-point name or registration name).
    module : str
        Dotted module path where the plugin is defined.
    plugin_type : str
        One of ``"backend"``, ``"strategy"``, ``"data_source"``.
    loaded : bool
        Whether the plugin object has been successfully loaded.
    error : str | None
        Error message if loading failed.
    metadata : dict[str, Any]
        Arbitrary extra metadata.
    """

    name: str
    module: str = ""
    plugin_type: str = ""
    loaded: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Backend discovery via entry points
# ------------------------------------------------------------------


def discover_backends() -> dict[str, PluginInfo]:
    """Discover backends registered via the ``qufin.backends`` entry-point group.

    Returns
    -------
    dict[str, PluginInfo]
        Mapping of backend name to plugin info.
    """
    discovered: dict[str, PluginInfo] = {}
    eps = importlib.metadata.entry_points()

    # Python 3.12+ returns a SelectableGroups; 3.10-3.11 returns dict
    if hasattr(eps, "select"):
        backend_eps = eps.select(group=BACKEND_EP_GROUP)
    else:
        backend_eps = eps.get(BACKEND_EP_GROUP, [])  # type: ignore[assignment]

    for ep in backend_eps:
        info = PluginInfo(
            name=ep.name,
            module=ep.value,
            plugin_type="backend",
        )
        discovered[ep.name] = info

    return discovered


def load_backend(name: str) -> Any:
    """Load and validate a backend from entry points.

    Parameters
    ----------
    name : str
        The entry-point name of the backend.

    Returns
    -------
    type
        The backend class.

    Raises
    ------
    PluginError
        If the backend cannot be found or fails validation.
    """
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        backend_eps = list(eps.select(group=BACKEND_EP_GROUP, name=name))
    else:
        backend_eps = [
            ep for ep in eps.get(BACKEND_EP_GROUP, [])  # type: ignore[union-attr]
            if ep.name == name
        ]

    if not backend_eps:
        raise PluginError(f"Backend plugin '{name}' not found in entry points.")

    ep = backend_eps[0]
    try:
        backend_cls = ep.load()
    except Exception as exc:
        raise PluginError(f"Failed to load backend '{name}': {exc}") from exc

    _validate_backend(backend_cls, name)
    return backend_cls


def _validate_backend(cls: Any, name: str) -> None:
    """Validate that a class looks like a proper Backend.

    Raises
    ------
    PluginError
        If the class does not satisfy the backend interface.
    """
    required_attrs = ("backend_id", "run")
    for attr in required_attrs:
        if not hasattr(cls, attr):
            raise PluginError(
                f"Backend plugin '{name}' is missing required attribute '{attr}'. "
                f"Backends must implement the qufin.backends.base.Backend interface."
            )


# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------

_strategy_registry: dict[str, dict[str, Any]] = {}


def register_strategy(
    name: str,
    category: str = "general",
    description: str = "",
) -> Callable:
    """Decorator to register a custom strategy function.

    Parameters
    ----------
    name : str
        Unique name for the strategy.
    category : str
        Category (e.g. ``"portfolio"``, ``"pricing"``, ``"risk"``).
    description : str
        Human-readable description of the strategy.

    Returns
    -------
    Callable
        Decorator that registers the function and returns it unchanged.

    Examples
    --------
    >>> @register_strategy("momentum_qaoa", category="portfolio",
    ...                    description="QAOA with momentum overlay")
    ... def momentum_qaoa(returns, cov, **kwargs):
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        if name in _strategy_registry:
            logger.warning(
                "Strategy '%s' is already registered; overwriting.", name
            )
        _strategy_registry[name] = {
            "func": func,
            "category": category,
            "description": description,
            "name": name,
        }
        return func

    return decorator


def get_strategy(name: str) -> Callable:
    """Retrieve a registered strategy by name.

    Parameters
    ----------
    name : str
        The registered strategy name.

    Returns
    -------
    Callable
        The strategy function.

    Raises
    ------
    PluginError
        If no strategy with that name is registered.
    """
    if name not in _strategy_registry:
        available = list(_strategy_registry.keys())
        raise PluginError(
            f"Strategy '{name}' is not registered. Available: {available}"
        )
    return _strategy_registry[name]["func"]


def list_strategies(category: str | None = None) -> list[dict[str, Any]]:
    """List registered strategies, optionally filtered by category.

    Parameters
    ----------
    category : str | None
        If given, only return strategies in this category.

    Returns
    -------
    list[dict[str, Any]]
        List of strategy metadata dicts with keys
        ``name``, ``category``, ``description``.
    """
    results = []
    for entry in _strategy_registry.values():
        if category is None or entry["category"] == category:
            results.append({
                "name": entry["name"],
                "category": entry["category"],
                "description": entry["description"],
            })
    return results


def unregister_strategy(name: str) -> None:
    """Remove a strategy from the registry.

    Parameters
    ----------
    name : str
        The strategy name to remove.
    """
    _strategy_registry.pop(name, None)


def clear_strategies() -> None:
    """Remove all registered strategies."""
    _strategy_registry.clear()


# ------------------------------------------------------------------
# Data source plugin interface
# ------------------------------------------------------------------


class DataSourcePlugin(ABC):
    """Abstract base class for data source plugins.

    Subclasses must set a ``name`` attribute and implement ``fetch``
    and ``validate`` methods.

    Examples
    --------
    >>> class CSVSource(DataSourcePlugin):
    ...     name = "csv"
    ...     def fetch(self, tickers, start, end, **kwargs):
    ...         import pandas as pd
    ...         return pd.read_csv(kwargs["path"])
    ...     def validate(self):
    ...         return True
    """

    name: str = ""

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        **kwargs: Any,
    ) -> Any:
        """Fetch data for the given tickers and date range.

        Parameters
        ----------
        tickers : list[str]
            List of ticker symbols.
        start : str
            Start date (ISO format).
        end : str
            End date (ISO format).

        Returns
        -------
        Any
            Typically a pandas DataFrame.
        """

    @abstractmethod
    def validate(self) -> bool:
        """Validate that this data source is properly configured.

        Returns
        -------
        bool
            True if the data source is ready to use.
        """


_data_source_registry: dict[str, DataSourcePlugin] = {}


def register_data_source(source: DataSourcePlugin) -> None:
    """Register a data source plugin instance.

    Parameters
    ----------
    source : DataSourcePlugin
        An instance of a DataSourcePlugin subclass.

    Raises
    ------
    PluginError
        If the source lacks a name or is not a DataSourcePlugin.
    """
    if not isinstance(source, DataSourcePlugin):
        raise PluginError(
            f"Data source must be a DataSourcePlugin instance, got {type(source).__name__}"
        )
    if not source.name:
        raise PluginError("Data source must have a non-empty 'name' attribute.")
    if source.name in _data_source_registry:
        logger.warning(
            "Data source '%s' is already registered; overwriting.", source.name
        )
    _data_source_registry[source.name] = source


def get_data_source(name: str) -> DataSourcePlugin:
    """Retrieve a registered data source by name.

    Parameters
    ----------
    name : str
        The data source name.

    Returns
    -------
    DataSourcePlugin

    Raises
    ------
    PluginError
        If no data source with that name is registered.
    """
    if name not in _data_source_registry:
        available = list(_data_source_registry.keys())
        raise PluginError(
            f"Data source '{name}' is not registered. Available: {available}"
        )
    return _data_source_registry[name]


def list_data_sources() -> list[str]:
    """List names of all registered data sources.

    Returns
    -------
    list[str]
    """
    return list(_data_source_registry.keys())


def unregister_data_source(name: str) -> None:
    """Remove a data source from the registry."""
    _data_source_registry.pop(name, None)


def clear_data_sources() -> None:
    """Remove all registered data sources."""
    _data_source_registry.clear()


# ------------------------------------------------------------------
# Unified discovery
# ------------------------------------------------------------------


def discover_all() -> dict[str, list[PluginInfo]]:
    """Discover all plugins across all entry-point groups.

    Returns
    -------
    dict[str, list[PluginInfo]]
        Mapping of plugin type to list of discovered plugin info.
    """
    result: dict[str, list[PluginInfo]] = {
        "backends": list(discover_backends().values()),
        "strategies": [
            PluginInfo(
                name=s["name"],
                plugin_type="strategy",
                loaded=True,
                metadata={"category": s["category"], "description": s["description"]},
            )
            for s in _strategy_registry.values()
        ],
        "data_sources": [
            PluginInfo(
                name=name,
                plugin_type="data_source",
                loaded=True,
            )
            for name in _data_source_registry
        ],
    }
    return result


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------


class PluginError(Exception):
    """Raised when a plugin cannot be loaded or validated."""
