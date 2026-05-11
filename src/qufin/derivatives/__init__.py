"""Exotic derivatives: basket, path-dependent, Bermudan LSM, autocallable."""

from __future__ import annotations

from qufin.derivatives.autocallable import (
    AutocallableSpec,
    autocallable_mc,
    resource_estimate_chakrabarti,
)
from qufin.derivatives.basket import (
    BasketOptionSpec,
    BasketResult,
    basket_mc,
    geometric_basket_closed_form,
)
from qufin.derivatives.bermudan_lsm import lsm_price
from qufin.derivatives.path_dependent import (
    LookbackOptionSpec,
    cliquet_mc,
    lookback_mc,
)

__all__ = [
    "AutocallableSpec",
    "BasketOptionSpec",
    "BasketResult",
    "LookbackOptionSpec",
    "autocallable_mc",
    "basket_mc",
    "cliquet_mc",
    "geometric_basket_closed_form",
    "lookback_mc",
    "lsm_price",
    "resource_estimate_chakrabarti",
]
