"""Differentiable market-microstructure estimators.

Standard measures of information asymmetry and illiquidity, implemented in pure
JAX. Every function here is ``jit``-compilable, ``vmap``-able across a
cross-section, and differentiable with respect to its inputs -- which is what
lets an estimated PIN or Kyle's lambda sit *inside* a larger differentiable
model rather than being computed as a fixed preprocessing step.

The intended pipeline:

.. code-block:: text

    raw trades
        |
        |  classification.lee_ready / tick_rule
        v
    signed trades ---> classification.aggregate_daily_counts ---> pin.estimate_pin
        |
        |  vpin.volume_bars
        v
    volume bars ---> vpin.vpin
"""

from .classification import (
    aggregate_daily_counts,
    bulk_volume_classification,
    lee_ready,
    quote_rule,
    tick_rule,
)
from .liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
    effective_spread,
    kyle_lambda,
    price_impact,
    realized_spread,
    roll_spread,
)
from .pin import (
    PINParams,
    PINResult,
    estimate_pin,
    estimate_pin_panel,
    initial_parameter_grid,
    pin_log_likelihood,
)
from .vpin import volume_bars, vpin

__all__ = [
    # Trade classification
    "tick_rule",
    "quote_rule",
    "lee_ready",
    "bulk_volume_classification",
    "aggregate_daily_counts",
    # PIN
    "PINParams",
    "PINResult",
    "pin_log_likelihood",
    "initial_parameter_grid",
    "estimate_pin",
    "estimate_pin_panel",
    # VPIN
    "volume_bars",
    "vpin",
    # Liquidity
    "kyle_lambda",
    "amihud_illiquidity",
    "roll_spread",
    "corwin_schultz_spread",
    "effective_spread",
    "realized_spread",
    "price_impact",
]
