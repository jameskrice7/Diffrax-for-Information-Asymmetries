"""End-to-end: from raw trades to a PIN-conditioned neural SDE.

Runnable as-is (it simulates its own data), and structured so that swapping the
`simulate_trades` call for a real TAQ or Refinitiv extract is the only change
needed to run it on live data.

The pipeline:

1. Simulate a market where informed trading intensity varies over time, giving
   already-classified daily buy and sell counts. With real trade-level data you
   would first run :func:`finax.microstructure.lee_ready` and
   :func:`~finax.microstructure.aggregate_daily_counts` to produce these.
2. Estimate PIN per firm-period, for the whole cross-section at once.
3. Compute complementary liquidity measures.
4. Fit a neural SDE to prices, using the estimated PIN as a conditioning input.
5. Verify the fitted model with the diagnostics module.

Run with::

    python examples/information_asymmetry_pipeline.py
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from finax.core import SolveConfig
from finax.diagnostics import moment_report
from finax.inference import fit, mse
from finax.microstructure import (
    amihud_illiquidity,
    estimate_pin_panel,
    kyle_lambda,
    roll_spread,
)
from finax.models import NeuralSDE
from finax.processes import GeometricBrownianMotion

N_FIRMS = 12
N_PERIODS = 20
DAYS_PER_PERIOD = 60


def simulate_trades(key):
    """Simulate daily buy/sell counts for a panel with time-varying informed flow.

    Replace this with your own data loader. The contract is just two arrays of
    shape ``(firm, period, day)`` holding buyer- and seller-initiated counts.
    """
    k_alpha, k_event, k_side, k_buy, k_sell = jr.split(key, 5)

    # Each firm-period has its own information-event probability.
    alpha = jr.uniform(k_alpha, (N_FIRMS, N_PERIODS), minval=0.05, maxval=0.55)
    eps_b = eps_s = 80.0
    mu = 110.0

    shape = (N_FIRMS, N_PERIODS, DAYS_PER_PERIOD)
    event = jr.bernoulli(k_event, alpha[..., None], shape)
    bad = jr.bernoulli(k_side, 0.5, shape)

    buys = jr.poisson(k_buy, eps_b + mu * (event & ~bad)).astype(jnp.float32)
    sells = jr.poisson(k_sell, eps_s + mu * (event & bad)).astype(jnp.float32)

    true_pin = alpha * mu / (alpha * mu + eps_b + eps_s)
    return buys, sells, true_pin


def main() -> None:
    key = jr.PRNGKey(0)
    k_data, k_price, k_model, k_fit, k_check = jr.split(key, 5)

    # -- 1. Data -----------------------------------------------------------
    buys, sells, true_pin = simulate_trades(k_data)
    print(f"Panel: {N_FIRMS} firms x {N_PERIODS} periods x {DAYS_PER_PERIOD} days")

    # -- 2. PIN for every firm-period, as one batched solve ----------------
    flat_buys = buys.reshape(-1, DAYS_PER_PERIOD)
    flat_sells = sells.reshape(-1, DAYS_PER_PERIOD)

    result = estimate_pin_panel(flat_buys, flat_sells, steps=400)
    pin = result.pin.reshape(N_FIRMS, N_PERIODS)

    error = jnp.abs(pin - true_pin)
    print(f"\nPIN estimated for {pin.size} firm-periods in one vectorised solve")
    print(f"  mean |error| vs truth : {float(jnp.mean(error)):.4f}")
    print(f"  max  |error| vs truth : {float(jnp.max(error)):.4f}")
    print(f"  boundary solutions    : {int(jnp.sum(result.at_boundary))}")

    # -- 3. Complementary liquidity measures -------------------------------
    prices = GeometricBrownianMotion(mu=0.05, sigma=0.25).sample(
        jnp.array(100.0),
        ts=jnp.linspace(0, N_PERIODS, N_PERIODS * DAYS_PER_PERIOD),
        key=k_price,
        n_paths=N_FIRMS,
    )
    returns = jnp.diff(jnp.log(prices), axis=1)
    dollar_volume = (buys + sells).reshape(N_FIRMS, -1)[:, 1:] * prices[:, 1:]

    illiq = jax.vmap(amihud_illiquidity)(returns, dollar_volume)
    signed_flow = (buys - sells).reshape(N_FIRMS, -1)[:, 1:]
    lam = jax.vmap(kyle_lambda)(jnp.diff(prices, axis=1), signed_flow)
    spread = jax.vmap(roll_spread)(prices)

    print("\nLiquidity measures (cross-sectional means):")
    print(f"  Amihud illiquidity : {float(jnp.mean(illiq)):.3e}")
    print(f"  Kyle's lambda      : {float(jnp.mean(lam)):.3e}")
    print(f"  Roll spread        : {float(jnp.mean(spread)):.4f}")

    # -- 4. Neural SDE conditioned on estimated asymmetry ------------------
    # The state is (log price, PIN). Carrying PIN as a state coordinate lets the
    # learned drift and diffusion depend on the estimated information asymmetry.
    period_pin = jnp.mean(pin, axis=0)
    log_prices = jnp.log(prices).reshape(N_FIRMS, N_PERIODS, DAYS_PER_PERIOD)

    y0 = jnp.stack(
        [log_prices[:, :-1, 0].reshape(-1), jnp.tile(period_pin[:-1], N_FIRMS)],
        axis=-1,
    )
    target = log_prices[:, 1:, 0].reshape(-1, 1)

    model = NeuralSDE.from_hyperparameters(
        state_size=2,
        width=32,
        depth=2,
        key=k_model,
        config=SolveConfig(dt0=0.05),
    )

    def loss(m, y0_batch, target_batch):
        keys = jr.split(k_fit, y0_batch.shape[0])
        terminal = jax.vmap(lambda y, k: m(y, 0.0, 1.0, key=k))(y0_batch, keys)
        return mse(terminal[:, :1], target_batch)

    print("\nTraining neural SDE conditioned on estimated PIN...")
    fitted = fit(
        model,
        loss,
        (y0, target),
        steps=150,
        batch_size=32,
        key=k_fit,
        learning_rate=1e-2,
    )
    print(
        f"  loss {float(fitted.train_losses[0]):.5f}"
        f" -> {float(fitted.train_losses[-1]):.5f}"
    )

    # -- 5. Verify the fitted model ----------------------------------------
    sample = fitted.model.sample(y0[0], 0.0, 1.0, key=k_check, n_paths=2000)
    report = moment_report(sample[:, 0])
    print("\nTerminal log-price distribution of the fitted model:")
    print(
        f"  mean {report['mean']['sample']:.4f}"
        f"  sd {report['variance']['sample'] ** 0.5:.4f}"
        f"  kurtosis {report['kurtosis']['sample']:.3f}"
    )

    params = sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(fitted.model, eqx.is_inexact_array))
    )
    print(f"\nModel parameters: {params:,}")


if __name__ == "__main__":
    main()
