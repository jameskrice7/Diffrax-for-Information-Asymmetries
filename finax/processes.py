"""Classical stochastic processes with exact sampling where it exists.

Each process is an ``eqx.Module`` exposing:

``drift(t, y, args)`` / ``diffusion(t, y, args)``
    Ito coefficients, so the process can be handed to any Diffrax solver or used
    as the prior in a :class:`~finax.models.LatentSDE`.
``sample(...)``
    Draws paths. Where the transition density is known in closed form this is an
    **exact** sampler with no discretisation error at all -- and it is exact at
    any step size, so you can jump straight to the horizon.
``log_likelihood(...)``
    Exact transition log-likelihood where available, enabling
    :func:`~finax.inference.calibrate.fit_mle`.

Why exact sampling matters
--------------------------
Two reasons beyond speed. First, it removes discretisation bias from Monte Carlo
estimates entirely. Second, it gives :mod:`finax.diagnostics` a ground truth to
measure a numerical solver against -- you cannot verify a solver's convergence
order without an exact reference.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ._typing import Array, Float, PRNGKeyArray

__all__ = [
    "GeometricBrownianMotion",
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "Heston",
    "MertonJumpDiffusion",
]


class GeometricBrownianMotion(eqx.Module):
    """``dS = mu S dt + sigma S dW``: the Black--Scholes asset price.

    Exactly solvable: ``S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)``.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    >>> paths = gbm.sample(jnp.array(100.0), ts=jnp.linspace(0, 1, 253),
    ...                    key=jr.PRNGKey(0), n_paths=20000)
    >>> paths.shape
    (20000, 253)

    The sample mean matches ``S_0 exp(mu t)`` because the sampler is exact:

    >>> bool(abs(float(jnp.mean(paths[:, -1])) - 100 * jnp.exp(0.05)) < 0.5)
    True

    Log-returns are exactly Gaussian with the right variance:

    >>> lr = jnp.log(paths[:, -1] / 100.0)
    >>> bool(abs(float(jnp.std(lr)) - 0.2) < 0.005)
    True
    """

    mu: Float[Array, ""]
    sigma: Float[Array, ""]

    def drift(self, t, y, args=None):
        return self.mu * y

    def diffusion(self, t, y, args=None):
        return self.sigma * y

    def sample(
        self,
        y0: Float[Array, ""],
        *,
        ts: Float[Array, " time"],
        key: PRNGKeyArray,
        n_paths: int = 1,
    ) -> Float[Array, "path time"]:
        """Sample exactly at the times ``ts``."""
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts, prepend=ts[0])
        noise = jax.random.normal(key, (n_paths, ts.shape[0]))
        increments = (self.mu - 0.5 * self.sigma**2) * dts + self.sigma * jnp.sqrt(
            dts
        ) * noise
        return y0 * jnp.exp(jnp.cumsum(increments, axis=1))

    def log_likelihood(
        self, path: Float[Array, " time"], ts: Float[Array, " time"]
    ) -> Float[Array, ""]:
        """Exact log-likelihood of an observed path, from Gaussian log-returns."""
        path = jnp.asarray(path)
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts)
        log_returns = jnp.diff(jnp.log(path))
        mean = (self.mu - 0.5 * self.sigma**2) * dts
        var = jnp.maximum(self.sigma**2 * dts, 1e-12)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * (log_returns - mean) ** 2 / var
        )


class OrnsteinUhlenbeck(eqx.Module):
    """``dX = kappa (theta - X) dt + sigma dW``: mean-reverting Gaussian dynamics.

    Known in finance as the Vasicek short-rate model. Exactly solvable: the
    transition density is Gaussian with

    .. math::
        \\mathbb{E}[X_{t+\\Delta} \\mid X_t]
            = \\theta + (X_t - \\theta) e^{-\\kappa \\Delta}, \\quad
        \\mathrm{Var} = \\frac{\\sigma^2}{2\\kappa}(1 - e^{-2\\kappa\\Delta}).

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> ou = OrnsteinUhlenbeck(kappa=2.0, theta=0.05, sigma=0.1)
    >>> paths = ou.sample(jnp.array(0.5), ts=jnp.linspace(0, 20, 2001),
    ...                   key=jr.PRNGKey(0), n_paths=2000)

    From far away it reverts to ``theta``, and the stationary variance is
    ``sigma^2 / (2 kappa)``:

    >>> bool(abs(float(jnp.mean(paths[:, -1])) - 0.05) < 0.01)
    True
    >>> bool(abs(float(jnp.var(paths[:, -1])) - 0.1**2 / (2 * 2.0)) < 0.001)
    True
    """

    kappa: Float[Array, ""]
    theta: Float[Array, ""]
    sigma: Float[Array, ""]

    def drift(self, t, y, args=None):
        return self.kappa * (self.theta - y)

    def diffusion(self, t, y, args=None):
        return jnp.broadcast_to(jnp.asarray(self.sigma), jnp.shape(y))

    def sample(
        self,
        y0: Float[Array, ""],
        *,
        ts: Float[Array, " time"],
        key: PRNGKeyArray,
        n_paths: int = 1,
    ) -> Float[Array, "path time"]:
        """Sample exactly at the times ``ts``."""
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts, prepend=ts[0])
        noise = jax.random.normal(key, (n_paths, ts.shape[0]))

        decay = jnp.exp(-self.kappa * dts)
        std = jnp.sqrt(
            jnp.maximum(self.sigma**2 / (2 * self.kappa) * (1 - decay**2), 0.0)
        )

        def step(x, inputs):
            d, s, z = inputs
            x_next = self.theta + (x - self.theta) * d + s * z
            return x_next, x_next

        def one_path(z_row):
            _, xs = jax.lax.scan(step, y0, (decay, std, z_row))
            return xs

        return jax.vmap(one_path)(noise)

    def log_likelihood(
        self, path: Float[Array, " time"], ts: Float[Array, " time"]
    ) -> Float[Array, ""]:
        """Exact Gaussian transition log-likelihood."""
        path = jnp.asarray(path)
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts)
        decay = jnp.exp(-self.kappa * dts)
        mean = self.theta + (path[:-1] - self.theta) * decay
        var = jnp.maximum(self.sigma**2 / (2 * self.kappa) * (1 - decay**2), 1e-12)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * (path[1:] - mean) ** 2 / var
        )


class CoxIngersollRoss(eqx.Module):
    """``dX = kappa (theta - X) dt + sigma sqrt(X) dW``: mean-reverting and non-negative.

    The square-root diffusion keeps ``X`` non-negative, which is why CIR is the
    standard model for interest rates and for stochastic variance.

    The **Feller condition** ``2 kappa theta >= sigma^2`` guarantees ``X`` stays
    strictly positive; :attr:`feller_satisfied` reports whether it holds, since a
    violated Feller condition is the usual reason a CIR simulation produces NaNs.

    Sampling uses the exact non-central chi-squared transition law, so it is
    correct even when the Feller condition fails -- where a naive Euler scheme
    would go negative and then produce ``nan`` from ``sqrt``.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> cir = CoxIngersollRoss(kappa=2.0, theta=0.04, sigma=0.15)
    >>> bool(cir.feller_satisfied)
    True
    >>> paths = cir.sample(jnp.array(0.04), ts=jnp.linspace(0, 10, 1001),
    ...                    key=jr.PRNGKey(0), n_paths=2000)
    >>> bool(jnp.all(paths >= 0.0))
    True
    >>> bool(abs(float(jnp.mean(paths[:, -1])) - 0.04) < 0.005)
    True

    It stays non-negative even when Feller is violated:

    >>> rough = CoxIngersollRoss(kappa=0.5, theta=0.02, sigma=0.9)
    >>> bool(rough.feller_satisfied)
    False
    >>> p = rough.sample(jnp.array(0.02), ts=jnp.linspace(0, 5, 501),
    ...                  key=jr.PRNGKey(1), n_paths=500)
    >>> bool(jnp.all(p >= 0.0) and jnp.all(jnp.isfinite(p)))
    True
    """

    kappa: Float[Array, ""]
    theta: Float[Array, ""]
    sigma: Float[Array, ""]

    @property
    def feller_satisfied(self) -> Array:
        """Whether ``2 kappa theta >= sigma^2``, so the process cannot reach zero."""
        return 2 * self.kappa * self.theta >= self.sigma**2

    def drift(self, t, y, args=None):
        return self.kappa * (self.theta - y)

    def diffusion(self, t, y, args=None):
        # Clamp inside the sqrt: a numerical solver can step slightly negative,
        # and sqrt of a negative number would poison the whole path with NaN.
        return self.sigma * jnp.sqrt(jnp.maximum(y, 0.0))

    def sample(
        self,
        y0: Float[Array, ""],
        *,
        ts: Float[Array, " time"],
        key: PRNGKeyArray,
        n_paths: int = 1,
    ) -> Float[Array, "path time"]:
        """Sample exactly using the non-central chi-squared transition law.

        ``X_{t+dt} = c * chi2_ncx(df, lambda)`` with
        ``c = sigma^2 (1 - e^{-kappa dt}) / (4 kappa)``,
        ``df = 4 kappa theta / sigma^2`` and
        ``lambda = X_t e^{-kappa dt} / c``.

        JAX has no non-central chi-squared sampler, so it is built from the
        Poisson-mixture-of-gammas identity

        .. code-block:: text

            N ~ Poisson(lambda / 2),  chi2_ncx(df, lambda) = 2 * Gamma(df/2 + N)

        which, unlike the ``chi2(df-1) + (Z + sqrt(lambda))^2`` construction,
        stays valid for ``df < 1`` -- exactly the regime where the Feller
        condition fails and correctness matters most.
        """
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts, prepend=ts[0])
        keys = jax.random.split(key, ts.shape[0])

        df = 4 * self.kappa * self.theta / self.sigma**2

        def step(x, inputs):
            dt, k = inputs
            k_pois, k_gamma = jax.random.split(k)
            decay = jnp.exp(-self.kappa * dt)
            c = self.sigma**2 * (1 - decay) / (4 * self.kappa)
            # dt == 0 (the prepended first entry) must leave the state untouched.
            c = jnp.where(dt > 0, c, 1.0)
            lam = x * decay / c
            n = jax.random.poisson(k_pois, 0.5 * lam, shape=x.shape)
            draw = 2.0 * c * jax.random.gamma(k_gamma, 0.5 * df + n, shape=x.shape)
            x_next = jnp.where(dt > 0, draw, x)
            return x_next, x_next

        x0 = jnp.full((n_paths,), y0)
        _, xs = jax.lax.scan(step, x0, (dts, keys))
        return jnp.maximum(xs.T, 0.0)


class Heston(eqx.Module):
    """Stochastic volatility: a GBM whose variance follows a CIR process.

    .. math::
        \\mathrm{d}S = \\mu S\\,\\mathrm{d}t + \\sqrt{v} S\\,\\mathrm{d}W^1, \\quad
        \\mathrm{d}v = \\kappa(\\theta - v)\\,\\mathrm{d}t
                       + \\xi\\sqrt{v}\\,\\mathrm{d}W^2, \\quad
        \\mathrm{d}W^1 \\mathrm{d}W^2 = \\rho\\,\\mathrm{d}t.

    The correlation ``rho`` is what produces the volatility skew: with
    ``rho < 0``, falling prices coincide with rising volatility.

    The state is ``(log S, v)``. Working in log-price makes the price
    automatically positive and the drift state-independent.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> h = Heston(mu=0.03, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    >>> log_s, v = h.sample(jnp.array(jnp.log(100.0)), jnp.array(0.04),
    ...                     ts=jnp.linspace(0, 1, 253), key=jr.PRNGKey(0),
    ...                     n_paths=4000)
    >>> log_s.shape, v.shape
    ((4000, 253), (4000, 253))
    >>> bool(jnp.all(v >= 0.0))
    True

    Negative ``rho`` produces negatively skewed returns -- the leverage effect:

    >>> r = log_s[:, -1] - jnp.log(100.0)
    >>> z = (r - jnp.mean(r)) / jnp.std(r)
    >>> bool(jnp.mean(z ** 3) < -0.1)
    True
    """

    mu: Float[Array, ""]
    kappa: Float[Array, ""]
    theta: Float[Array, ""]
    xi: Float[Array, ""]
    rho: Float[Array, ""]

    def drift(self, t, y, args=None):
        """Drift of the state ``(log S, v)``."""
        v = jnp.maximum(y[1], 0.0)
        return jnp.stack([self.mu - 0.5 * v, self.kappa * (self.theta - v)])

    def diffusion(self, t, y, args=None):
        """``(2, 2)`` diffusion matrix mapping two Brownian factors to the state.

        Built from the Cholesky factor of the correlation matrix, so the two
        driving Brownian motions can be taken independent.
        """
        v = jnp.maximum(y[1], 0.0)
        sqrt_v = jnp.sqrt(v)
        return jnp.array(
            [
                [sqrt_v, 0.0],
                [
                    self.xi * sqrt_v * self.rho,
                    self.xi * sqrt_v * jnp.sqrt(jnp.maximum(1 - self.rho**2, 0.0)),
                ],
            ]
        )

    def sample(
        self,
        log_s0: Float[Array, ""],
        v0: Float[Array, ""],
        *,
        ts: Float[Array, " time"],
        key: PRNGKeyArray,
        n_paths: int = 1,
    ) -> tuple[Float[Array, "path time"], Float[Array, "path time"]]:
        """Simulate with a full-truncation Euler scheme for ``v``.

        Full truncation -- clamping ``v`` at zero inside both the drift and the
        diffusion while letting the state itself go negative before the clamp --
        is the discretisation shown by Lord et al. (2010) to have the smallest
        bias among the simple Heston schemes.
        """
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts, prepend=ts[0])
        key_1, key_2 = jax.random.split(key)
        z1 = jax.random.normal(key_1, (n_paths, ts.shape[0]))
        z2 = jax.random.normal(key_2, (n_paths, ts.shape[0]))
        # Correlate the second factor with the first.
        w_v = self.rho * z1 + jnp.sqrt(jnp.maximum(1 - self.rho**2, 0.0)) * z2

        def step(carry, inputs):
            log_s, v = carry
            dt, dz_s, dz_v = inputs
            sqrt_dt = jnp.sqrt(dt)
            v_pos = jnp.maximum(v, 0.0)
            sqrt_v = jnp.sqrt(v_pos)
            log_s_next = log_s + (self.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * dz_s
            v_next = (
                v
                + self.kappa * (self.theta - v_pos) * dt
                + self.xi * sqrt_v * sqrt_dt * dz_v
            )
            return (log_s_next, v_next), (log_s_next, jnp.maximum(v_next, 0.0))

        def one_path(dz_s, dz_v):
            init = (jnp.asarray(log_s0), jnp.asarray(v0))
            _, out = jax.lax.scan(step, init, (dts, dz_s, dz_v))
            return out

        return jax.vmap(one_path)(z1, w_v)


class MertonJumpDiffusion(eqx.Module):
    """GBM plus lognormal jumps: ``dS/S = mu dt + sigma dW + (e^J - 1) dN``.

    Merton (1976). Jump sizes are ``J ~ Normal(jump_mean, jump_std)`` in log
    space and arrive as a Poisson process with intensity ``intensity``. The
    drift is compensated so that ``E[S_t] = S_0 e^{mu t}`` regardless of the
    jump parameters -- jumps add tail risk without changing the expected return.

    Exactly sampled by conditioning on the number of jumps in each interval,
    which is possible because a sum of ``n`` iid normal jumps is itself normal.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr
    >>> m = MertonJumpDiffusion(mu=0.05, sigma=0.2, intensity=1.0,
    ...                         jump_mean=-0.1, jump_std=0.15)
    >>> paths = m.sample(jnp.array(100.0), ts=jnp.linspace(0, 1, 253),
    ...                  key=jr.PRNGKey(0), n_paths=40000)
    >>> bool(abs(float(jnp.mean(paths[:, -1])) - 100 * jnp.exp(0.05)) < 1.0)
    True

    Jumps make returns fat-tailed relative to the pure-diffusion case:

    >>> r = jnp.log(paths[:, -1] / 100.0)
    >>> z = (r - jnp.mean(r)) / jnp.std(r)
    >>> bool(jnp.mean(z ** 4) > 3.5)  # excess kurtosis over the Gaussian's 3
    True
    """

    mu: Float[Array, ""]
    sigma: Float[Array, ""]
    intensity: Float[Array, ""]
    jump_mean: Float[Array, ""]
    jump_std: Float[Array, ""]

    @property
    def compensator(self) -> Float[Array, ""]:
        """``lambda * (E[e^J] - 1)``: the adjustment keeping jumps a martingale."""
        return self.intensity * (jnp.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1.0)

    def drift(self, t, y, args=None):
        return (self.mu - self.compensator) * y

    def diffusion(self, t, y, args=None):
        return self.sigma * y

    def sample(
        self,
        y0: Float[Array, ""],
        *,
        ts: Float[Array, " time"],
        key: PRNGKeyArray,
        n_paths: int = 1,
    ) -> Float[Array, "path time"]:
        """Sample exactly by conditioning on the jump count in each interval."""
        ts = jnp.asarray(ts)
        dts = jnp.diff(ts, prepend=ts[0])
        k_w, k_n, k_j = jax.random.split(key, 3)
        shape = (n_paths, ts.shape[0])

        diffusive = (
            self.mu - self.compensator - 0.5 * self.sigma**2
        ) * dts + self.sigma * jnp.sqrt(dts) * jax.random.normal(k_w, shape)

        counts = jax.random.poisson(k_n, self.intensity * dts, shape).astype(
            diffusive.dtype
        )
        # Sum of `n` iid Normal(m, s^2) jumps is Normal(n*m, n*s^2).
        jumps = counts * self.jump_mean + jnp.sqrt(
            counts
        ) * self.jump_std * jax.random.normal(k_j, shape)

        return y0 * jnp.exp(jnp.cumsum(diffusive + jumps, axis=1))
