"""Latent SDEs: variational inference for stochastic dynamics.

Implements the latent SDE of Li, Wong, Chen & Duvenaud, *Scalable Gradients for
Stochastic Differential Equations* (AISTATS 2020). A prior SDE and an
observation-conditioned posterior SDE **share a diffusion term**, which is what
makes the KL divergence between their path measures tractable via Girsanov's
theorem:

.. math::
    \\mathrm{KL}(q \\| p) = \\mathbb{E}_q \\int_{t_0}^{t_1}
        \\tfrac{1}{2} \\left\\| u(z_s, s) \\right\\|^2 \\mathrm{d}s,
    \\qquad
    u = \\frac{f_q(z, s) - f_p(z, s)}{g(z, s)}.

The practical consequence is that the KL is just another state you integrate
alongside ``z``, which is exactly how it is implemented here: the augmented
state is ``(z, kl)`` and the drift of ``kl`` is ``0.5 * ||u||^2``.

Why this matters for information asymmetry
------------------------------------------
The latent state is an unobserved process inferred from observable data. That
is structurally the same problem as recovering a latent "informed trading
intensity" from observed prices and volumes: the quantity of interest is never
measured directly, only its noisy imprint on prices. The posterior drift is
where the information asymmetry signal lives.
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import Array, Float, PRNGKeyArray
from ..core.paths import ControlPath
from ..core.solve import SolveConfig
from .mlp import VectorFieldMLP

__all__ = ["LatentSDE", "LatentSDEOutput"]


class LatentSDEOutput(eqx.Module):
    """Result of a latent SDE forward pass.

    Attributes
    ----------
    outputs:
        Decoded observations, shape ``(time, output_size)``.
    latents:
        Latent trajectory, shape ``(time, latent_size)``.
    kl:
        Path-wise KL divergence between posterior and prior, a scalar. Add this
        to the reconstruction loss with a weight to form the ELBO.
    """

    outputs: Float[Array, "time output"]
    latents: Float[Array, "time latent"]
    kl: Float[Array, ""]


class LatentSDE(eqx.Module):
    """A latent SDE trained by variational inference.

    The model has four learnable pieces:

    ``context_net``
        A neural CDE field that summarises the observed path into a context
        vector, so the posterior drift can condition on the data.
    ``prior_drift``
        ``f_p(t, z)``: the dynamics believed a priori.
    ``posterior_drift``
        ``f_q(t, z, context)``: the data-conditioned dynamics.
    ``diffusion``
        ``g(t, z)``: shared between prior and posterior, and strictly positive.

    Parameters
    ----------
    latent_size:
        Dimension of the latent state ``z``.
    output_size:
        Dimension of the decoded observation.
    context_size:
        Width of the summary vector fed to the posterior drift.

    Examples
    --------
    >>> import jax.numpy as jnp, jax.random as jr, equinox as eqx
    >>> from finax.core import build_control_path, SolveConfig
    >>> ts = jnp.linspace(0.0, 1.0, 8)
    >>> ys = jnp.sin(ts)[:, None]
    >>> path = build_control_path(ts, ys)
    >>> model = LatentSDE.from_hyperparameters(
    ...     input_size=path.n_channels, latent_size=4, output_size=1,
    ...     context_size=4, width=16, key=jr.PRNGKey(0),
    ...     config=SolveConfig(dt0=0.02))
    >>> out = model(path, ts=ts, key=jr.PRNGKey(1))
    >>> out.outputs.shape, out.latents.shape
    ((8, 1), (8, 4))

    The KL is a non-negative scalar, and gradients reach the prior drift:

    >>> bool(out.kl >= 0.0)
    True
    >>> g = eqx.filter_grad(lambda m: m(path, ts=ts, key=jr.PRNGKey(1)).kl)(model)
    >>> bool(jnp.any(g.prior_drift.mlp.layers[0].weight != 0))
    True
    """

    context_net: eqx.Module
    context_initial: eqx.nn.Linear
    prior_drift: eqx.Module
    posterior_drift: eqx.Module
    diffusion_net: eqx.Module
    decoder: eqx.nn.Linear
    initial_latent: eqx.nn.Linear
    config: SolveConfig
    latent_size: int = eqx.field(static=True)
    context_size: int = eqx.field(static=True)
    diffusion_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        context_net,
        context_initial,
        prior_drift,
        posterior_drift,
        diffusion_net,
        decoder,
        initial_latent,
        latent_size: int,
        context_size: int,
        diffusion_floor: float = 1e-3,
        config: SolveConfig | None = None,
    ):
        self.context_net = context_net
        self.context_initial = context_initial
        self.prior_drift = prior_drift
        self.posterior_drift = posterior_drift
        self.diffusion_net = diffusion_net
        self.decoder = decoder
        self.initial_latent = initial_latent
        self.latent_size = latent_size
        self.context_size = context_size
        self.diffusion_floor = diffusion_floor
        self.config = config if config is not None else SolveConfig()

    @classmethod
    def from_hyperparameters(
        cls,
        *,
        input_size: int,
        latent_size: int,
        output_size: int,
        context_size: int = 8,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
        rank: int | None = None,
        diffusion_floor: float = 1e-3,
        config: SolveConfig | None = None,
    ) -> LatentSDE:
        """Build a latent SDE with freshly-initialised components."""
        from .mlp import LowRankTensorField, TensorFieldMLP

        keys = jax.random.split(key, 7)

        if rank is None:
            context_net: eqx.Module = TensorFieldMLP(
                state_size=context_size,
                control_size=input_size,
                width=width,
                depth=depth,
                key=keys[0],
            )
        else:
            context_net = LowRankTensorField(
                state_size=context_size,
                control_size=input_size,
                rank=rank,
                width=width,
                depth=depth,
                key=keys[0],
            )

        return cls(
            context_net=context_net,
            context_initial=eqx.nn.Linear(input_size, context_size, key=keys[1]),
            prior_drift=VectorFieldMLP(
                in_size=latent_size,
                out_size=latent_size,
                width=width,
                depth=depth,
                key=keys[2],
            ),
            posterior_drift=VectorFieldMLP(
                in_size=latent_size + context_size,
                out_size=latent_size,
                width=width,
                depth=depth,
                key=keys[3],
            ),
            diffusion_net=VectorFieldMLP(
                in_size=latent_size,
                out_size=latent_size,
                width=width,
                depth=depth,
                key=keys[4],
            ),
            decoder=eqx.nn.Linear(latent_size, output_size, key=keys[5]),
            initial_latent=eqx.nn.Linear(context_size, latent_size, key=keys[6]),
            latent_size=latent_size,
            context_size=context_size,
            diffusion_floor=diffusion_floor,
            config=config,
        )

    # -- Components --------------------------------------------------------

    def _diffusion(self, t, z):
        """Strictly positive diagonal diffusion, shared by prior and posterior."""
        return jax.nn.softplus(self.diffusion_net(t, z, None)) + self.diffusion_floor

    def encode(
        self, path: ControlPath, *, config: SolveConfig | None = None
    ) -> diffrax.DenseInterpolation:
        """Summarise the observed path into a continuously-queryable context.

        Solved as a neural CDE with a dense output so the posterior drift can
        read the context at any ``t`` the SDE solver happens to land on -- which
        matters because the SDE and the context CDE do not share a step grid.
        """
        cfg = config if config is not None else self.config
        c0 = self.context_initial(path.evaluate(path.t0))
        term = diffrax.ControlTerm(
            lambda t, y, args: self.context_net(t, y, args), path.interpolation
        ).to_ode()
        sol = diffrax.diffeqsolve(
            term,
            cfg.solver if cfg.solver is not None else diffrax.Tsit5(),
            t0=path.t0,
            t1=path.t1,
            dt0=cfg.dt0,
            y0=c0,
            saveat=diffrax.SaveAt(dense=True),
            stepsize_controller=cfg.stepsize_controller,
            adjoint=cfg.adjoint,
            max_steps=cfg.max_steps,
            throw=cfg.throw,
        )
        return sol.evaluate

    # -- Forward pass ------------------------------------------------------

    def __call__(
        self,
        path: ControlPath,
        *,
        key: PRNGKeyArray,
        ts: Float[Array, " time"] | None = None,
        config: SolveConfig | None = None,
    ) -> LatentSDEOutput:
        """Run the posterior SDE and return decoded outputs, latents and the KL.

        Parameters
        ----------
        path:
            Control path built from the observations.
        key:
            PRNG key for the Brownian motion.
        ts:
            Times at which to decode. Defaults to the path endpoints.
        """
        cfg = config if config is not None else self.config
        context = self.encode(path, config=cfg)

        if ts is None:
            ts = jnp.stack([path.t0, path.t1])
        ts = jnp.asarray(ts)

        z0 = self.initial_latent(context(path.t0))
        # Augmented state of size latent_size + 1: the latent z, plus a trailing
        # scalar accumulating the KL integral. Kept as one flat array rather
        # than a (z, kl) tuple so that the control contraction stays a plain
        # matrix-vector product.
        n = self.latent_size
        y0 = jnp.concatenate([z0, jnp.zeros((1,), z0.dtype)])

        def drift(t, y, args):
            z = y[:n]
            c = context(t)
            f_q = self.posterior_drift(t, jnp.concatenate([z, c]), None)
            f_p = self.prior_drift(t, z, None)
            g = self._diffusion(t, z)
            u = (f_q - f_p) / g
            return jnp.concatenate([f_q, 0.5 * jnp.sum(u**2)[None]])

        def diffusion(t, y, args):
            z = y[:n]
            # (latent + 1, latent): diagonal diffusion on z, and a zero final
            # row because the KL accumulator is driven only by dt.
            return jnp.concatenate(
                [jnp.diag(self._diffusion(t, z)), jnp.zeros((1, n), y.dtype)]
            )

        brownian = diffrax.VirtualBrownianTree(
            t0=path.t0,
            t1=path.t1,
            tol=(cfg.dt0 / 10.0) if cfg.dt0 is not None else 1e-3,
            shape=(n,),
            key=key,
            levy_area=diffrax.SpaceTimeLevyArea,
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, brownian),
        )
        sol = diffrax.diffeqsolve(
            terms,
            cfg.solver if cfg.solver is not None else diffrax.Euler(),
            t0=path.t0,
            t1=path.t1,
            dt0=cfg.dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=cfg.stepsize_controller,
            adjoint=cfg.adjoint,
            max_steps=cfg.max_steps,
            throw=cfg.throw,
        )
        latents = sol.ys[:, :n]
        return LatentSDEOutput(
            outputs=jax.vmap(self.decoder)(latents),
            latents=latents,
            kl=sol.ys[-1, n],
        )

    def sample_prior(
        self,
        z0: Float[Array, " latent"],
        t0: float,
        t1: float,
        *,
        key: PRNGKeyArray,
        ts: Float[Array, " time"] | None = None,
        config: SolveConfig | None = None,
    ) -> Float[Array, "time output"]:
        """Sample from the *prior* SDE, ignoring any data.

        This is how you generate unconditional trajectories from a trained model
        -- the generative counterpart to the data-conditioned posterior.
        """
        cfg = config if config is not None else self.config
        from .sde import NeuralSDE

        prior = NeuralSDE(
            lambda t, z, args: self.prior_drift(t, z, None),
            lambda t, z, args: self._diffusion(t, z),
            noise_type="diagonal",
            config=cfg,
        )
        zs = prior(z0, t0, t1, key=key, ts=ts)
        if ts is None:
            return self.decoder(zs)
        return jax.vmap(self.decoder)(zs)
