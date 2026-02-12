import jax.random as jr
import jax.numpy as jnp

from finax.app import AppBlueprint, DataSourceConfig, ModelServiceConfig
from finax.modeling import (
    NSDEConfig,
    NeuralFinancialSDE,
    ProcessSandbox,
    SandboxScenario,
    estimate_parameter_count,
    simulate_jump_diffusion,
)


def test_jump_diffusion_shape_and_positive_prices():
    key = jr.PRNGKey(0)
    prices = simulate_jump_diffusion(key, n_steps=32, dt=1 / 252)
    assert prices.shape == (33,)
    assert jnp.all(prices > 0)


def test_nsde_supports_large_parameter_models():
    cfg = NSDEConfig(state_dim=4, hidden_dims=(128, 128, 128, 128))
    layer_dims = (cfg.state_dim, *cfg.hidden_dims, cfg.state_dim)
    assert estimate_parameter_count(layer_dims) > 10_000

    model = NeuralFinancialSDE(cfg)
    params = model.init(jr.PRNGKey(1))
    traj = model.simulate(params, jnp.ones((4,)), n_steps=8, dt=0.01, key=jr.PRNGKey(2))
    assert traj.shape == (9, 4)


def test_process_sandbox_and_app_blueprint():
    sandbox = ProcessSandbox(seed=123)
    out = sandbox.run_jump_diffusion(SandboxScenario(name="stress", n_steps=16))
    assert out["process"] == "jump_diffusion"
    assert "summary" in out

    blueprint = AppBlueprint(app_name="finax-lab")
    blueprint.register_data_source(DataSourceConfig(name="prices", kind="stream"))
    blueprint.register_model_service(ModelServiceConfig(model_name="nsde", endpoint="/simulate"))
    as_dict = blueprint.to_dict()
    assert as_dict["app_name"] == "finax-lab"
    assert len(as_dict["data_sources"]) == 1
    assert len(as_dict["model_services"]) == 1
