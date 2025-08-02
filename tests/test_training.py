import importlib.util
import pathlib

import jax.numpy as jnp
import optax
import pytest

# Load module directly to avoid importing optional dependencies
module_path = pathlib.Path(__file__).resolve().parents[1] / "finax" / "modeling" / "training.py"
spec = importlib.util.spec_from_file_location("training", module_path)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

rolling_cv = training.rolling_cv
train = training.train


def test_rolling_cv_boundaries():
    data = list(range(10))
    window = 4
    step = 2
    splits = list(rolling_cv(lambda x: x, data, window, step))
    assert splits == [
        (list(range(0, 4)), list(range(4, 6))),
        (list(range(2, 6)), list(range(6, 8))),
        (list(range(4, 8)), list(range(8, 10))),
    ]


def test_train_updates_params():
    params = jnp.array(1.0)

    def loss_fn(p, _):
        return jnp.square(p)

    trained_params, history = train(
        params,
        loss_fn,
        [None],
        steps=20,
        optimizer=optax.sgd(0.1),
        record_history=True,
    )

    assert history[0] > history[-1]
    assert abs(float(trained_params)) < 0.05
