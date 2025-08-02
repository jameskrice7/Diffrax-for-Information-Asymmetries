import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from finax.modeling.training import rolling_cv, train


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


def test_early_stopping():
    data = [None] * 10

    metrics = []

    def step_fn(batch, lr, step):
        metric = 1.0 / (step + 1)
        metrics.append(metric)
        return metric

    def early_stop(step, metric):
        return metric < 0.3

    last_step, last_metric = train(
        step_fn,
        data,
        steps=10,
        early_stopping=early_stop,
        lr_schedule=lambda s: 1.0,
    )

    assert last_step == 3
    assert pytest.approx(last_metric) == 0.25
    assert len(metrics) == 4
