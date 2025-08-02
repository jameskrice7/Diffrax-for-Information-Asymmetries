# Visualization

Finax provides plotting helpers built on top of Matplotlib and Seaborn to make it easy to
inspect financial time series, training curves, and simulated paths from neural ODE and SDE
models.

## Time Series

```python
from finax.visualization import plot_time_series
plot_time_series(df)
```

## Model Solutions

Both :class:`finax.modeling.NeuralODE` and :class:`finax.modeling.NeuralSDE` return objects
compatible with Diffrax's ``diffeqsolve``. These solutions can be visualized via
``plot_solution``:

```python
solution = model.solve(y0, 0.0, 1.0)
model.plot(solution)
```

## Training History

```python
from finax.visualization import plot_training_history
plot_training_history(losses)
```

Install the visualization extras with:

```bash
pip install finax[visualization]
```
