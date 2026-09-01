"""Plotting helpers. Requires the ``viz`` extra."""

from .plots import (
    figure_to_base64,
    plot_convergence,
    plot_fan_chart,
    plot_paths,
    plot_solution,
    plot_training_history,
)

__all__ = [
    "plot_paths",
    "plot_fan_chart",
    "plot_training_history",
    "plot_convergence",
    "plot_solution",
    "figure_to_base64",
]
