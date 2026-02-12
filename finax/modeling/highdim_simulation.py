"""Simulation tools for high-dimensional and sparse systems."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_sparse_factor_process(
    *,
    steps: int = 250,
    dimensions: int = 50,
    latent_factors: int = 5,
    sparsity: float = 0.85,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate a high-dimensional process driven by sparse factor loadings."""

    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1).")

    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 1.0, size=(steps, latent_factors)).cumsum(axis=0)

    loadings = rng.normal(0.0, 0.4, size=(latent_factors, dimensions))
    mask = rng.random(size=loadings.shape) < sparsity
    loadings[mask] = 0.0

    noise = rng.normal(0.0, 0.1, size=(steps, dimensions))
    values = factors @ loadings + noise
    return pd.DataFrame(values, columns=[f"x_{i}" for i in range(dimensions)])


def summarize_sparse_structure(df: pd.DataFrame) -> dict[str, float]:
    """Compute quick diagnostics for sparse/high-dimensional matrices."""

    numeric = df.select_dtypes(include=[np.number])
    total = float(numeric.size) if numeric.size else 1.0
    zero_ratio = float((numeric == 0).sum().sum()) / total
    corr = numeric.corr().abs()
    if corr.shape[0] <= 1:
        mean_abs_corr = 0.0
    else:
        mask = ~np.eye(corr.shape[0], dtype=bool)
        mean_abs_corr = float(corr.where(mask).stack().mean())
    return {
        "rows": float(numeric.shape[0]),
        "columns": float(numeric.shape[1]),
        "zero_ratio": zero_ratio,
        "mean_abs_corr": mean_abs_corr,
    }
