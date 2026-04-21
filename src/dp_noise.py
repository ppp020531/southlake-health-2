"""Differential-privacy-style noise injection for numeric fields.

Provides Laplace-mechanism noise calibrated by a privacy budget epsilon:
  noise ~ Laplace(0, sensitivity / epsilon)

For synthetic data generation, we apply per-field Laplace noise where
sensitivity is estimated from the field's empirical range. Lower epsilon
means stronger privacy (more noise); higher epsilon means less noise and
more fidelity to the source data.

This is a simplified DP mechanism suitable for adding privacy-preserving
noise to individual values in a synthetic output. It is NOT a full
epsilon-DP release — true DP requires careful accounting across all
released statistics, clipping, and composition theorems.

For a pitch demo: shows a principled privacy knob (epsilon) vs ad-hoc
noise_level, grounded in DP theory from Dwork & Roth (2014).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Privacy budget mapping to friendly preset modes
PRIVACY_PRESETS = {
    "Maximum fidelity": 10.0,    # epsilon=10 -> almost no noise
    "Balanced":         2.0,     # epsilon=2  -> moderate noise (default)
    "Strong privacy":   0.5,     # epsilon=0.5 -> significant noise
    "Maximum privacy":  0.1,     # epsilon=0.1 -> heavy noise (privacy dominates)
}


def epsilon_for_preset(preset: str) -> float:
    """Map a friendly preset name to an epsilon value."""
    return PRIVACY_PRESETS.get(preset, 2.0)


def _laplace_noise(
    size: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from Laplace(0, scale)."""
    if scale <= 0:
        return np.zeros(size)
    # Laplace via inverse-CDF method: -scale * sign(u) * log(1 - 2|u|)
    # NumPy has rng.laplace() directly; prefer it for numerical stability.
    return rng.laplace(loc=0.0, scale=scale, size=size)


def apply_dp_noise_numeric(
    values: np.ndarray,
    epsilon: float,
    sensitivity: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply Laplace noise to a numeric array calibrated by epsilon.

    Parameters
    ----------
    values : np.ndarray
        Numeric values to noise.
    epsilon : float
        Privacy budget. Lower = more privacy, higher = more fidelity.
        Values <= 0 are treated as epsilon=infinity (no noise).
    sensitivity : float or None
        Estimated L1 sensitivity. If None, estimated as (p95 - p5) of values.
    rng : np.random.Generator
        Seeded generator.
    """
    if epsilon <= 0 or values.size == 0:
        return values

    # Practical cap: very large epsilon => virtually no noise
    if epsilon >= 100.0:
        return values

    if sensitivity is None:
        # Estimate sensitivity from central 90% range (robust to outliers)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return values
        sensitivity = float(np.percentile(finite, 95) - np.percentile(finite, 5))
        # Safety floor
        sensitivity = max(sensitivity, 1e-6)

    # DP Laplace mechanism: noise scale = sensitivity / epsilon
    scale = sensitivity / epsilon
    return values + _laplace_noise(values.size, scale, rng)


def estimate_sensitivity(series: pd.Series) -> float:
    """Estimate L1 sensitivity for a numeric column.

    Uses the 5th-95th percentile range, which is more robust to outliers
    than (max - min) and is the convention used in many DP implementations
    for bounded-range queries.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    low = float(numeric.quantile(0.05))
    high = float(numeric.quantile(0.95))
    sensitivity = max(high - low, 1e-6)
    return sensitivity
