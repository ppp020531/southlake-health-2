"""Per-field sampling strategies and the dispatcher.

Strategies available per column:
- 'empirical'    : classic sample-from-source + Gaussian noise (default for numeric)
- 'kde'          : kernel density estimation sampling (smoother numeric distributions)
- 'copula'       : joint sample from fitted Gaussian Copula (set at table level,
                   individual columns inherit)
- 'categorical'  : empirical frequency sampling with smoothing (default for cat)
- 'identifier'   : surrogate token generation
- 'dp_laplace'   : apply Laplace DP noise on top of chosen base strategy

This module exposes a strategy dispatcher used by the new enhanced generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kde_sample_numeric(
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """Sample from a Gaussian KDE fitted to the numeric series.

    Uses Silverman's rule of thumb for bandwidth selection — standard in
    statistical practice. Result: smooth, realistic-looking continuous values
    that respect the source distribution shape without copying exact values
    (unlike empirical sampling which can leak individual records).

    Privacy benefit: KDE naturally blurs individual records into a density.
    Fidelity benefit: preserves multimodal distributions better than
    "mean + sigma noise" approaches.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(numeric)
    if n < 2:
        # Fall back: repeat available value
        if n == 1:
            return np.full(row_count, numeric[0])
        return np.full(row_count, np.nan)

    # Silverman's rule: h = 1.06 * sigma * n^{-1/5}
    std = float(np.std(numeric, ddof=1))
    if std <= 0:
        return np.full(row_count, float(numeric[0]))
    h = 1.06 * std * (n ** (-1.0 / 5.0)) * bandwidth_scale

    # Sample: pick a random source point + add Gaussian noise with bandwidth h
    base_indices = rng.integers(0, n, size=row_count)
    base_values = numeric[base_indices]
    noise = rng.normal(loc=0.0, scale=h, size=row_count)
    samples = base_values + noise

    # Respect integer-ness: if source is integers, round
    if np.allclose(numeric, np.round(numeric), atol=1e-9):
        samples = np.round(samples)

    return samples


# Strategy identifiers used in metadata per-column overrides
STRATEGY_OPTIONS = [
    "auto",        # agent picks based on semantic role
    "empirical",   # sample source + Gaussian noise
    "kde",         # kernel density estimation
    "copula",      # joint copula (marked at table level; value stored on column)
    "dp_laplace",  # empirical + DP noise
    "identifier",  # surrogate tokens
]


STRATEGY_LABELS = {
    "auto":        "Auto (agent-selected)",
    "empirical":   "Empirical + Gaussian noise",
    "kde":         "Kernel density (smooth)",
    "copula":      "Joint copula (multivariate)",
    "dp_laplace":  "Differential privacy (Laplace)",
    "identifier":  "Surrogate token (new ID)",
}


STRATEGY_DESCRIPTIONS = {
    "auto":        "Let the agent choose based on field type and role.",
    "empirical":   "Sample from source values with calibrated Gaussian noise. Fast and faithful.",
    "kde":         "Smooth the distribution via Silverman's rule kernel density. Reduces record-level leakage.",
    "copula":      "Preserves joint relationships between fields via Gaussian Copula. Best for multivariate fidelity.",
    "dp_laplace":  "Inject Laplace noise calibrated by the privacy budget ε. Strongest privacy guarantee.",
    "identifier":  "Generate new surrogate tokens (e.g., ENC-00001). Source IDs never leave.",
}
