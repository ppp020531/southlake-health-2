"""Gaussian Copula for multivariate synthetic data generation.

A Gaussian Copula preserves joint relationships (correlations) between fields
while maintaining arbitrary marginal distributions. The algorithm:

1. For each column, transform values to rank-based uniform u = rank(x) / (N+1)
2. Apply standard normal quantile: z = Phi^-1(u)  -> all columns are N(0, 1)
3. Estimate correlation matrix Sigma from these z-scores
4. Sample multivariate normal with Sigma
5. Transform each sampled z back via CDF (rank-based inverse)

This preserves BOTH marginal distributions AND inter-column correlations,
which simple per-column sampling cannot do.

References: Sklar's theorem (1959). Modern implementations in Synthea,
Mostly AI, and SDV (Synthetic Data Vault) libraries.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Standard normal quantile / CDF without scipy dependency
# Using Abramowitz & Stegun approximations (accurate to ~1e-7)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF: Phi(x). Clips extreme values."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    """Standard normal quantile / inverse CDF: Phi^-1(u).

    Uses Peter Acklam's rational approximation (widely-used, accurate).
    """
    u = np.asarray(u, dtype=float)
    u = np.clip(u, 1e-10, 1 - 1e-10)  # avoid +/-inf

    # Coefficients for Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    out = np.empty_like(u)
    lower = u < 0.02425
    upper = u > 0.97575
    mid = ~(lower | upper)

    # Lower tail
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(u[lower]))
        out[lower] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                     ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # Upper tail
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - u[upper]))
        out[upper] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                     ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # Central region (rational approx)
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    return out


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 approximation of erf."""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def _empirical_cdf_values(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted values + their empirical CDF points.

    Used for inverse-CDF transform back to data space.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return np.array([]), np.array([])
    sorted_values = np.sort(values)
    cdf = (np.arange(1, len(sorted_values) + 1)) / (len(sorted_values) + 1)
    return sorted_values, cdf


def _to_normal_scores(series: pd.Series) -> np.ndarray:
    """Rank-based normal scores: transform each value to its standard normal equivalent.

    This is the heart of the Gaussian Copula transformation. Each data point x
    becomes z = Phi^-1(rank(x) / (N+1)), making all columns standard normal
    regardless of original distribution.
    """
    values = pd.to_numeric(series, errors="coerce")
    n = len(values)
    if n == 0:
        return np.array([])

    # Average-rank-based (handles ties): convert to ranks in [1, N]
    ranks = values.rank(method="average", na_option="bottom").to_numpy()
    u = ranks / (n + 1)
    return _norm_ppf(u)


def fit_copula(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, Any]:
    """Fit a Gaussian Copula to the given columns.

    Returns a dict with:
    - 'correlation_matrix': Sigma (k x k)
    - 'marginals': dict[column -> (sorted_values, empirical_cdf)] for inverse transform
    - 'columns': the fitted column names
    """
    if not columns:
        return {"correlation_matrix": np.array([]), "marginals": {}, "columns": []}

    # Transform each column to normal scores
    z_columns: dict[str, np.ndarray] = {}
    marginals: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.dropna().empty:
            continue
        # Fill NaN with median for rank transformation (NaN-ranks are pushed to end)
        z_columns[col] = _to_normal_scores(series)
        marginals[col] = _empirical_cdf_values(series)

    if not z_columns:
        return {"correlation_matrix": np.array([]), "marginals": {}, "columns": []}

    used_columns = list(z_columns.keys())
    # Stack into (N, k) matrix of z-scores
    z_matrix = np.column_stack([z_columns[c] for c in used_columns])

    # Compute Pearson correlation of z-scores (which is Spearman of originals)
    # Handle degenerate cases
    try:
        corr = np.corrcoef(z_matrix, rowvar=False)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        # Replace any NaN with 0 (happens for constant columns)
        corr = np.nan_to_num(corr, nan=0.0)
        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr, 1.0)
    except Exception:
        corr = np.eye(len(used_columns))

    return {
        "correlation_matrix": corr,
        "marginals": marginals,
        "columns": used_columns,
    }


def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite matrix.

    Needed because sample correlations may not be strictly PD (esp. with
    small samples or missing data).
    """
    if matrix.size == 0:
        return matrix
    # Symmetrize
    sym = (matrix + matrix.T) / 2
    # Eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(sym)
        # Clip negative eigenvalues to small positive
        eigvals = np.clip(eigvals, 1e-8, None)
        reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-symmetrize for numerical stability
        reconstructed = (reconstructed + reconstructed.T) / 2
        # Normalize to correlation (diagonal = 1)
        d = np.sqrt(np.diag(reconstructed))
        d = np.where(d > 0, d, 1.0)
        reconstructed = reconstructed / np.outer(d, d)
        np.fill_diagonal(reconstructed, 1.0)
        return reconstructed
    except np.linalg.LinAlgError:
        return np.eye(matrix.shape[0])


def sample_copula(
    copula: dict[str, Any],
    row_count: int,
    rng: np.random.Generator,
    correlation_strength: float = 1.0,
) -> pd.DataFrame:
    """Sample from the fitted Gaussian Copula.

    correlation_strength: 0.0 = independent (Sigma=I), 1.0 = full correlation,
                         values between blend toward identity matrix.
    """
    columns = copula.get("columns", [])
    corr = copula.get("correlation_matrix")
    marginals = copula.get("marginals", {})

    if not columns or corr is None or corr.size == 0 or row_count <= 0:
        return pd.DataFrame({c: [pd.NA] * row_count for c in columns})

    k = len(columns)
    # Blend correlation toward identity based on strength
    blended = correlation_strength * corr + (1 - correlation_strength) * np.eye(k)
    blended = _nearest_positive_definite(blended)

    # Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(blended)
    except np.linalg.LinAlgError:
        # Fallback: use square root via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(blended)
        eigvals = np.clip(eigvals, 1e-8, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # Sample multivariate standard normal
    standard_normal = rng.standard_normal(size=(row_count, k))
    z_samples = standard_normal @ L.T  # (row_count, k) with correlation

    # Transform back to data space via inverse empirical CDF
    output: dict[str, np.ndarray] = {}
    for idx, col in enumerate(columns):
        z = z_samples[:, idx]
        u = _norm_cdf(z)
        sorted_values, cdf = marginals.get(col, (np.array([]), np.array([])))
        if sorted_values.size == 0:
            output[col] = np.full(row_count, np.nan)
            continue
        # Interpolate: find sorted_values[i] such that cdf[i] ~ u
        output[col] = np.interp(u, cdf, sorted_values)

    return pd.DataFrame(output)
