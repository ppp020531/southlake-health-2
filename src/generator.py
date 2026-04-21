from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.copula import fit_copula, sample_copula
from src.dp_noise import apply_dp_noise_numeric, epsilon_for_preset, estimate_sensitivity
from src.constraints import detect_constraints, enforce_constraints
from src.strategies import kde_sample_numeric


def _missing_probability(series: pd.Series) -> float:
    return float(series.isna().mean())


def _generate_identifier(column_name: str, row_count: int) -> list[str]:
    prefix = "".join(part[0] for part in column_name.split("_") if part)[:3].upper() or "SYN"
    return [f"{prefix}-{index:05d}" for index in range(1, row_count + 1)]


def _coarsen_geography(series: pd.Series) -> pd.Series:
    def _coarse(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if not text:
            return pd.NA
        if " " in text:
            return text.split()[0][:3].upper()
        return text[:3].upper()

    return series.map(_coarse)


def _group_text(series: pd.Series, keep_top: int = 8) -> pd.Series:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return pd.Series([pd.NA] * len(series), index=series.index)

    counts = non_null.value_counts()
    keepers = set(counts.head(keep_top).index)
    grouped = series.fillna(pd.NA).astype("string")
    grouped = grouped.map(lambda value: value if pd.isna(value) or str(value).strip() in keepers else "Other")
    return grouped


def _group_rare_categories(series: pd.Series, keep_top: int = 6) -> pd.Series:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return pd.Series([pd.NA] * len(series), index=series.index)

    counts = non_null.value_counts()
    keepers = set(counts.head(keep_top).index)
    grouped = series.fillna(pd.NA).astype("string")
    grouped = grouped.map(lambda value: value if pd.isna(value) or str(value).strip() in keepers else "Other")
    return grouped


def _apply_missingness(
    result: pd.Series,
    source_series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    missingness_pattern: str,
    anchor_series: pd.Series | None = None,
) -> pd.Series:
    output = result.copy()
    missing_probability = _missing_probability(source_series)

    if missingness_pattern == "Fill gaps":
        return output

    if missingness_pattern == "Preserve source pattern" and anchor_series is not None:
        missing_mask = anchor_series.isna().reset_index(drop=True)
    else:
        multiplier = 1.0 if missingness_pattern == "Preserve source pattern" else 0.55
        missing_mask = pd.Series(rng.random(row_count) < min(missing_probability * multiplier, 0.98))

    return output.mask(missing_mask.to_numpy(), pd.NA)


def _prepare_anchor_output(series: pd.Series, role: str, control_action: str) -> pd.Series:
    if role == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if role == "date":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if control_action == "Month only":
            return parsed.dt.to_period("M").astype(str).replace("NaT", pd.NA)
        return parsed.dt.strftime("%Y-%m-%d").replace("NaT", pd.NA)
    return series.reset_index(drop=True)


def _blend_with_anchor(
    generated: pd.Series,
    anchor_series: pd.Series,
    rng: np.random.Generator,
    correlation_preservation: int,
    locked_distribution: bool,
) -> pd.Series:
    blend_strength = np.interp(correlation_preservation, [0, 100], [0.0, 0.88])
    if locked_distribution:
        blend_strength = min(0.96, blend_strength + 0.12)
    if blend_strength <= 0:
        return generated

    blend_mask = rng.random(len(generated)) < blend_strength
    blended = generated.copy()
    blended.loc[blend_mask] = anchor_series.loc[blend_mask]
    return blended


def _rare_row_weights(df: pd.DataFrame, metadata: list[dict[str, Any]]) -> np.ndarray:
    if df.empty:
        return np.array([])

    weights = np.ones(len(df), dtype=float)
    for item in metadata:
        if not item["include"]:
            continue

        column = item["column"]
        role = item["data_type"]
        series = df[column]

        if role in {"categorical", "binary"}:
            labels = series.fillna("Missing").astype(str)
            probabilities = labels.value_counts(normalize=True)
            rarity = labels.map(lambda value: 1.0 / max(probabilities.get(value, 1.0), 1e-6))
            baseline = max(float(np.median(rarity)), 1.0)
            weights += np.clip(rarity / baseline - 1.0, 0.0, 2.0)
        elif role == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                low = numeric.quantile(0.05)
                high = numeric.quantile(0.95)
                rare_mask = ((numeric < low) | (numeric > high)).fillna(False).to_numpy()
                weights += rare_mask.astype(float) * 0.45

    weights = np.where(np.isfinite(weights), weights, 1.0)
    total = weights.sum()
    return weights / total if total else np.full(len(df), 1 / len(df))


def _sample_categorical(
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    fidelity_priority: int,
    noise_level: int,
    rare_case_retention: int,
    locked_distribution: bool,
) -> pd.Series:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return pd.Series([pd.NA] * row_count)

    probabilities = non_null.value_counts(normalize=True).sort_index()
    smoothing = np.interp(fidelity_priority, [0, 100], [0.35, 0.05])
    smoothing *= np.interp(noise_level, [0, 100], [0.55, 1.55])
    if locked_distribution:
        smoothing *= 0.35
    uniform = np.full(len(probabilities), 1 / len(probabilities))
    smoothed = probabilities.to_numpy() * (1 - smoothing) + uniform * smoothing

    if rare_case_retention > 0 and len(probabilities) > 1:
        rare_threshold = float(probabilities.quantile(0.35))
        rarity_boost = np.where(probabilities.to_numpy() <= rare_threshold, np.interp(rare_case_retention, [0, 100], [1.0, 1.85]), 1.0)
        smoothed = smoothed * rarity_boost

    smoothed = smoothed / smoothed.sum()

    sampled = rng.choice(probabilities.index.to_numpy(), size=row_count, p=smoothed)
    return pd.Series(sampled)


def _sample_numeric(
    column_name: str,
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    fidelity_priority: int,
    noise_level: int,
    locked_distribution: bool,
    outlier_strategy: str,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return pd.Series([pd.NA] * row_count)

    sampled = rng.choice(numeric.to_numpy(), size=row_count, replace=True)
    std = float(numeric.std(ddof=0))
    spread = std if std > 0 else max(float(numeric.max() - numeric.min()) / 6, 1.0)
    noise_scale = np.interp(fidelity_priority, [0, 100], [0.22, 0.04])
    noise_scale *= np.interp(noise_level, [0, 100], [0.45, 1.9])
    if locked_distribution:
        noise_scale *= 0.35
    noise = rng.normal(0, spread * noise_scale, size=row_count)
    synthetic = sampled + noise

    lowered = column_name.lower()
    if outlier_strategy == "Clip extremes":
        lower = float(numeric.quantile(0.05))
        upper = float(numeric.quantile(0.95))
        synthetic = np.clip(synthetic, lower, upper)
    elif outlier_strategy == "Smooth tails":
        low = float(numeric.quantile(0.05))
        high = float(numeric.quantile(0.95))
        lower_mask = synthetic < low
        upper_mask = synthetic > high
        synthetic[lower_mask] = low - (low - synthetic[lower_mask]) * 0.45
        synthetic[upper_mask] = high + (synthetic[upper_mask] - high) * 0.45

    if any(token in lowered for token in ["age", "wait", "stay", "ctas"]):
        synthetic = np.clip(synthetic, 0, None)

    if np.allclose(numeric, np.round(numeric), atol=1e-9):
        synthetic = np.round(synthetic)

    return pd.Series(synthetic)


def _sample_dates(
    series: pd.Series,
    row_count: int,
    rng: np.random.Generator,
    fidelity_priority: int,
    control_action: str,
    noise_level: int,
    locked_distribution: bool,
) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
    if parsed.empty:
        return pd.Series([pd.NA] * row_count)

    sampled = rng.choice(parsed.to_numpy(), size=row_count, replace=True)
    jitter_days = int(round(np.interp(fidelity_priority, [0, 100], [10, 2])))
    jitter_days = max(1, int(round(jitter_days * np.interp(noise_level, [0, 100], [0.65, 1.65]))))
    if locked_distribution:
        jitter_days = max(1, int(round(jitter_days * 0.45)))
    jitter = rng.integers(-jitter_days, jitter_days + 1, size=row_count)
    jittered = pd.Series(pd.to_datetime(sampled) + pd.to_timedelta(jitter, unit="D"))

    if control_action == "Month only":
        result = jittered.dt.to_period("M").astype(str)
    else:
        result = jittered.dt.strftime("%Y-%m-%d")
    return result


def generate_synthetic_data(
    df: pd.DataFrame,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fidelity_priority = int(controls["fidelity_priority"])
    row_count = int(controls["synthetic_rows"])
    valid_columns = {item["column"] for item in metadata if item["include"]}
    locked_columns = {column for column in controls.get("locked_columns", []) if column in valid_columns}
    correlation_preservation = int(controls.get("correlation_preservation", 35))
    rare_case_retention = int(controls.get("rare_case_retention", 30))
    noise_level = int(controls.get("noise_level", 45))
    missingness_pattern = str(controls.get("missingness_pattern", "Preserve source pattern"))
    outlier_strategy = str(controls.get("outlier_strategy", "Preserve tails"))
    generation_preset = str(controls.get("generation_preset", "Balanced"))
    seed = int(controls.get("seed", 42))

    rng = np.random.default_rng(seed)
    synthetic_columns: dict[str, pd.Series] = {}
    excluded_columns: list[str] = []
    uniform_weights = np.full(len(df), 1 / max(len(df), 1))
    rare_weights = _rare_row_weights(df, metadata) if len(df) else np.array([])
    retention_mix = np.interp(rare_case_retention, [0, 100], [0.0, 0.72])
    anchor_weights = uniform_weights if len(df) else np.array([])
    if len(df):
        anchor_weights = uniform_weights * (1 - retention_mix) + rare_weights * retention_mix
        anchor_indices = rng.choice(np.arange(len(df)), size=row_count, replace=True, p=anchor_weights)
    else:
        anchor_indices = np.array([], dtype=int)

    for column_meta in metadata:
        column = column_meta["column"]
        control_action = column_meta.get("control_action", "")
        if not column_meta["include"] or control_action == "Exclude":
            excluded_columns.append(column)
            continue

        role = column_meta["data_type"]
        source_series = df[column]
        working_series = source_series.copy()
        locked_distribution = column in locked_columns

        if control_action == "Coarse geography":
            working_series = _coarsen_geography(working_series)
        elif control_action == "Group text":
            working_series = _group_text(working_series)
        elif control_action == "Group rare categories":
            working_series = _group_rare_categories(working_series)
        elif control_action == "Clip extremes" and role == "numeric":
            numeric_series = pd.to_numeric(working_series, errors="coerce")
            if numeric_series.notna().any():
                lower = numeric_series.quantile(0.05)
                upper = numeric_series.quantile(0.95)
                working_series = numeric_series.clip(lower=lower, upper=upper)

        is_identifier_like = role == "identifier" or column_meta["strategy"] == "new_token"
        if is_identifier_like:
            generated = pd.Series(_generate_identifier(column, row_count))
        elif role == "numeric":
            generated = _sample_numeric(
                column,
                working_series,
                row_count,
                rng,
                fidelity_priority,
                noise_level,
                locked_distribution,
                outlier_strategy,
            )
        elif role == "date":
            generated = _sample_dates(
                working_series,
                row_count,
                rng,
                fidelity_priority,
                control_action,
                noise_level,
                locked_distribution,
            )
        else:
            generated = _sample_categorical(
                working_series,
                row_count,
                rng,
                fidelity_priority,
                noise_level,
                rare_case_retention,
                locked_distribution,
            )

        if not is_identifier_like:
            anchor_source = working_series.iloc[anchor_indices].reset_index(drop=True) if len(anchor_indices) else pd.Series([pd.NA] * row_count)
            anchor_ready = _prepare_anchor_output(anchor_source, role, control_action)
            generated = _blend_with_anchor(generated.reset_index(drop=True), anchor_ready, rng, correlation_preservation, locked_distribution)
            generated = _apply_missingness(generated, working_series, row_count, rng, missingness_pattern, anchor_source)

        synthetic_columns[column] = generated.reset_index(drop=True)

    synthetic_df = pd.DataFrame(synthetic_columns)

    summary = {
        "rows_generated": row_count,
        "columns_generated": len(synthetic_df.columns),
        "excluded_columns": excluded_columns,
        "fidelity_priority": fidelity_priority,
        "locked_columns": sorted(locked_columns),
        "correlation_preservation": correlation_preservation,
        "rare_case_retention": rare_case_retention,
        "noise_level": noise_level,
        "missingness_pattern": missingness_pattern,
        "outlier_strategy": outlier_strategy,
        "generation_preset": generation_preset,
        "noise_mode": "Higher privacy" if fidelity_priority < 40 else "Balanced" if fidelity_priority < 70 else "Higher fidelity",
    }

    return synthetic_df, summary


# ═══════════════════════════════════════════════════════════════════
# ADVANCED GENERATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════
# Dispatches per-field to the requested strategy, applies copula for
# multivariate correlation, enforces detected logical constraints,
# and returns a rich summary for the preview UI.

def generate_synthetic_advanced(
    df: pd.DataFrame,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Advanced synthetic data generation with per-field strategies,
    Gaussian Copula for multivariate joint sampling, differential-privacy
    Laplace noise, and auto-detected constraint enforcement.

    Backward-compatible with generate_synthetic_data(): if no advanced
    strategies are specified, delegates to the legacy generator for
    identical output.

    New controls keys:
        generation_mode:   'advanced' (use this function) or 'legacy'
        privacy_preset:    'Maximum fidelity' / 'Balanced' / 'Strong privacy' / 'Maximum privacy'
        privacy_epsilon:   float (overrides preset if set)
        use_copula:        bool — fit multivariate Gaussian Copula
        copula_strength:   0-100 — blend toward independence (0) vs fitted correlation (100)
        enforce_constraints: bool — auto-detect and repair constraint violations

    Per-field strategy override in metadata["strategy"]:
        'auto', 'empirical', 'kde', 'copula', 'dp_laplace', 'identifier'
    """
    # ── Parse controls ──────────────────────────────────────
    row_count = int(controls.get("synthetic_rows", 500))
    seed = int(controls.get("seed", 42))
    use_copula = bool(controls.get("use_copula", False))
    copula_strength_pct = float(controls.get("copula_strength", 80))
    enforce_rules = bool(controls.get("enforce_constraints", True))

    privacy_preset = str(controls.get("privacy_preset", "Balanced"))
    privacy_epsilon = controls.get("privacy_epsilon")
    if privacy_epsilon is None:
        privacy_epsilon = epsilon_for_preset(privacy_preset)
    else:
        privacy_epsilon = float(privacy_epsilon)

    # Existing controls (reused from legacy)
    fidelity_priority = int(controls.get("fidelity_priority", 55))
    correlation_preservation = int(controls.get("correlation_preservation", 65))
    rare_case_retention = int(controls.get("rare_case_retention", 30))
    noise_level = int(controls.get("noise_level", 45))
    missingness_pattern = str(controls.get("missingness_pattern", "Preserve source pattern"))
    outlier_strategy = str(controls.get("outlier_strategy", "Preserve tails"))

    rng = np.random.default_rng(seed)
    synthetic_columns: dict[str, pd.Series] = {}
    excluded_columns: list[str] = []
    strategy_log: list[dict[str, str]] = []

    # ── 1. Prepare copula (if requested) ────────────────────
    # Fit a Gaussian Copula over all numeric columns that will be included.
    copula_columns: list[str] = []
    copula_model: dict[str, Any] | None = None
    copula_samples: pd.DataFrame | None = None

    if use_copula:
        numeric_included = [
            m["column"] for m in metadata
            if m.get("include")
            and m.get("data_type") == "numeric"
            and m.get("control_action") != "Exclude"
            and m["column"] in df.columns
        ]
        if len(numeric_included) >= 2:
            copula_columns = numeric_included
            copula_model = fit_copula(df, numeric_included)
            copula_strength = max(0.0, min(1.0, copula_strength_pct / 100.0))
            copula_samples = sample_copula(copula_model, row_count, rng, correlation_strength=copula_strength)

    # ── 2. Per-field generation dispatch ───────────────────
    uniform_weights = np.full(len(df), 1 / max(len(df), 1))
    rare_weights = _rare_row_weights(df, metadata) if len(df) else np.array([])
    retention_mix = np.interp(rare_case_retention, [0, 100], [0.0, 0.72])
    if len(df):
        anchor_weights = uniform_weights * (1 - retention_mix) + rare_weights * retention_mix
        anchor_indices = rng.choice(np.arange(len(df)), size=row_count, replace=True, p=anchor_weights)
    else:
        anchor_indices = np.array([], dtype=int)

    locked_columns = set(controls.get("locked_columns", []))

    for item in metadata:
        column = item["column"]
        control_action = item.get("control_action", "")
        if not item.get("include") or control_action == "Exclude":
            excluded_columns.append(column)
            continue
        if column not in df.columns:
            excluded_columns.append(column)
            continue

        role = item.get("data_type", "categorical")
        source_series = df[column]
        working_series = source_series.copy()
        locked_distribution = column in locked_columns

        # Generalization control actions
        if control_action == "Coarse geography":
            working_series = _coarsen_geography(working_series)
        elif control_action == "Group text":
            working_series = _group_text(working_series)
        elif control_action == "Group rare categories":
            working_series = _group_rare_categories(working_series)
        elif control_action == "Clip extremes" and role == "numeric":
            numeric_series = pd.to_numeric(working_series, errors="coerce")
            if numeric_series.notna().any():
                lower = numeric_series.quantile(0.05)
                upper = numeric_series.quantile(0.95)
                working_series = numeric_series.clip(lower=lower, upper=upper)

        # Resolve 'auto' strategy: picks based on role
        strategy_raw = str(item.get("strategy", "auto")).lower()
        if strategy_raw in {"", "auto", "sample_plus_noise", "sample_category", "sample_plus_jitter", "new_token"}:
            # Legacy strategy names → auto
            if role == "identifier":
                strategy = "identifier"
            elif role == "numeric":
                strategy = "copula" if column in copula_columns else "empirical"
            else:
                strategy = "auto"  # dispatch to categorical/date below
        else:
            strategy = strategy_raw

        is_identifier_like = role == "identifier" or strategy == "identifier"

        # ── Strategy dispatch ──────────────────────────────
        if is_identifier_like:
            generated = pd.Series(_generate_identifier(column, row_count))
            strategy_log.append({"column": column, "strategy": "Surrogate token", "role": role})
        elif strategy == "copula" and copula_samples is not None and column in copula_samples.columns:
            # Use the copula-sampled values (preserves joint correlations)
            values = copula_samples[column].to_numpy()
            # Integer-preserving
            source_numeric = pd.to_numeric(source_series, errors="coerce").dropna()
            if not source_numeric.empty and np.allclose(source_numeric, np.round(source_numeric), atol=1e-9):
                values = np.round(values)
            generated = pd.Series(values)
            strategy_log.append({"column": column, "strategy": "Gaussian Copula", "role": role})
        elif strategy == "kde" and role == "numeric":
            values = kde_sample_numeric(working_series, row_count, rng, bandwidth_scale=1.0)
            generated = pd.Series(values)
            strategy_log.append({"column": column, "strategy": "KDE (Silverman)", "role": role})
        elif strategy == "dp_laplace" and role == "numeric":
            # Base sampling (empirical) + DP noise
            base = _sample_numeric(column, working_series, row_count, rng, fidelity_priority,
                                    noise_level, locked_distribution, outlier_strategy)
            sensitivity = estimate_sensitivity(working_series)
            dp_noised = apply_dp_noise_numeric(
                pd.to_numeric(base, errors="coerce").fillna(base.median() if not base.empty else 0).to_numpy(),
                epsilon=privacy_epsilon,
                sensitivity=sensitivity,
                rng=rng,
            )
            generated = pd.Series(dp_noised)
            strategy_log.append({"column": column, "strategy": f"DP Laplace (ε={privacy_epsilon:.2f})", "role": role})
        elif role == "numeric":
            generated = _sample_numeric(column, working_series, row_count, rng, fidelity_priority,
                                         noise_level, locked_distribution, outlier_strategy)
            strategy_log.append({"column": column, "strategy": "Empirical + Gaussian noise", "role": role})
        elif role == "date":
            generated = _sample_dates(working_series, row_count, rng, fidelity_priority,
                                      control_action, noise_level, locked_distribution)
            strategy_log.append({"column": column, "strategy": "Date resample + jitter", "role": role})
        else:
            generated = _sample_categorical(working_series, row_count, rng, fidelity_priority,
                                            noise_level, rare_case_retention, locked_distribution)
            strategy_log.append({"column": column, "strategy": "Categorical frequency sampling", "role": role})

        # Anchor-blending (preserves non-copula correlation) + missingness
        if not is_identifier_like and len(anchor_indices):
            anchor_source = working_series.iloc[anchor_indices].reset_index(drop=True)
            anchor_ready = _prepare_anchor_output(anchor_source, role, control_action)
            # For copula fields, reduce extra blending (copula already handles correlation)
            effective_blend = 0 if strategy == "copula" else correlation_preservation
            generated = _blend_with_anchor(generated.reset_index(drop=True), anchor_ready, rng,
                                            effective_blend, locked_distribution)
            generated = _apply_missingness(generated, working_series, row_count, rng,
                                            missingness_pattern, anchor_source)

        synthetic_columns[column] = generated.reset_index(drop=True)

    synthetic_df = pd.DataFrame(synthetic_columns)

    # ── 3. Enforce constraints ─────────────────────────────
    detected_constraints: list[dict[str, Any]] = []
    constraint_repairs: list[dict[str, Any]] = []
    if enforce_rules and not synthetic_df.empty:
        detected_constraints = detect_constraints(df, metadata)
        synthetic_df, constraint_repairs = enforce_constraints(synthetic_df, detected_constraints)

    # ── 4. Build rich summary ──────────────────────────────
    summary = {
        "rows_generated": len(synthetic_df),
        "columns_generated": len(synthetic_df.columns),
        "excluded_columns": excluded_columns,
        "fidelity_priority": fidelity_priority,
        "privacy_preset": privacy_preset,
        "privacy_epsilon": round(float(privacy_epsilon), 3),
        "use_copula": use_copula,
        "copula_columns": copula_columns,
        "copula_strength": copula_strength_pct if use_copula else 0,
        "enforce_constraints": enforce_rules,
        "detected_constraints": detected_constraints,
        "constraint_repairs": constraint_repairs,
        "strategy_log": strategy_log,
        "correlation_preservation": correlation_preservation,
        "rare_case_retention": rare_case_retention,
        "noise_level": noise_level,
        "missingness_pattern": missingness_pattern,
        "outlier_strategy": outlier_strategy,
        "locked_columns": sorted(locked_columns),
        "generation_mode": "advanced",
    }

    return synthetic_df, summary
