from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric_score(original: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    original_numeric = pd.to_numeric(original, errors="coerce").dropna()
    synthetic_numeric = pd.to_numeric(synthetic, errors="coerce").dropna()

    if original_numeric.empty or synthetic_numeric.empty:
        return {"score": 0.0, "details": "Insufficient numeric data."}

    mean_gap = abs(float(original_numeric.mean() - synthetic_numeric.mean())) / max(abs(float(original_numeric.mean())), 1.0)
    std_gap = abs(float(original_numeric.std(ddof=0) - synthetic_numeric.std(ddof=0))) / max(float(original_numeric.std(ddof=0)), 1.0)
    quantile_gap = (
        abs(original_numeric.quantile([0.25, 0.5, 0.75]) - synthetic_numeric.quantile([0.25, 0.5, 0.75]))
        .mean()
        / max(float(original_numeric.std(ddof=0)), 1.0)
    )
    raw_penalty = float(np.mean([mean_gap, std_gap, quantile_gap]))
    score = max(0.0, 1.0 - raw_penalty)
    return {
        "score": round(score * 100, 1),
        "details": f"Mean gap {mean_gap:.2f}, spread gap {std_gap:.2f}, quantile gap {quantile_gap:.2f}.",
    }


def _categorical_score(original: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    original_distribution = original.fillna("Missing").astype(str).value_counts(normalize=True)
    synthetic_distribution = synthetic.fillna("Missing").astype(str).value_counts(normalize=True)
    all_categories = sorted(set(original_distribution.index).union(set(synthetic_distribution.index)))

    if not all_categories:
        return {"score": 0.0, "details": "Insufficient categorical data."}

    original_probs = original_distribution.reindex(all_categories, fill_value=0.0)
    synthetic_probs = synthetic_distribution.reindex(all_categories, fill_value=0.0)
    total_variation_distance = 0.5 * np.abs(original_probs - synthetic_probs).sum()
    score = max(0.0, 1.0 - float(total_variation_distance))
    return {
        "score": round(score * 100, 1),
        "details": f"Distribution drift {float(total_variation_distance):.2f}.",
    }


def _correlation_preservation_score(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Measure how well inter-column correlations are preserved.

    Uses Frobenius distance between the original and synthetic correlation
    matrices for numeric columns. Returns a 0-100 score (100 = perfectly
    preserved correlations, 0 = completely lost).

    Reports the top field pairs with largest correlation drift for analyst
    feedback.
    """
    numeric_cols = [
        item["column"] for item in metadata
        if item["include"] and item["data_type"] == "numeric" and item["column"] in synthetic_df.columns
    ]
    if len(numeric_cols) < 2:
        return 0.0, {"details": "Need ≥2 numeric fields for correlation analysis.", "drift_pairs": []}

    # Build numeric sub-frames, aligning columns
    orig_numeric = original_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    synth_numeric = synthetic_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    try:
        orig_corr = orig_numeric.corr().fillna(0).to_numpy()
        synth_corr = synth_numeric.corr().fillna(0).to_numpy()
    except Exception:
        return 0.0, {"details": "Correlation computation failed.", "drift_pairs": []}

    # Frobenius distance (normalized)
    diff = orig_corr - synth_corr
    frob_distance = float(np.sqrt((diff ** 2).sum()))
    # Maximum possible (if everything went from +1 to -1 on all off-diagonals)
    k = len(numeric_cols)
    max_distance = float(np.sqrt(k * (k - 1) * 4))
    if max_distance > 0:
        normalized_distance = min(1.0, frob_distance / max_distance)
    else:
        normalized_distance = 0.0

    score = round((1.0 - normalized_distance) * 100, 1)

    # Top drift pairs
    drift_pairs: list[dict[str, Any]] = []
    for i in range(k):
        for j in range(i + 1, k):
            delta = float(abs(orig_corr[i, j] - synth_corr[i, j]))
            drift_pairs.append({
                "pair": f"{numeric_cols[i]} ↔ {numeric_cols[j]}",
                "original": round(float(orig_corr[i, j]), 3),
                "synthetic": round(float(synth_corr[i, j]), 3),
                "drift": round(delta, 3),
            })
    drift_pairs.sort(key=lambda x: x["drift"], reverse=True)

    return score, {
        "details": f"Frobenius distance {frob_distance:.2f} across {k} numeric fields.",
        "drift_pairs": drift_pairs[:8],  # top 8
    }


def validate_synthetic_data(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: list[dict[str, Any]],
    controls: dict[str, Any],
) -> dict[str, Any]:
    fidelity_rows: list[dict[str, Any]] = []
    active_columns = [item for item in metadata if item["include"] and item["column"] in synthetic_df.columns]

    for column_meta in active_columns:
        column = column_meta["column"]
        role = column_meta["data_type"]

        if role == "numeric":
            score_card = _numeric_score(original_df[column], synthetic_df[column])
        else:
            score_card = _categorical_score(original_df[column], synthetic_df[column])

        fidelity_rows.append(
            {
                "column": column,
                "role": role,
                "score": score_card["score"],
                "details": score_card["details"],
            }
        )

    fidelity_table = pd.DataFrame(fidelity_rows)
    if not fidelity_table.empty:
        fidelity_table = fidelity_table.sort_values(by="score", ascending=False)
    else:
        fidelity_table = pd.DataFrame(columns=["column", "role", "score", "details"])
    fidelity_score = round(float(fidelity_table["score"].mean()), 1) if not fidelity_table.empty else 0.0

    comparison_columns = [
        item["column"] for item in active_columns if item["data_type"] not in {"identifier", "date"}
    ]
    if comparison_columns:
        original_signatures = set(
            original_df[comparison_columns].fillna("Missing").astype(str).agg(" | ".join, axis=1).tolist()
        )
        synthetic_signatures = set(
            synthetic_df[comparison_columns].fillna("Missing").astype(str).agg(" | ".join, axis=1).tolist()
        )
        exact_overlap_rate = len(original_signatures.intersection(synthetic_signatures)) / max(
            len(synthetic_signatures), 1
        )
    else:
        exact_overlap_rate = 0.0

    identifier_overlap = 0
    for item in active_columns:
        if item["data_type"] == "identifier":
            common = set(original_df[item["column"]].astype(str)).intersection(
                set(synthetic_df[item["column"]].astype(str))
            )
            identifier_overlap += len(common)

    fidelity_priority = int(controls.get("fidelity_priority", 55))
    privacy_pressure = (fidelity_priority / 100) * 10
    privacy_score = max(
        0.0,
        min(
            100.0,
            100.0 - exact_overlap_rate * 100 - identifier_overlap * 10 - privacy_pressure,
        ),
    )

    privacy_checks = pd.DataFrame(
        [
            {
                "check": "Exact non-identifier row overlap",
                "result": f"{exact_overlap_rate * 100:.1f}%",
                "interpretation": "Lower is better for privacy.",
            },
            {
                "check": "Identifier reuse",
                "result": str(identifier_overlap),
                "interpretation": "Should stay at zero when surrogate tokens are used.",
            },
            {
                "check": "Current privacy posture",
                "result": "Higher privacy" if fidelity_priority < 40 else "Balanced" if fidelity_priority < 70 else "Higher fidelity",
                "interpretation": "Derived from the privacy-vs-fidelity control.",
            },
        ]
    )

    overall_score = round(fidelity_score * 0.6 + privacy_score * 0.4, 1)
    if overall_score >= 85:
        verdict = "Demo ready: the synthetic output is representative with a comfortable privacy buffer."
    elif overall_score >= 70:
        verdict = "Good baseline: review weaker columns and decide whether to trade a bit more fidelity for privacy."
    else:
        verdict = "Needs another pass: adjust metadata or generation controls before presenting the dataset as production-like."

    # New: correlation preservation metric
    correlation_score, correlation_details = _correlation_preservation_score(
        original_df, synthetic_df, metadata
    )

    return {
        "overall_score": overall_score,
        "fidelity_score": fidelity_score,
        "privacy_score": round(privacy_score, 1),
        "correlation_score": correlation_score,
        "correlation_details": correlation_details,
        "verdict": verdict,
        "fidelity_table": fidelity_table,
        "privacy_checks": privacy_checks,
    }
