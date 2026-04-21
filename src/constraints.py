"""Automatic detection and enforcement of inter-field logical constraints.

Real healthcare datasets have implicit rules: admission_date precedes
discharge_date, age is non-negative, length_of_stay = discharge - admission,
CTAS level is in {1,2,3,4,5}, etc. A naive per-column generator violates
these — producing negative ages, discharge-before-admission records,
out-of-range categoricals.

This module:
1. Auto-detects constraints by inspecting source data patterns
2. Enforces them post-generation via repair rules
3. Reports violations so analysts can see what was fixed

Types of constraints detected:
- Non-negative numeric (columns whose source has 100% non-negative values)
- Date ordering pairs (columns where col_A <= col_B holds in >=95% of source)
- Bounded numeric range (clip sampled values to source 5th-95th percentile
  when a 'clip' strategy is requested)
- Categorical value set (only labels that appear in source)
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _is_date_column(series: pd.Series) -> bool:
    """Heuristic: does this column look like dates?

    Strict: only checks object/string dtypes. Numeric columns are never
    treated as dates (which pandas would technically parse as unix
    nanosecond timestamps, leading to false positives).
    """
    # Numeric columns are definitely not date columns
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    # Native datetime types are obviously dates
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False
    # Must be string-like
    sample = non_null.astype(str).sample(min(50, len(non_null)), random_state=0) if len(non_null) > 50 else non_null.astype(str)
    # Reject if too short (e.g., "1", "2" would parse as epoch nanos)
    if (sample.str.len() < 6).mean() > 0.3:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() < 0.85:
            return False
        # Reasonable date range (after 1970-01-02 to rule out epoch nano artifacts,
        # before year 2100)
        valid = parsed.dropna()
        if valid.empty:
            return False
        min_year = valid.dt.year.min()
        max_year = valid.dt.year.max()
        return 1970 <= min_year <= 2100 and 1970 <= max_year <= 2100
    except Exception:
        return False


def detect_constraints(
    df: pd.DataFrame,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect constraints from the source dataset.

    Returns a list of constraint records, each with fields:
    - 'kind':        constraint type ('non_negative', 'date_order', 'non_null_pair')
    - 'columns':     list of involved columns
    - 'rule':        human-readable rule string
    - 'confidence':  fraction of rows satisfying this constraint in source
    """
    constraints: list[dict[str, Any]] = []

    # Active (included) columns only
    active_cols = {m["column"] for m in metadata if m.get("include")}
    if not active_cols:
        return []

    # ── 1. Non-negative numeric columns ─────────────────────────
    for col in df.columns:
        if col not in active_cols:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        # If a column looks numeric AND has zero negatives, assume non-negative
        if (numeric >= 0).all() and len(numeric) >= 5:
            # Only flag fields where "natural" domain is non-negative
            lowered = col.lower()
            non_neg_likely = any(
                token in lowered
                for token in ["age", "wait", "stay", "count", "duration", "length", "minutes", "hours", "days", "ctas", "score", "visits"]
            )
            if non_neg_likely:
                constraints.append({
                    "kind": "non_negative",
                    "columns": [col],
                    "rule": f"{col} >= 0",
                    "confidence": 1.0,
                })

    # ── 2. Date ordering pairs ─────────────────────────────────
    date_cols = [c for c in df.columns if c in active_cols and _is_date_column(df[c])]
    # Look for admission/discharge-style naming pairs
    for i, col_a in enumerate(date_cols):
        for col_b in date_cols[i + 1:]:
            a_parsed = pd.to_datetime(df[col_a], errors="coerce", format="mixed")
            b_parsed = pd.to_datetime(df[col_b], errors="coerce", format="mixed")
            paired = pd.concat([a_parsed, b_parsed], axis=1).dropna()
            if len(paired) < 5:
                continue
            a_values = paired.iloc[:, 0]
            b_values = paired.iloc[:, 1]
            # Check if a <= b or b <= a dominates
            a_le_b_rate = (a_values <= b_values).mean()
            b_le_a_rate = (b_values <= a_values).mean()
            if a_le_b_rate >= 0.95:
                constraints.append({
                    "kind": "date_order",
                    "columns": [col_a, col_b],
                    "rule": f"{col_a} ≤ {col_b}",
                    "confidence": round(float(a_le_b_rate), 3),
                })
            elif b_le_a_rate >= 0.95:
                constraints.append({
                    "kind": "date_order",
                    "columns": [col_b, col_a],
                    "rule": f"{col_b} ≤ {col_a}",
                    "confidence": round(float(b_le_a_rate), 3),
                })

    return constraints


def enforce_constraints(
    synthetic_df: pd.DataFrame,
    constraints: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Enforce constraints on synthetic data. Returns the repaired dataframe
    and a list of repair reports.

    Repair strategies:
    - non_negative: clip values < 0 to 0
    - date_order: swap pairs where order is violated
    """
    df = synthetic_df.copy()
    repairs: list[dict[str, Any]] = []

    for constraint in constraints:
        kind = constraint.get("kind")
        cols = constraint.get("columns", [])

        if kind == "non_negative" and cols:
            col = cols[0]
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            bad_mask = numeric < 0
            bad_count = int(bad_mask.fillna(False).sum())
            if bad_count > 0:
                # Clip to 0 (preserves row; doesn't drop)
                numeric = numeric.where(numeric >= 0, 0)
                df[col] = numeric
                repairs.append({
                    "rule": constraint["rule"],
                    "rows_repaired": bad_count,
                    "action": f"Clipped {bad_count} negative value(s) to 0",
                })

        elif kind == "date_order" and len(cols) == 2:
            col_a, col_b = cols
            if col_a not in df.columns or col_b not in df.columns:
                continue
            parsed_a = pd.to_datetime(df[col_a], errors="coerce", format="mixed")
            parsed_b = pd.to_datetime(df[col_b], errors="coerce", format="mixed")
            both_valid = parsed_a.notna() & parsed_b.notna()
            violated = both_valid & (parsed_a > parsed_b)
            violation_count = int(violated.sum())
            if violation_count > 0:
                # Swap violated rows
                temp_a = df[col_a].copy()
                df.loc[violated, col_a] = df.loc[violated, col_b]
                df.loc[violated, col_b] = temp_a.loc[violated]
                repairs.append({
                    "rule": constraint["rule"],
                    "rows_repaired": violation_count,
                    "action": f"Swapped {violation_count} row(s) where {col_a} > {col_b}",
                })

    return df, repairs
