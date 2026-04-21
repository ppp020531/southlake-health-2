from __future__ import annotations

from typing import Any

import pandas as pd


def _clean_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.replace(r"^\s*$", pd.NA, regex=True)
    return series


def _semantic_role(column_name: str, series: pd.Series) -> str:
    name = column_name.lower()
    non_null = series.dropna()

    if any(token in name for token in ["encounter_id", "patient_id", "visit_id", "mrn"]):
        return "identifier"

    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        if non_null.nunique() <= 2:
            return "binary"
        return "numeric"

    if not non_null.empty:
        parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if parsed_dates.notna().mean() >= 0.85 and non_null.nunique() > 2:
            return "date"

    normalized_values = non_null.astype(str).str.strip().str.lower()
    if not non_null.empty and normalized_values.nunique() <= 2:
        return "binary"

    return "categorical"


def _column_profile(column_name: str, series: pd.Series, row_count: int) -> dict[str, Any]:
    cleaned = _clean_series(series)
    role = _semantic_role(column_name, cleaned)
    non_null = cleaned.dropna()

    profile: dict[str, Any] = {
        "column": column_name,
        "semantic_role": role,
        "dtype": str(cleaned.dtype),
        "missing_count": int(cleaned.isna().sum()),
        "missing_pct": round(float(cleaned.isna().mean() * 100), 2),
        "unique_count": int(non_null.nunique()),
        "unique_ratio": round(float(non_null.nunique() / max(len(non_null), 1)), 3),
        "examples": [str(value) for value in non_null.astype(str).head(3).tolist()],
    }

    if role == "numeric":
        numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
        if not numeric.empty:
            profile.update(
                {
                    "min": round(float(numeric.min()), 3),
                    "max": round(float(numeric.max()), 3),
                    "mean": round(float(numeric.mean()), 3),
                    "std": round(float(numeric.std(ddof=0)), 3),
                    "q1": round(float(numeric.quantile(0.25)), 3),
                    "median": round(float(numeric.quantile(0.5)), 3),
                    "q3": round(float(numeric.quantile(0.75)), 3),
                }
            )
    elif role == "date":
        parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed").dropna()
        if not parsed.empty:
            profile.update(
                {
                    "min": str(parsed.min().date()),
                    "max": str(parsed.max().date()),
                }
            )
    else:
        top_values = (
            non_null.astype(str).value_counts(normalize=True).head(5).mul(100).round(1).to_dict()
        )
        profile["top_values"] = top_values

    if row_count:
        profile["completeness_score"] = round(
            float(100 - (profile["missing_count"] / row_count) * 100),
            1,
        )
    else:
        profile["completeness_score"] = 0.0

    return profile


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    cleaned_df = df.copy()
    for column in cleaned_df.columns:
        cleaned_df[column] = _clean_series(cleaned_df[column])

    row_count, column_count = cleaned_df.shape
    columns = {
        column: _column_profile(column, cleaned_df[column], row_count) for column in cleaned_df.columns
    }

    role_counts: dict[str, int] = {}
    for details in columns.values():
        role = details["semantic_role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    summary = {
        "rows": int(row_count),
        "columns": int(column_count),
        "missing_cells": int(cleaned_df.isna().sum().sum()),
        "duplicate_rows": int(cleaned_df.duplicated().sum()),
        "numeric_columns": role_counts.get("numeric", 0),
        "categorical_columns": role_counts.get("categorical", 0),
        "date_columns": role_counts.get("date", 0),
        "identifier_columns": role_counts.get("identifier", 0),
    }

    return {
        "summary": summary,
        "role_counts": role_counts,
        "columns": columns,
        "dataset_story": (
            f"{row_count} rows across {column_count} columns, with "
            f"{summary['missing_cells']} missing cells and {summary['duplicate_rows']} duplicate rows."
        ),
    }
