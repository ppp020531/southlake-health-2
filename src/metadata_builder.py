from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_STRATEGIES = {
    "identifier": "identifier",
    "numeric": "copula",
    "categorical": "auto",
    "binary": "auto",
    "date": "auto",
}


def _default_control_action(column_name: str, role: str) -> str:
    lowered = column_name.lower()
    if role == "identifier":
        return "Tokenize"
    if role == "date":
        return "Date shift"
    if "postal" in lowered or "address" in lowered:
        return "Coarse geography"
    if "complaint" in lowered or "note" in lowered or "text" in lowered:
        return "Group text"
    if role == "numeric":
        return "Preserve"
    return "Preserve"


def build_metadata(df: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []

    for column in df.columns:
        details = profile["columns"][column]
        role = details["semantic_role"]
        lowered = column.lower()

        notes = "Preserve the field in the synthetic schema."
        if role == "identifier":
            notes = "Generate surrogate identifiers so the schema stays familiar without reusing source values."
        elif "postal" in lowered:
            notes = "Keep only coarse geography; avoid moving to full postal codes or addresses."
        elif "wait" in lowered:
            notes = "Scenario controls can reduce the longest waits while preserving normal operational patterns."
        elif role == "date":
            notes = "Dates are resampled and slightly jittered to protect exact encounter timing."

        metadata.append(
            {
                "column": column,
                "include": True,
                "data_type": role,
                "strategy": DEFAULT_STRATEGIES[role],
                "control_action": _default_control_action(column, role),
                "nullable": bool(details["missing_count"] > 0),
                "notes": notes,
            }
        )

    return metadata


def metadata_to_editor_frame(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(metadata)[["column", "include", "data_type", "strategy", "control_action", "nullable", "notes"]]


def editor_frame_to_metadata(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.copy()
    for column in ["include", "nullable"]:
        cleaned[column] = cleaned[column].astype(bool)
    for column in ["data_type", "strategy", "control_action", "notes"]:
        cleaned[column] = cleaned[column].fillna("").astype(str)
    return cleaned.to_dict(orient="records")
