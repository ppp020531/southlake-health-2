from __future__ import annotations

import base64
import json
import time
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from src.cleaner import apply_hygiene_fixes
from src.generator import generate_synthetic_data, generate_synthetic_advanced
from src.hygiene_advisor import review_hygiene
from src.metadata_builder import build_metadata, editor_frame_to_metadata, metadata_to_editor_frame
from src.profiler import profile_dataframe
from src.validator import validate_synthetic_data
from src.strategies import STRATEGY_LABELS, STRATEGY_DESCRIPTIONS
from src.dp_noise import PRIVACY_PRESETS
from src.agent_orchestrator import (
    render_agent_orchestration_panel,
    render_agent_timeline,
    build_release_readiness_verdicts,
    render_release_readiness_verdicts,
    agent_event_label,
    render_agent_readiness_panel,
    render_classified_hygiene,
    build_metadata_review_artifact,
    render_metadata_review_artifact,
    build_stakeholder_interpretations,
    render_stakeholder_interpretations,
    render_upload_status_panel,
    render_review_package_summary,
    render_synthetic_verification_summary,
)

APP_TITLE = "Southlake Health — Agentic Synthetic Data Workspace"
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data.csv"
LOGO_PATH = Path(__file__).parent / "SL_SouthlakeHealth-logo-rgb.png"

STEP_CONFIG = [
    {
        "title": "Upload Source Data",
        "description": "Register the source dataset and open a governed request.",
        "owner": "Data Analyst",
    },
    {
        "title": "Scan Source Data",
        "description": "Explore the cleaned source dataset before metadata design.",
        "owner": "System",
    },
    {
        "title": "Configure & Generate Package",
        "description": "Configure metadata, generate synthetic preview, and submit for reviewer sign-off.",
        "owner": "Data Analyst",
    },
    {
        "title": "Manager Review & Approve",
        "description": "Reviewer inspects the submitted package including synthetic preview before approval.",
        "owner": "Manager / Reviewer",
    },
    {
        "title": "Release Synthetic Package",
        "description": "Download the approved synthetic package for controlled distribution.",
        "owner": "Data Analyst",
    },
]

ROLE_TO_GROUP: dict[str, str] = {
    "Data Analyst": "Analyst workspace",
    "Manager / Reviewer": "Review workspace",
}

ROLE_VISIBLE_STEPS: dict[str, list[int]] = {
    "Data Analyst": [0, 1, 2, 3, 4],
    "Manager / Reviewer": [3, 4],
}

ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "Data Analyst": {
        "password": "test",
        "summary": "Owns upload, scan review, data settings, request submission, and final download.",
        "permissions": {
            "upload",
            "view_raw",
            "remediate",
            "edit_metadata",
            "submit_metadata",
            "generate",
            "download_results",
            "share_results",
            "rollback",
        },
    },
    "Manager / Reviewer": {
        "password": "test",
        "summary": "Approves or rejects submitted requests and reviews the final synthetic output.",
        "permissions": {"approve_metadata", "view_audit", "review_results"},
    },
}

EMPTY_METADATA_COLUMNS = ["column", "include", "data_type", "strategy", "control_action", "nullable", "notes"]

WORKFLOW_STATE_KEYS = [
    "source_df",
    "source_label",
    "project_purpose",
    "source_file_size",
    "profile",
    "hygiene",
    "metadata_editor_df",
    "controls",
    "intake_confirmed",
    "hygiene_reviewed",
    "settings_reviewed",
    "settings_review_signature",
    "metadata_status",
    "metadata_submitted_by",
    "metadata_submitted_at",
    "metadata_approved_by",
    "metadata_approved_at",
    "metadata_review_note",
    "metadata_reviewed_by",
    "metadata_reviewed_at",
    "last_reviewed_metadata_signature",
    "last_cleaning_actions",
    "current_metadata_package_id",
    "metadata_package_log",
    "metadata_has_unsubmitted_changes",
    "synthetic_df",
    "generation_summary",
    "validation",
    "last_generation_signature",
    "release_status",
    "export_requested_by",
    "export_policy_approved_by",
    "export_approved_by",
    "results_shared_by",
    "results_shared_at",
    "audit_events",
]

SHARED_ROOT_KEYS = [
    "request_registry",
    "active_request_id",
    "next_request_number",
]


@st.cache_resource
def get_shared_workspace_store() -> dict[str, Any]:
    return {"state": {}}


def load_shared_workspace_state() -> None:
    shared_state = get_shared_workspace_store()["state"]
    if not shared_state:
        return
    for key in SHARED_ROOT_KEYS:
        if key in shared_state:
            st.session_state[key] = deepcopy(shared_state[key])

    if st.session_state.get("request_registry"):
        active_id = st.session_state.get("active_request_id") or st.session_state["request_registry"][0]["request_id"]
        record = next((item for item in st.session_state["request_registry"] if item["request_id"] == active_id), None)
        if record is None:
            record = st.session_state["request_registry"][0]
            st.session_state["active_request_id"] = record["request_id"]
        snapshot = record.get("snapshot", {})
        for key in WORKFLOW_STATE_KEYS:
            if key in snapshot:
                st.session_state[key] = deepcopy(snapshot[key])


def persist_shared_workspace_state() -> None:
    sync_active_request_snapshot()
    shared_state = {}
    for key in SHARED_ROOT_KEYS:
        if key in st.session_state:
            shared_state[key] = deepcopy(st.session_state[key])
    get_shared_workspace_store()["state"] = shared_state


def rerun_with_persist() -> None:
    persist_shared_workspace_state()
    st.rerun()


def capture_workflow_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in WORKFLOW_STATE_KEYS:
        if key in st.session_state:
            snapshot[key] = deepcopy(st.session_state[key])
    return snapshot


def request_status_from_snapshot(snapshot: dict[str, Any]) -> str:
    if snapshot.get("source_df") is None or snapshot.get("profile") is None:
        return "Awaiting upload"
    metadata = editor_frame_to_metadata(snapshot.get("metadata_editor_df", pd.DataFrame())) if snapshot.get("metadata_editor_df") is not None else []
    controls = snapshot.get("controls", {})
    metadata_status = snapshot.get("metadata_status", "Draft")
    if not snapshot.get("intake_confirmed"):
        return "Uploaded"
    if not snapshot.get("hygiene_reviewed"):
        return "Scanned"
    if not snapshot.get("settings_reviewed"):
        return "Needs settings review"
    if metadata_status == "Changes Requested":
        return "Changes requested"
    if metadata_status == "Rejected":
        return "Rejected"
    if metadata_status == "In Review":
        return "Pending approval"
    if metadata_status != "Approved":
        return "Reviewed"
    if snapshot.get("synthetic_df") is None:
        return "Approved"
    if snapshot.get("last_generation_signature") and snapshot.get("last_generation_signature") != build_generation_signature(metadata, controls):
        return "Regeneration required"
    if snapshot.get("results_shared_at"):
        return "Shared"
    return "Generated"


def sync_active_request_snapshot() -> None:
    request_id = st.session_state.get("active_request_id")
    registry = st.session_state.get("request_registry", [])
    if not request_id or not registry:
        return
    snapshot = capture_workflow_snapshot()
    for record in registry:
        if record["request_id"] == request_id:
            record["snapshot"] = snapshot
            record["status"] = request_status_from_snapshot(snapshot)
            record["updated_at"] = format_timestamp()
            record["source_label"] = snapshot.get("source_label", record.get("source_label", ""))
            break


def restore_request_workspace(request_id: str) -> None:
    registry = st.session_state.get("request_registry", [])
    record = next((item for item in registry if item["request_id"] == request_id), None)
    if record is None:
        return
    snapshot = deepcopy(record.get("snapshot", {}))
    for key in WORKFLOW_STATE_KEYS:
        if key in snapshot:
            st.session_state[key] = snapshot[key]
    st.session_state.active_request_id = request_id
    st.session_state.pending_active_request_selector = request_id
    st.session_state.uploaded_signature = None


def create_new_request(df: pd.DataFrame, label: str) -> str:
    request_number = int(st.session_state.get("next_request_number", 1))
    request_id = f"REQ-{request_number:03d}"
    st.session_state.next_request_number = request_number + 1
    set_source_dataframe(df, label)
    st.session_state.active_request_id = request_id
    snapshot = capture_workflow_snapshot()
    record = {
        "request_id": request_id,
        "source_label": label,
        "created_by": st.session_state.get("current_role", "System"),
        "created_at": format_timestamp(),
        "updated_at": format_timestamp(),
        "status": request_status_from_snapshot(snapshot),
        "snapshot": snapshot,
    }
    st.session_state.request_registry.insert(0, record)
    st.session_state.pending_active_request_selector = request_id
    persist_shared_workspace_state()
    return request_id


def default_generation_controls(row_count: int = 20) -> dict[str, Any]:
    return {
        "generation_preset": "Internal prototyping",
        "fidelity_priority": 62,
        "synthetic_rows": max(int(row_count), 20),
        "row_count_choice": "Same as source",
        "locked_columns": [],
        "correlation_preservation": 65,
        "rare_case_retention": 35,
        "noise_level": 45,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
        "seed": 42,
        # Advanced algorithm controls (new)
        "privacy_preset": "Balanced",
        "privacy_epsilon": 2.0,
        "use_copula": True,
        "copula_strength": 80,
        "enforce_constraints": True,
    }


def empty_metadata_editor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_METADATA_COLUMNS)


def has_active_dataset() -> bool:
    return st.session_state.get("source_df") is not None and st.session_state.get("profile") is not None


def reset_workspace_to_empty_request(label: str = "Awaiting dataset upload") -> None:
    st.session_state.source_df = None
    st.session_state.source_label = label
    st.session_state.project_purpose = ""
    st.session_state.source_file_size = None
    st.session_state.profile = None
    st.session_state.hygiene = {
        "issues": [],
        "severity_counts": {"High": 0, "Medium": 0, "Low": 0},
        "quality_score": 100,
        "summary": {"issues_found": 0, "high_priority": 0, "duplicate_rows": 0},
    }
    st.session_state.metadata_editor_df = empty_metadata_editor_frame()
    st.session_state.controls = default_generation_controls()
    st.session_state.intake_confirmed = False
    st.session_state.hygiene_reviewed = False
    st.session_state.settings_reviewed = False
    st.session_state.settings_review_signature = None
    st.session_state.metadata_status = "Draft"
    st.session_state.metadata_submitted_by = None
    st.session_state.metadata_submitted_at = None
    st.session_state.metadata_approved_by = None
    st.session_state.metadata_approved_at = None
    st.session_state.metadata_review_note = None
    st.session_state.metadata_reviewed_by = None
    st.session_state.metadata_reviewed_at = None
    st.session_state.last_reviewed_metadata_signature = None
    st.session_state.last_cleaning_actions = []
    st.session_state.current_metadata_package_id = None
    st.session_state.metadata_package_log = []
    st.session_state.metadata_has_unsubmitted_changes = False
    clear_generation_outputs()
    st.session_state.audit_events = []
    st.session_state.uploaded_signature = None
    st.session_state.current_step = 0


def create_blank_request(label: str = "Awaiting dataset upload") -> str:
    request_number = int(st.session_state.get("next_request_number", 1))
    request_id = f"REQ-{request_number:03d}"
    st.session_state.next_request_number = request_number + 1
    reset_workspace_to_empty_request(label)
    st.session_state.active_request_id = request_id
    snapshot = capture_workflow_snapshot()
    record = {
        "request_id": request_id,
        "source_label": label,
        "created_by": st.session_state.get("current_role", "System"),
        "created_at": format_timestamp(),
        "updated_at": format_timestamp(),
        "status": request_status_from_snapshot(snapshot),
        "snapshot": snapshot,
    }
    st.session_state.request_registry.insert(0, record)
    st.session_state.pending_active_request_selector = request_id
    persist_shared_workspace_state()
    return request_id


def role_with_group(role: str) -> str:
    return f"{role} · {ROLE_TO_GROUP.get(role, 'Users')}"


def clean_dataset_label(label: str) -> str:
    if " • " in label:
        return label.split(" • ", 1)[-1]
    return label


def request_display_label(request_id: str) -> str:
    registry = st.session_state.get("request_registry", [])
    record = next((item for item in registry if item["request_id"] == request_id), None)
    if record is None:
        return request_id
    label = clean_dataset_label(record.get("source_label", "Dataset"))
    return f"{record['request_id']} · {record['status']} · {label}"


def current_dataset_label() -> str:
    return clean_dataset_label(st.session_state.get("source_label", "No dataset uploaded"))


def format_file_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "Unknown size"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def dataset_status_summary() -> tuple[str, str]:
    if not has_active_dataset():
        return "Awaiting upload", "No source dataset is loaded in the current request."
    rows = st.session_state.profile["summary"]["rows"]
    columns = st.session_state.profile["summary"]["columns"]
    size = format_file_size(st.session_state.get("source_file_size"))
    return "Profile ready", f"{current_dataset_label()} · {size} · {rows:,} rows · {columns} columns"


def build_submission_checklist() -> list[tuple[str, bool]]:
    purpose_added = bool(st.session_state.get("project_purpose", "").strip())
    dataset_uploaded = st.session_state.get("source_df") is not None
    preview_generated = has_active_dataset()
    return [
        ("Project purpose added", purpose_added),
        ("Dataset uploaded", dataset_uploaded),
        ("Workspace preview generated", preview_generated),
    ]


def submission_ready() -> bool:
    return all(done for _, done in build_submission_checklist())


def submission_missing_items() -> list[str]:
    return [label for label, done in build_submission_checklist() if not done]


def build_current_request_status_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[tuple[str, str]]:
    package_record = current_review_package_record() or active_metadata_package_record(metadata)
    review_value = metadata_display_status(metadata)
    if package_record is not None:
        review_value = f"{review_value} · {package_record['package_id']}"
    return [
        ("Current request", st.session_state.active_request_id or "Not created"),
        ("Request status", request_status_from_snapshot(capture_workflow_snapshot())),
        ("Review status", review_value),
        ("Final output", effective_release_status(metadata, controls)),
    ]


def build_request_queue_frame() -> pd.DataFrame:
    rows = []
    for record in st.session_state.get("request_registry", []):
        snapshot = record.get("snapshot", {})
        rows.append(
            {
                "Request": record["request_id"],
                "Dataset": clean_dataset_label(record.get("source_label", "")),
                "Purpose": snapshot.get("project_purpose", "") or "Not set",
                "Status": record.get("status", "Draft"),
                "Updated": record.get("updated_at", ""),
                "Created by": record.get("created_by", ""),
            }
        )
    return pd.DataFrame(rows)


def clear_request_queue() -> None:
    st.session_state.request_registry = []
    st.session_state.active_request_id = None
    st.session_state.pending_active_request_selector = None
    st.session_state.next_request_number = 1
    reset_workspace_to_empty_request()
    create_blank_request()


def schedule_request_queue_clear() -> None:
    st.session_state.pending_queue_clear = True
    st.rerun()

GENERATION_PRESETS: dict[str, dict[str, Any]] = {
    "Analytics sandbox": {
        "description": "Maximum fidelity for internal modeling. Preserves joint distributions and rare operational edge cases. Use for ED flow simulation, capacity modeling, and analytics pipeline development.",
        "privacy_preset": "Maximum fidelity",
        "privacy_epsilon": 10.0,
        "use_copula": True,
        "copula_strength": 90,
        "enforce_constraints": True,
        "fidelity_priority": 78,
        "correlation_preservation": 85,
        "rare_case_retention": 70,
        "noise_level": 25,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
    },
    "Internal prototyping": {
        "description": "Balanced fidelity and privacy. Default for pipeline development, vendor sandbox testing, and workflow rehearsal inside Southlake's governance boundary.",
        "privacy_preset": "Balanced",
        "privacy_epsilon": 2.0,
        "use_copula": True,
        "copula_strength": 80,
        "enforce_constraints": True,
        "fidelity_priority": 62,
        "correlation_preservation": 65,
        "rare_case_retention": 35,
        "noise_level": 45,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Preserve tails",
    },
    "Privacy-first review": {
        "description": "Strongest privacy guarantees with moderate structural realism. Heavy Laplace noise, reduced rare-case retention, clipped outliers. Use for compliance review, external audit, and privacy attestation.",
        "privacy_preset": "Strong privacy",
        "privacy_epsilon": 0.5,
        "use_copula": True,
        "copula_strength": 60,
        "enforce_constraints": True,
        "fidelity_priority": 38,
        "correlation_preservation": 40,
        "rare_case_retention": 10,
        "noise_level": 65,
        "missingness_pattern": "Preserve source pattern",
        "outlier_strategy": "Clip extremes",
    },
}

METADATA_QUICK_PRESETS: dict[str, str] = {
    "Use approved metadata": "",
    "Tighten sensitive fields": "tighten_phi",
    "Preserve analytics detail": "preserve_analytics",
    "Reset metadata defaults": "reset_defaults",
}


@st.cache_data(show_spinner=False)
def load_logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    suffix = LOGO_PATH.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{encoded}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

            :root {
                --brand: #0b5ea8;
                --brand-deep: #08467d;
                --accent: #19cbc5;
                --bg: #e9f0f7;              /* was #f4f8fb — deeper so cards pop */
                --bg-deep: #dce7f1;
                --surface: #ffffff;
                --surface-soft: #f5f9fc;    /* was #f8fbfd — slight tint */
                --surface-tinted: #fbfdff;  /* for gradient card bottoms */
                --line: #b8c9da;            /* was #d6e2ec — stronger border */
                --line-soft: #d6e2ec;       /* original subtle line for dividers */
                --text: #0f2742;            /* was #17324d — a touch darker */
                --text-strong: #0a1f36;
                --muted: #5b7893;           /* was #668097 — tighter contrast */
                --warn: #9c6a17;
                --warn-bg: #fff6e3;
                --danger: #9d2b3c;
                --danger-bg: #fff1f3;
                --good: #136b48;
                --good-bg: #edf9f3;
                --shadow: 0 2px 4px rgba(8, 70, 125, 0.05), 0 14px 32px rgba(8, 70, 125, 0.12);
                --shadow-lift: 0 3px 6px rgba(8, 70, 125, 0.06), 0 20px 44px rgba(8, 70, 125, 0.16);
                --shadow-inset: inset 0 1px 0 rgba(255, 255, 255, 0.75);
            }

            html {
                color-scheme: light;
            }

            html, body, [class*="css"] {
                font-family: "IBM Plex Sans", sans-serif;
                color: var(--text);
                background: var(--bg);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(ellipse 80% 60% at 50% -10%, rgba(11, 94, 168, 0.16), transparent 55%),
                    radial-gradient(circle at 10% 0%, rgba(11, 94, 168, 0.10), transparent 30%),
                    radial-gradient(circle at 95% 5%, rgba(25, 203, 197, 0.12), transparent 28%),
                    linear-gradient(180deg, #eaf1f8 0%, var(--bg) 40%, var(--bg-deep) 100%);
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"] {
                background: transparent !important;
            }

            section[data-testid="stSidebar"],
            [data-testid="collapsedControl"] {
                display: none !important;
            }

            .block-container {
                max-width: 1360px;
                padding-top: 1.2rem;
                padding-bottom: 2.8rem;
            }

            .banner {
                background: linear-gradient(180deg, #ffffff 0%, var(--surface-tinted) 100%);
                border: 1.5px solid var(--line);
                border-radius: 24px;
                padding: 1.3rem 1.4rem;
                color: var(--text);
                box-shadow: var(--shadow), var(--shadow-inset);
                margin-bottom: 1rem;
                position: relative;
            }
            .banner::before {
                content: "";
                position: absolute;
                left: 18px;
                right: 18px;
                top: 0;
                height: 2.5px;
                background: linear-gradient(90deg, rgba(11, 94, 168, 0.0) 0%, rgba(11, 94, 168, 0.55) 30%, rgba(25, 203, 197, 0.55) 70%, rgba(25, 203, 197, 0.0) 100%);
                border-radius: 2px;
            }

            .banner-kicker {
                display: inline-block;
                background: rgba(11, 94, 168, 0.08);
                border: 1px solid rgba(11, 94, 168, 0.12);
                border-radius: 999px;
                padding: 0.34rem 0.72rem;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                margin-bottom: 0.8rem;
                color: var(--brand);
            }

            .banner h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.12;
                color: var(--text);
            }

            .banner p {
                margin: 0.7rem 0 0 0;
                max-width: 860px;
                line-height: 1.55;
                color: var(--muted);
            }

            .topbar-shell {
                display: flex;
                justify-content: flex-start;
                gap: 1rem;
                align-items: center;
                background: linear-gradient(180deg, #ffffff 0%, var(--surface-tinted) 100%);
                border: 1.5px solid var(--line);
                border-radius: 24px;
                padding: 1rem 1.2rem;
                box-shadow: var(--shadow), var(--shadow-inset);
                margin-bottom: 0.85rem;
                position: relative;
            }
            .topbar-shell::before {
                content: "";
                position: absolute;
                left: 18px;
                right: 18px;
                top: 0;
                height: 2px;
                background: linear-gradient(90deg, rgba(11, 94, 168, 0.0) 0%, rgba(11, 94, 168, 0.40) 35%, rgba(25, 203, 197, 0.45) 65%, rgba(25, 203, 197, 0.0) 100%);
                border-radius: 2px;
            }

            .brand-lockup {
                display: flex;
                align-items: center;
                gap: 1rem;
                min-width: 0;
            }

            .brand-logo {
                width: 240px;
                height: auto;
                object-fit: contain;
                flex: 0 0 auto;
            }

            .login-layout {
                min-height: calc(100vh - 3rem);
                display: flex;
                align-items: center;
            }

            .login-brand-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,253,0.98));
                border: 1px solid var(--line);
                border-radius: 28px;
                padding: 1.8rem 1.85rem;
                box-shadow: var(--shadow);
                min-height: 640px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            }

            .login-brand-lockup {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .login-logo {
                width: 420px;
                max-width: 100%;
                height: auto;
                object-fit: contain;
                margin-bottom: 0.4rem;
            }

            .login-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                width: fit-content;
                border-radius: 999px;
                padding: 0.32rem 0.72rem;
                background: rgba(11, 94, 168, 0.08);
                color: var(--brand);
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                border: 1px solid rgba(11, 94, 168, 0.12);
            }

            .login-title {
                font-size: 2rem;
                font-weight: 700;
                line-height: 1.08;
                color: var(--text);
                margin: 0;
                max-width: 520px;
            }

            .login-subtitle {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.55;
                max-width: 520px;
                margin-top: 0.5rem;
            }

            .trust-badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
                margin-top: 1.2rem;
            }

            .trust-badge {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.44rem 0.78rem;
                background: var(--surface);
                border: 1px solid var(--line);
                color: var(--text);
                font-size: 0.86rem;
                font-weight: 600;
            }

            .group-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 1.4rem;
            }

            .group-chip {
                background: rgba(11, 94, 168, 0.04);
                border: 1px solid rgba(11, 94, 168, 0.1);
                border-radius: 18px;
                padding: 0.85rem 0.95rem;
            }

            .group-chip-label {
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .group-chip-value {
                color: var(--text);
                font-size: 0.92rem;
                line-height: 1.45;
                font-weight: 600;
            }

            .group-chip-meta {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
                margin-top: 0.35rem;
            }

            .login-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1.35rem 1.4rem 1.2rem 1.4rem;
                box-shadow: var(--shadow);
                min-height: 640px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            }

            .login-card-header {
                margin-bottom: 1rem;
            }

            .login-card-title {
                margin: 0;
                color: var(--text);
                font-size: 1.45rem;
                font-weight: 700;
                line-height: 1.15;
            }

            .login-card-subtitle {
                margin-top: 0.45rem;
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.5;
            }

            .environment-tag {
                display: inline-flex;
                align-items: center;
                margin-bottom: 0.8rem;
                border-radius: 999px;
                padding: 0.34rem 0.72rem;
                background: rgba(25, 203, 197, 0.1);
                color: #0d6b69;
                border: 1px solid rgba(25, 203, 197, 0.16);
                font-size: 0.78rem;
                font-weight: 700;
            }

            .login-links {
                display: flex;
                gap: 1rem;
                align-items: center;
                margin-top: 0.5rem;
                margin-bottom: 1rem;
                font-size: 0.9rem;
            }

            .login-links a {
                color: var(--brand);
                text-decoration: none;
                font-weight: 600;
            }

            .security-notice {
                background: rgba(11, 94, 168, 0.04);
                border: 1px solid rgba(11, 94, 168, 0.12);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.9rem;
            }

            .security-notice strong {
                color: var(--text);
                display: block;
                margin-bottom: 0.3rem;
            }

            .login-helper {
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.5;
                margin-top: 0.7rem;
            }

            .login-divider {
                height: 1px;
                background: linear-gradient(90deg, rgba(11, 94, 168, 0.14), rgba(11, 94, 168, 0.04));
                margin: 1rem 0;
            }

            .brand-copy {
                min-width: 0;
            }

            .brand-title {
                font-size: 1.85rem;
                font-weight: 700;
                line-height: 1.1;
                color: var(--text);
                margin: 0;
            }

            .brand-subtitle {
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.45;
            }

            .session-rail {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                min-width: 420px;
            }

            .session-item {
                background: var(--surface-soft);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.85rem 0.9rem;
            }

            .session-label {
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .session-value {
                font-size: 1.02rem;
                font-weight: 700;
                color: var(--text);
                line-height: 1.3;
            }

            .session-meta {
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 0.3rem;
                line-height: 1.4;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.9rem;
                margin-bottom: 1rem;
            }

            .summary-grid.three {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .summary-grid.compact {
                margin-bottom: 0.8rem;
            }

            .login-shell {
                min-height: 100vh;
                display: flex;
                align-items: center;
            }

            .summary-tile {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                box-shadow: var(--shadow);
                min-height: 128px;
            }

            .summary-tile.slim {
                min-height: 108px;
                padding: 0.9rem 0.95rem;
            }

            .summary-label {
                font-size: 0.76rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--brand);
                margin-bottom: 0.42rem;
            }

            .summary-value {
                font-size: 1.5rem;
                font-weight: 700;
                line-height: 1.15;
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .summary-meta {
                color: var(--muted);
                line-height: 1.45;
                font-size: 0.88rem;
            }

            .action-shell {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 0.95rem 1.05rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.85rem;
            }

            .action-shell h4 {
                margin: 0;
                color: var(--brand-deep);
                font-size: 0.86rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .action-headline {
                color: var(--text);
                font-size: 1.2rem;
                font-weight: 700;
                line-height: 1.35;
                margin: 0.55rem 0 0.35rem 0;
            }

            .action-subtext {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.55;
                margin-bottom: 0.85rem;
            }

            .plain-list {
                margin: 0;
                padding-left: 1rem;
                color: var(--muted);
            }

            .plain-list li {
                margin-bottom: 0.35rem;
                line-height: 1.5;
            }

            .status-lines {
                display: flex;
                flex-direction: column;
                gap: 0.7rem;
            }

            .status-line {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: baseline;
                padding-bottom: 0.55rem;
                border-bottom: 1px solid rgba(214, 226, 236, 0.72);
            }

            .status-line:last-child {
                border-bottom: none;
                padding-bottom: 0;
            }

            .status-line-label {
                color: var(--muted);
                font-size: 0.86rem;
                font-weight: 600;
            }

            .status-line-value {
                color: var(--text);
                font-size: 0.96rem;
                font-weight: 700;
                text-align: right;
            }

            .workflow-progress {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 0.55rem;
                margin-bottom: 0.85rem;
            }

            .workflow-progress-step {
                background: linear-gradient(180deg, #ffffff 0%, var(--surface-tinted) 100%);
                border: 1.5px solid var(--line);
                border-radius: 16px;
                padding: 0.78rem 0.85rem;
                min-height: 92px;
                box-shadow: 0 1px 2px rgba(8, 70, 125, 0.04), 0 6px 14px rgba(8, 70, 125, 0.06);
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                transition: box-shadow 180ms ease, border-color 180ms ease, transform 180ms ease;
            }

            .workflow-progress-step.complete {
                background: linear-gradient(180deg, rgba(25, 203, 197, 0.14) 0%, rgba(25, 203, 197, 0.05) 100%);
                border-color: rgba(25, 203, 197, 0.40);
                box-shadow: 0 1px 2px rgba(13, 107, 105, 0.05), 0 8px 18px rgba(13, 107, 105, 0.10);
            }

            .workflow-progress-step.current {
                background: linear-gradient(180deg, rgba(11, 94, 168, 0.18) 0%, rgba(25, 203, 197, 0.08) 100%);
                border-color: rgba(11, 94, 168, 0.55);
                box-shadow: 0 3px 6px rgba(11, 94, 168, 0.12), 0 16px 34px rgba(11, 94, 168, 0.20);
                transform: translateY(-2px);
            }

            .workflow-progress-step.next {
                background: linear-gradient(180deg, rgba(11, 94, 168, 0.08) 0%, rgba(11, 94, 168, 0.02) 100%);
                border-color: rgba(11, 94, 168, 0.30);
            }

            .workflow-progress-step.future {
                background: rgba(255, 255, 255, 0.55);
                border-style: dashed;
                border-color: rgba(139, 163, 186, 0.55);
                opacity: 0.70;
            }

            .workflow-progress-marker {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                background: rgba(11, 94, 168, 0.08);
                color: var(--brand);
                font-weight: 800;
                font-size: 0.92rem;
                margin-bottom: 0.7rem;
            }

            .workflow-progress-step.complete .workflow-progress-marker {
                background: rgba(25, 203, 197, 0.16);
                color: #0d6b69;
            }

            .workflow-progress-step.current .workflow-progress-marker {
                background: var(--brand);
                color: white;
            }

            .workflow-progress-owner {
                color: var(--brand);
                font-size: 0.72rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.28rem;
                line-height: 1.25;
            }

            .workflow-progress-title {
                color: var(--brand-deep);
                font-size: 0.94rem;
                font-weight: 800;
                line-height: 1.22;
                margin-bottom: 0.24rem;
            }

            .workflow-progress-state {
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 700;
                margin-top: auto;
            }

            .request-file-card {
                background: linear-gradient(180deg, rgba(11, 94, 168, 0.05), rgba(25, 203, 197, 0.04));
                border: 1px solid rgba(11, 94, 168, 0.16);
                border-radius: 20px;
                padding: 1rem 1.05rem;
                margin-top: 0.9rem;
            }

            .request-file-kicker {
                color: var(--brand);
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.35rem;
            }

            .request-file-title {
                color: var(--text);
                font-size: 1.08rem;
                font-weight: 700;
                line-height: 1.35;
                word-break: break-word;
            }

            .request-file-meta {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.45;
                margin-top: 0.25rem;
            }

            .request-file-stats {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.7rem;
                margin-top: 0.9rem;
            }

            .request-file-stat {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(11, 94, 168, 0.08);
                border-radius: 14px;
                padding: 0.72rem 0.78rem;
            }

            .request-file-stat-label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.22rem;
            }

            .request-file-stat-value {
                color: var(--brand-deep);
                font-size: 0.95rem;
                font-weight: 800;
                line-height: 1.25;
            }

            .upload-callout {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.5;
                margin-top: 0.15rem;
            }

            .checklist {
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
                margin-top: 0.2rem;
            }

            .checklist-row {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .checklist-icon {
                width: 1.5rem;
                height: 1.5rem;
                border-radius: 999px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 0.78rem;
                font-weight: 800;
                flex: 0 0 auto;
                border: 1px solid var(--line);
                background: var(--surface);
                color: var(--muted);
            }

            .checklist-row.done .checklist-icon {
                background: var(--good-bg);
                border-color: rgba(19, 107, 72, 0.16);
                color: var(--good);
            }

            .checklist-row.done {
                color: var(--text);
            }

            .muted-note {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .section-kicker {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .section-shell {
                background: linear-gradient(180deg, #ffffff 0%, var(--surface-tinted) 100%);
                border: 1.5px solid var(--line);
                border-radius: 20px;
                padding: 1.05rem 1.2rem;
                box-shadow: var(--shadow), var(--shadow-inset);
                margin-bottom: 1rem;
                position: relative;
            }
            .section-shell::before {
                content: "";
                position: absolute;
                left: 16px;
                right: 16px;
                top: 0;
                height: 2px;
                background: linear-gradient(90deg, rgba(11, 94, 168, 0.0) 0%, rgba(11, 94, 168, 0.45) 30%, rgba(25, 203, 197, 0.50) 70%, rgba(25, 203, 197, 0.0) 100%);
                border-radius: 2px;
            }

            .section-shell h3 {
                margin: 0;
                font-size: 1.32rem;
                font-weight: 700;
                color: var(--text-strong);
                letter-spacing: -0.01em;
            }

            .section-shell p {
                margin: 0.45rem 0 0 0;
                color: var(--muted);
                line-height: 1.6;
            }

            .st-key-request_details_panel,
            .st-key-upload_task_panel,
            .st-key-submission_readiness_panel,
            .st-key-intake_summary_panel,
            .st-key-dataset_shape_panel,
            .st-key-data_preview_panel,
            .st-key-missing_heatmap_panel,
            .st-key-field_distributions_panel,
            .st-key-metadata_summary_panel,
            .st-key-package_summary_panel,
            .st-key-field_settings_panel,
            .st-key-generation_strategy_panel,
            .st-key-synthetic_preview_panel,
            .st-key-action_bar_panel,
            .st-key-submission_summary_panel,
            .st-key-approval_action_panel,
            .st-key-release_summary_panel,
            .st-key-download_panel {
                background: var(--surface) !important;
                border: 1px solid var(--line) !important;
                border-radius: 20px !important;
                box-shadow: var(--shadow) !important;
                padding: 1rem 1.15rem !important;
            }

            .st-key-request_details_panel {
                padding-bottom: 1.35rem !important;
            }

            .st-key-upload_task_panel {
                padding-bottom: 0.9rem !important;
            }

            .st-key-intake_summary_panel {
                padding-bottom: 1.45rem !important;
            }

            .st-key-dataset_shape_panel {
                padding-bottom: 1.4rem !important;
            }

            .st-key-data_preview_panel {
                padding-bottom: 1rem !important;
            }

            .st-key-missing_heatmap_panel {
                padding-bottom: 1.1rem !important;
            }

            .st-key-field_distributions_panel {
                padding-bottom: 1.2rem !important;
            }

            .st-key-metadata_summary_panel,
            .st-key-package_summary_panel,
            .st-key-submission_summary_panel,
            .st-key-release_summary_panel {
                padding-bottom: 1.4rem !important;
            }

            .st-key-field_settings_panel,
            .st-key-generation_strategy_panel,
            .st-key-synthetic_preview_panel,
            .st-key-action_bar_panel,
            .st-key-approval_action_panel,
            .st-key-download_panel {
                padding-bottom: 1.1rem !important;
            }

            .st-key-request_details_panel > div,
            .st-key-upload_task_panel > div,
            .st-key-submission_readiness_panel > div,
            .st-key-intake_summary_panel > div,
            .st-key-dataset_shape_panel > div,
            .st-key-data_preview_panel > div,
            .st-key-missing_heatmap_panel > div,
            .st-key-field_distributions_panel > div,
            .st-key-metadata_summary_panel > div,
            .st-key-package_summary_panel > div,
            .st-key-field_settings_panel > div,
            .st-key-generation_strategy_panel > div,
            .st-key-synthetic_preview_panel > div,
            .st-key-action_bar_panel > div,
            .st-key-submission_summary_panel > div,
            .st-key-approval_action_panel > div,
            .st-key-release_summary_panel > div,
            .st-key-download_panel > div {
                background: transparent !important;
            }

            .st-key-request_details_panel [data-testid="stTextInput"] {
                margin-bottom: 0.45rem !important;
            }

            .pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.28rem 0.62rem;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-right: 0.4rem;
                margin-bottom: 0.35rem;
                border: 1px solid var(--line);
                background: var(--surface-soft);
                color: var(--brand-deep);
            }

            .state-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1rem 1.05rem;
                box-shadow: var(--shadow);
                min-height: 148px;
            }

            .state-card h4 {
                margin: 0;
                font-size: 0.88rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .state-value {
                font-size: 1.65rem;
                font-weight: 700;
                color: var(--brand);
                margin-top: 0.55rem;
                margin-bottom: 0.25rem;
                line-height: 1.15;
            }

            .state-text {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.94rem;
            }

            .status-good,
            .status-warn,
            .status-bad {
                display: inline-block;
                margin-top: 0.55rem;
                border-radius: 999px;
                padding: 0.22rem 0.6rem;
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .status-good {
                background: var(--good-bg);
                color: var(--good);
                border: 1px solid rgba(22, 112, 74, 0.22);
            }

            .status-warn {
                background: var(--warn-bg);
                color: var(--warn);
                border: 1px solid rgba(138, 97, 22, 0.22);
            }

            .status-bad {
                background: var(--danger-bg);
                color: var(--danger);
                border: 1px solid rgba(143, 45, 53, 0.22);
            }

            .step-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 0.95rem 0.95rem 0.85rem 0.95rem;
                box-shadow: var(--shadow);
                min-height: 156px;
            }

            .step-card.active {
                border-color: rgba(28, 216, 211, 0.5);
                box-shadow: 0 16px 32px rgba(15, 79, 149, 0.12);
            }

            .step-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 14px;
                background: rgba(15, 79, 149, 0.08);
                color: var(--brand);
                font-weight: 700;
                margin-bottom: 0.6rem;
            }

            .step-title {
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .step-description {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.9rem;
                min-height: 98px;
                margin-bottom: 0.7rem;
            }

            .step-action-note {
                color: var(--muted);
                font-size: 0.8rem;
                font-weight: 600;
                margin-top: 0.15rem;
            }

            .note-card {
                background: var(--surface-soft);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                line-height: 1.6;
                color: var(--muted);
                margin-bottom: 0.75rem;
            }

            .role-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: var(--shadow);
                min-height: 124px;
            }

            .role-card strong {
                display: block;
                color: var(--text);
                margin-bottom: 0.35rem;
            }

            .approval-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem;
                box-shadow: var(--shadow);
                min-height: 182px;
            }

            .approval-level {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.4rem;
            }

            .approval-card h4 {
                margin: 0 0 0.35rem 0;
                color: var(--text);
                font-size: 1.02rem;
            }

            .approval-owner {
                color: var(--muted);
                font-size: 0.88rem;
                margin-bottom: 0.55rem;
            }

            .approval-detail {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.9rem;
                margin-top: 0.6rem;
            }

            .bullet-list {
                margin: 0.2rem 0 0 0;
                padding-left: 1rem;
                color: var(--muted);
            }

            .bullet-list li {
                margin-bottom: 0.32rem;
                line-height: 1.5;
            }

            .mini-label {
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: var(--brand);
                margin-bottom: 0.35rem;
            }

            .steps-wrap {
                margin-top: 0.35rem;
            }

            .step-pill-compact {
                display: inline-block;
                margin-right: 0.32rem;
                margin-bottom: 0.38rem;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                border: 1px solid rgba(11, 94, 168, 0.14);
                background: rgba(11, 94, 168, 0.06);
                color: var(--brand);
                font-size: 0.78rem;
                font-weight: 700;
            }

            .step-nav-meta {
                text-align: center;
                margin-top: 0.15rem;
            }

            .step-nav-owner {
                color: var(--brand-deep);
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .step-nav-status {
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 700;
                margin-top: 0.12rem;
            }

            .workflow-step-card,
            .status-flow-row {
                display: none;
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.7rem 0.85rem;
                box-shadow: var(--shadow);
                min-height: 116px;
            }

            div[data-testid="stMetric"] * {
                color: var(--text) !important;
            }

            div.stButton > button,
            div.stDownloadButton > button {
                min-height: 2.8rem;
                border-radius: 12px;
                border: 1px solid rgba(11, 94, 168, 0.18);
                background: var(--surface) !important;
                color: var(--text) !important;
                font-weight: 700;
                box-shadow: none !important;
            }

            div.stButton > button[kind="primary"] {
                background: var(--brand) !important;
                color: #ffffff !important;
                border: 1px solid rgba(8, 70, 125, 0.28) !important;
            }

            div[data-testid="stTextInput"] label p,
            div[data-testid="stSelectbox"] label p,
            div[data-testid="stCheckbox"] label p {
                color: var(--text) !important;
                font-weight: 600 !important;
            }

            div[data-testid="stTextInput"] input {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
            }

            div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
                min-height: 3rem !important;
                border-radius: 14px !important;
                border: 1px solid var(--line) !important;
                background: #fbfdff !important;
            }

            div[data-baseweb="button-group"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.35rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.45rem;
            }

            div[data-baseweb="button-group"] button {
                border-radius: 12px !important;
                border: 1px solid transparent !important;
                color: var(--brand-deep) !important;
                font-weight: 800 !important;
                min-height: 3.35rem !important;
                font-size: 1.02rem !important;
                padding: 0.8rem 0.95rem !important;
            }

            div[data-baseweb="button-group"] button[aria-pressed="true"] {
                background: linear-gradient(180deg, var(--brand), var(--brand-deep)) !important;
                color: #ffffff !important;
                box-shadow: 0 10px 20px rgba(11, 94, 168, 0.18) !important;
            }

            div[data-baseweb="button-group"] button:hover {
                background: rgba(11, 94, 168, 0.08) !important;
            }

            div[data-baseweb="select"] > div,
            div[data-testid="stTextArea"] textarea {
                border-radius: 14px;
                background: var(--surface) !important;
                color: var(--text) !important;
                -webkit-text-fill-color: var(--text) !important;
                border: 1px solid var(--line) !important;
            }

            /* Universal text field fix:
               BaseWeb clips borders when height/border are applied to the inner input.
               Put the frame on the wrapper and let the input stay borderless. */
            div[data-testid="stTextInput"] [data-baseweb="input"],
            div[data-testid="stNumberInput"] [data-baseweb="input"] {
                min-height: 3.15rem !important;
                border-radius: 14px !important;
                border: 0 !important;
                background: var(--surface) !important;
                box-shadow: inset 0 0 0 1px var(--line) !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
                background-clip: padding-box !important;
            }

            div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
            div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
                box-shadow:
                    inset 0 0 0 1px rgba(11, 94, 168, 0.62),
                    0 0 0 2px rgba(11, 94, 168, 0.1) !important;
            }

            div[data-testid="stTextInput"] [data-baseweb="input"] > div,
            div[data-testid="stNumberInput"] [data-baseweb="input"] > div {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
                min-height: calc(3.15rem - 2px) !important;
                display: flex !important;
                align-items: center !important;
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input {
                min-height: auto !important;
                height: auto !important;
                border: 0 !important;
                border-radius: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
                padding: 0 0.9rem !important;
                line-height: 1.35 !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stFileUploader"] section {
                background: var(--surface) !important;
                border: 1px solid var(--line) !important;
                border-radius: 18px !important;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                background: var(--surface) !important;
                border-radius: 18px;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: linear-gradient(180deg, #ffffff 0%, var(--surface-tinted) 100%) !important;
                border: 1.5px solid var(--line) !important;
                border-radius: 18px;
                box-shadow: var(--shadow), var(--shadow-inset);
                padding: 0.55rem 0.7rem 0.85rem 0.7rem;
                transition: box-shadow 180ms ease, border-color 180ms ease, transform 180ms ease;
                position: relative;
            }
            /* Subtle brand-colored accent line along the top edge for visual anchoring */
            div[data-testid="stVerticalBlockBorderWrapper"]::before {
                content: "";
                position: absolute;
                left: 14px;
                right: 14px;
                top: 0;
                height: 2px;
                background: linear-gradient(90deg, rgba(11, 94, 168, 0.0) 0%, rgba(11, 94, 168, 0.35) 30%, rgba(25, 203, 197, 0.40) 70%, rgba(25, 203, 197, 0.0) 100%);
                border-radius: 2px;
                pointer-events: none;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                box-shadow: var(--shadow-lift), var(--shadow-inset);
                border-color: #8cacc8;
            }
            /* Inner blocks transparent so the container gradient shows through */
            html body div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"],
            html body div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
                background: transparent !important;
            }
            html body div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="element-container"],
            html body div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] {
                background: transparent !important;
            }

            div[data-testid="stDataFrame"] *,
            div[data-testid="stTable"] * {
                color: var(--text) !important;
            }

            .control-card-kicker {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
                color: var(--brand);
                margin-bottom: 0.3rem;
            }

            .control-card-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.28rem;
            }

            .control-card-text {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
                margin-bottom: 0.9rem;
            }

            .control-compact-note {
                color: var(--muted);
                line-height: 1.5;
                font-size: 0.9rem;
                margin-top: 0.2rem;
            }

            .generate-action-card {
                background: linear-gradient(180deg, rgba(15, 79, 149, 0.05), rgba(28, 216, 211, 0.04)), var(--surface);
                border: 1px solid rgba(15, 79, 149, 0.16);
                border-radius: 20px;
                padding: 1rem 1rem 0.35rem 1rem;
                margin-bottom: 0.8rem;
            }

            .generate-action-title {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.28rem;
            }

            .generate-action-text {
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
                margin-bottom: 0.75rem;
            }

            @media (max-width: 1100px) {
                .topbar-shell {
                    flex-direction: column;
                }

                .session-rail,
                .summary-grid,
                .summary-grid.three,
                .group-strip,
                .workflow-progress {
                    grid-template-columns: 1fr;
                    min-width: 0;
                }

                .brand-lockup {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .brand-logo {
                    width: 240px;
                }

                .login-brand-card,
                .login-card {
                    min-height: auto;
                }

                .login-links {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.5rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes)).replace(r"^\s*$", pd.NA, regex=True)


@st.cache_data(show_spinner=False)
def load_sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH).replace(r"^\s*$", pd.NA, regex=True)


def build_generation_signature(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    return json.dumps({"metadata": metadata, "controls": controls}, sort_keys=True, default=str)


def build_metadata_signature(metadata: list[dict[str, Any]]) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)


def has_permission(permission: str) -> bool:
    role = st.session_state.get("current_role")
    if not role:
        return False
    return permission in ROLE_CONFIGS[role]["permissions"]


def current_role_summary() -> str:
    role = st.session_state.get("current_role", "")
    return ROLE_CONFIGS.get(role, {}).get("summary", "")


def current_role_group() -> str:
    role = st.session_state.get("current_role", "")
    return ROLE_TO_GROUP.get(role, "Users")


def visible_steps_for_role(role: str | None = None) -> list[int]:
    active_role = role or st.session_state.get("current_role")
    if not active_role:
        return list(range(len(STEP_CONFIG)))
    return ROLE_VISIBLE_STEPS.get(active_role, list(range(len(STEP_CONFIG))))


def format_timestamp() -> str:
    return datetime.now().strftime("%H:%M")


def record_audit_event(event: str, detail: str, status: str = "Logged") -> None:
    if "audit_events" not in st.session_state:
        st.session_state.audit_events = []
    st.session_state.audit_events.insert(
        0,
        {
            "time": format_timestamp(),
            "actor": st.session_state.get("current_role", "System"),
            "event": agent_event_label(event),
            "detail": detail,
            "status": status,
        },
    )
    st.session_state.audit_events = st.session_state.audit_events[:30]
    persist_shared_workspace_state()


def clear_generation_outputs() -> None:
    st.session_state.synthetic_df = None
    st.session_state.generation_summary = None
    st.session_state.validation = None
    st.session_state.last_generation_signature = None
    st.session_state.release_status = "Not ready"
    st.session_state.export_requested_by = None
    st.session_state.export_policy_approved_by = None
    st.session_state.export_approved_by = None
    st.session_state.results_shared_by = None
    st.session_state.results_shared_at = None


def set_source_dataframe(df: pd.DataFrame, label: str) -> None:
    existing_file_size = st.session_state.get("source_file_size")
    profile = profile_dataframe(df)
    hygiene = review_hygiene(df, profile)
    metadata = build_metadata(df, profile)

    st.session_state.source_df = df
    st.session_state.source_label = label
    st.session_state.source_file_size = existing_file_size
    st.session_state.profile = profile
    st.session_state.hygiene = hygiene
    st.session_state.metadata_editor_df = metadata_to_editor_frame(metadata)
    st.session_state.controls = default_generation_controls(len(df))
    st.session_state.intake_confirmed = False
    st.session_state.hygiene_reviewed = False
    st.session_state.settings_reviewed = False
    st.session_state.settings_review_signature = None
    st.session_state.metadata_status = "Draft"
    st.session_state.metadata_submitted_by = None
    st.session_state.metadata_submitted_at = None
    st.session_state.metadata_approved_by = None
    st.session_state.metadata_approved_at = None
    st.session_state.metadata_review_note = None
    st.session_state.metadata_reviewed_by = None
    st.session_state.metadata_reviewed_at = None
    st.session_state.last_reviewed_metadata_signature = None
    st.session_state.last_cleaning_actions = []
    st.session_state.current_metadata_package_id = None
    st.session_state.metadata_package_log = []
    st.session_state.metadata_has_unsubmitted_changes = False
    clear_generation_outputs()
    st.session_state.current_step = 0
    record_audit_event("Dataset loaded", label, status="Logged")


def initialize_app_state() -> None:
    defaults = {
        "authenticated": False,
        "current_role": None,
        "current_user_email": None,
        "project_purpose": "",
        "source_file_size": None,
        "shared_workspace_loaded": False,
        "current_step": 0,
        "audit_events": [],
        "release_status": "Not ready",
        "export_requested_by": None,
        "export_policy_approved_by": None,
        "export_approved_by": None,
        "results_shared_by": None,
        "results_shared_at": None,
        "uploaded_signature": None,
        "metadata_submitted_at": None,
        "metadata_approved_at": None,
        "metadata_review_note": None,
        "metadata_reviewed_by": None,
        "metadata_reviewed_at": None,
        "current_metadata_package_id": None,
        "metadata_package_log": [],
        "metadata_has_unsubmitted_changes": False,
        "settings_reviewed": False,
        "settings_review_signature": None,
        "request_registry": [],
        "active_request_id": None,
        "pending_active_request_selector": None,
        "pending_queue_clear": False,
        "next_request_number": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.shared_workspace_loaded:
        load_shared_workspace_state()
        st.session_state.shared_workspace_loaded = True


def ensure_dataset_loaded() -> None:
    if not st.session_state.get("request_registry"):
        create_blank_request()
    elif "source_df" not in st.session_state and st.session_state.get("active_request_id"):
        restore_request_workspace(st.session_state["active_request_id"])


def process_pending_workspace_actions() -> None:
    if st.session_state.get("pending_queue_clear"):
        st.session_state.pending_queue_clear = False
        clear_request_queue()


def effective_release_status(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    if not has_active_dataset():
        return "Awaiting dataset"
    if not st.session_state.intake_confirmed:
        return "Uploaded"
    if not st.session_state.hygiene_reviewed:
        return "Scanned"
    if not st.session_state.settings_reviewed:
        return "Reviewed"
    if st.session_state.metadata_status == "In Review":
        return "Pending approval"
    if st.session_state.metadata_status == "Changes Requested":
        return "Changes requested"
    if st.session_state.metadata_status == "Rejected":
        return "Rejected"
    if st.session_state.metadata_status != "Approved":
        return "Reviewed"
    if st.session_state.synthetic_df is None:
        return "Approved"
    if has_stale_generation(metadata, controls):
        return "Regeneration required"
    if st.session_state.results_shared_at:
        return "Shared"
    return "Generated"


def metadata_sensitivity(item: dict[str, Any]) -> str:
    lowered = item["column"].lower()
    if item["data_type"] == "identifier":
        return "Restricted"
    if "postal" in lowered or item["data_type"] == "date":
        return "Sensitive"
    if "complaint" in lowered or "note" in lowered or "text" in lowered:
        return "Sensitive"
    return "Operational"


def metadata_owner(item: dict[str, Any]) -> str:
    sensitivity = metadata_sensitivity(item)
    if sensitivity in {"Restricted", "Sensitive"}:
        return "Manager / Reviewer"
    return "Data Analyst"


def metadata_handling(item: dict[str, Any]) -> str:
    lowered = item["column"].lower()
    if not item["include"]:
        return "Excluded from generation."
    action = item.get("control_action", "")
    if action == "Tokenize" or item["data_type"] == "identifier":
        return "Replace with surrogate token before synthetic export."
    if action == "Month only":
        return "Release month-level timing only instead of exact dates."
    if action == "Date shift" or item["data_type"] == "date":
        return "Apply controlled date jitter before release."
    if action == "Coarse geography" or "postal" in lowered:
        return "Keep only coarse geography in synthetic output."
    if action == "Group text":
        return "Collapse free text into safer grouped categories before sampling."
    if action == "Group rare categories":
        return "Keep major categories and group low-frequency labels into 'Other'."
    if action == "Clip extremes":
        return "Clip extreme numeric values before sampling to reduce outlier leakage."
    if "complaint" in lowered:
        return "Normalize or group text before sampling."
    if item["strategy"] == "sample_plus_noise":
        return "Sample source behavior with bounded noise."
    return "Sample approved categories while preserving broad distribution."


def metadata_status_for_row(item: dict[str, Any]) -> str:
    if not item["include"]:
        return "Excluded"
    if st.session_state.metadata_status == "Approved":
        return "Approved"
    if st.session_state.metadata_status == "In Review":
        return "In review"
    if st.session_state.metadata_status == "Changes Requested":
        return "Changes requested"
    if st.session_state.metadata_status == "Rejected":
        return "Rejected"
    return "Draft"


def build_metadata_review_frame(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        rows.append(
            {
                "Field": item["column"],
                "Field type": item["data_type"].replace("_", " ").title(),
                "Sensitivity": metadata_sensitivity(item),
                "Field action": item.get("control_action", "Preserve"),
                "Generation rule": item["strategy"].replace("_", " ").title(),
                "Owner": metadata_owner(item),
                "Status": metadata_status_for_row(item),
                "Handling": metadata_handling(item),
            }
        )
    return pd.DataFrame(rows)


def build_phi_detection_frame(profile: dict[str, Any], metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        column = item["column"]
        details = profile["columns"][column]
        lowered = column.lower()
        if details["semantic_role"] == "identifier":
            rows.append(
                {
                    "Field": column,
                    "Detection": "Direct identifier",
                    "Why flagged": "Encounter-level ID pattern detected.",
                    "Control point": "Restricted to raw workspace; replaced with surrogate tokens after approval.",
                }
            )
        elif details["semantic_role"] == "date":
            rows.append(
                {
                    "Field": column,
                    "Detection": "Date or timing field",
                    "Why flagged": "Exact encounter timing can support re-identification when combined with other fields.",
                    "Control point": "Dates are jittered after governance approval and before synthetic release.",
                }
            )
        elif "postal" in lowered:
            rows.append(
                {
                    "Field": column,
                    "Detection": "Geographic quasi-identifier",
                    "Why flagged": "Location fields increase linkage risk when combined with dates or demographics.",
                    "Control point": "Only coarse geography is preserved in the synthetic dataset.",
                }
            )
        elif "complaint" in lowered or "note" in lowered or "text" in lowered:
            rows.append(
                {
                    "Field": column,
                    "Detection": "Free-text clinical context",
                    "Why flagged": "Free text can contain inconsistent or unexpectedly identifying detail.",
                    "Control point": "Normalize or group before sampling and keep under governance review.",
                }
            )

    if not rows:
        rows.append(
            {
                "Field": "No high-risk field detected",
                "Detection": "Operational schema",
                "Why flagged": "This sample contains no obvious direct PHI columns beyond identifiers.",
                "Control point": "Continue with metadata review and validation.",
            }
        )

    return pd.DataFrame(rows)


def build_missingness_strategy_frame(profile: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for column, details in profile["columns"].items():
        missing_pct = float(details["missing_pct"])
        if missing_pct <= 0:
            continue

        role = details["semantic_role"]
        if missing_pct >= 25:
            strategy = "Escalate before approval"
            rationale = "High missingness can distort downstream analysis and should be reviewed explicitly."
        elif role in {"categorical", "binary"}:
            strategy = "Preserve as explicit missing / unknown"
            rationale = "Keeping gaps visible helps analysts understand sparse operational coding."
        elif role == "numeric":
            strategy = "Preserve pattern unless business rule exists"
            rationale = "Avoid silent filling unless the data owner agrees on an imputation rule."
        elif role == "date":
            strategy = "Preserve timing gaps"
            rationale = "Missing dates can be operationally meaningful and should stay visible in validation."
        else:
            strategy = "Review and document"
            rationale = "Track the gap and decide handling during metadata review."

        rows.append(
            {
                "Field": column,
                "Missing %": missing_pct,
                "Recommended strategy": strategy,
                "Why": rationale,
            }
        )

    if not rows:
        rows.append(
            {
                "Field": "No incomplete field",
                "Missing %": 0.0,
                "Recommended strategy": "No action required",
                "Why": "The current dataset does not contain missing values.",
            }
        )

    return pd.DataFrame(rows).sort_values(by="Missing %", ascending=False, ignore_index=True)


def build_hygiene_option_defaults(hygiene: dict[str, Any]) -> dict[str, bool]:
    concerns = {issue["concern"] for issue in hygiene.get("issues", [])}
    findings = " ".join(issue["finding"].lower() for issue in hygiene.get("issues", []))
    return {
        "standardize_blank_strings": "Missingness" in concerns,
        "remove_duplicates": "Duplicate rows" in concerns,
        "normalize_categories": "Category normalization" in concerns,
        "fill_operational_gaps": "Missingness" in concerns,
        "fix_negative_values": "Negative values" in concerns,
        "repair_invalid_dates": "Invalid dates" in concerns,
        "cap_numeric_extremes": "Outliers" in concerns or "extreme" in findings,
        "group_rare_categories": False,
    }


def summarize_dataframe_change(original_df: pd.DataFrame, updated_df: pd.DataFrame) -> dict[str, int]:
    common_rows = min(len(original_df), len(updated_df))
    common_columns = [column for column in original_df.columns if column in updated_df.columns]
    changed_cells = 0
    changed_columns = 0
    if common_rows and common_columns:
        original_view = original_df.loc[: common_rows - 1, common_columns].reset_index(drop=True).astype("string").fillna("<NA>")
        updated_view = updated_df.loc[: common_rows - 1, common_columns].reset_index(drop=True).astype("string").fillna("<NA>")
        differences = original_view.ne(updated_view)
        changed_cells = int(differences.sum().sum()) + abs(len(original_df) - len(updated_df)) * len(common_columns)
        changed_columns = int(differences.any(axis=0).sum())
    return {
        "rows_before": int(len(original_df)),
        "rows_after": int(len(updated_df)),
        "changed_cells": int(changed_cells),
        "changed_columns": int(changed_columns),
        "missing_before": int(original_df.isna().sum().sum()),
        "missing_after": int(updated_df.isna().sum().sum()),
    }


def summarize_metadata_package(metadata: list[dict[str, Any]]) -> dict[str, int]:
    included = sum(1 for item in metadata if item["include"])
    excluded = sum(1 for item in metadata if not item["include"])
    restricted = sum(1 for item in metadata if item["include"] and metadata_sensitivity(item) == "Restricted")
    sensitive = sum(1 for item in metadata if item["include"] and metadata_sensitivity(item) == "Sensitive")
    targeted = sum(1 for item in metadata if item["include"] and item.get("control_action", "Preserve") != "Preserve")
    return {
        "included_fields": included,
        "excluded_fields": excluded,
        "restricted_fields": restricted,
        "sensitive_fields": sensitive,
        "targeted_actions": targeted,
    }


def register_metadata_submission(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    package_id = f"PKG-{datetime.now().strftime('%H%M%S')}-{len(st.session_state.metadata_package_log) + 1:02d}"
    record = {
        "package_id": package_id,
        "signature": build_metadata_signature(metadata),
        "snapshot": metadata,
        "summary": summarize_metadata_package(metadata),
        "submitted_by": st.session_state.current_role,
        "submitted_at": format_timestamp(),
        "approved_by": None,
        "approved_at": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "review_note": None,
        "status": "In Review",
    }
    st.session_state.metadata_package_log.insert(0, record)
    st.session_state.current_metadata_package_id = package_id
    return record


def register_metadata_approval(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    signature = build_metadata_signature(metadata)
    target_id = st.session_state.get("current_metadata_package_id")
    for record in st.session_state.metadata_package_log:
        if (target_id and record["package_id"] == target_id) or record["signature"] == signature:
            record["approved_by"] = st.session_state.current_role
            record["approved_at"] = format_timestamp()
            record["reviewed_by"] = st.session_state.current_role
            record["reviewed_at"] = format_timestamp()
            record["status"] = "Approved"
            st.session_state.current_metadata_package_id = record["package_id"]
            return record

    record = register_metadata_submission(metadata)
    record["approved_by"] = st.session_state.current_role
    record["approved_at"] = format_timestamp()
    record["reviewed_by"] = st.session_state.current_role
    record["reviewed_at"] = format_timestamp()
    record["status"] = "Approved"
    return record


def register_metadata_feedback(metadata: list[dict[str, Any]], outcome: str, note: str) -> dict[str, Any] | None:
    signature = build_metadata_signature(metadata)
    target_id = st.session_state.get("current_metadata_package_id")
    for record in st.session_state.metadata_package_log:
        if (target_id and record["package_id"] == target_id) or record["signature"] == signature:
            record["status"] = outcome
            record["reviewed_by"] = st.session_state.current_role
            record["reviewed_at"] = format_timestamp()
            record["review_note"] = note.strip()
            return record
    return None


def active_metadata_package_record(metadata: list[dict[str, Any]]) -> dict[str, Any] | None:
    target_id = st.session_state.get("current_metadata_package_id")
    if target_id:
        for record in st.session_state.metadata_package_log:
            if record["package_id"] == target_id:
                return record

    signature = build_metadata_signature(metadata)
    for record in st.session_state.metadata_package_log:
        if record["signature"] == signature:
            return record
    return st.session_state.metadata_package_log[0] if st.session_state.metadata_package_log else None


def current_review_package_record() -> dict[str, Any] | None:
    target_id = st.session_state.get("current_metadata_package_id")
    if target_id:
        for record in st.session_state.metadata_package_log:
            if record["package_id"] == target_id and record["status"] == "In Review":
                return record
    for record in st.session_state.metadata_package_log:
        if record["status"] == "In Review":
            return record
    return None


def build_metadata_package_log_frame() -> pd.DataFrame:
    rows = []
    for record in st.session_state.metadata_package_log:
        summary = record["summary"]
        rows.append(
            {
                "Package": record["package_id"],
                "Status": record["status"],
                "Submitted by": record["submitted_by"],
                "Submitted at": record["submitted_at"],
                "Approved by": record["approved_by"] or "Not approved",
                "Approved at": record["approved_at"] or "Not approved",
                "Reviewed by": record.get("reviewed_by") or "Not reviewed",
                "Review note": record.get("review_note") or "None",
                "Included fields": summary["included_fields"],
                "Restricted": summary["restricted_fields"],
                "Targeted actions": summary["targeted_actions"],
            }
        )
    return pd.DataFrame(rows)


def has_unsubmitted_metadata_changes(metadata: list[dict[str, Any]]) -> bool:
    reviewed_signature = st.session_state.get("last_reviewed_metadata_signature")
    if not reviewed_signature:
        return False
    return build_metadata_signature(metadata) != reviewed_signature


def metadata_display_status(metadata: list[dict[str, Any]]) -> str:
    if has_unsubmitted_metadata_changes(metadata):
        return "Draft changes"
    return st.session_state.metadata_status


def build_work_in_progress_frame(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> pd.DataFrame:
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()
    if not has_active_dataset():
        request_status = "Awaiting upload"
        request_owner = "Data Analyst"
        request_note = "No source dataset is in the workspace yet."
    else:
        request_status = "Uploaded" if st.session_state.intake_confirmed else "Draft"
        request_owner = "Data Analyst"
        request_note = current_dataset_label()

    if review_package is not None:
        review_status = f"{review_package['package_id']} pending"
        review_owner = "Manager / Reviewer"
        review_note = f"Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}."
    elif active_package is not None and active_package["status"] == "Approved":
        review_status = f"{active_package['package_id']} approved"
        review_owner = "Manager / Reviewer"
        review_note = f"Approved by {active_package['approved_by']} at {active_package['approved_at']}."
    elif active_package is not None and active_package["status"] in {"Changes Requested", "Rejected"}:
        review_status = f"{active_package['package_id']} · {active_package['status']}"
        review_owner = "Data Analyst"
        review_note = active_package.get("review_note") or "Revision is required before resubmission."
    elif st.session_state.settings_reviewed:
        review_status = "Ready to submit"
        review_owner = "Data Analyst"
        review_note = "Data settings are ready to be sent for manager review."
    else:
        review_status = "Pending settings review"
        review_owner = "Data Analyst"
        review_note = "Finish reviewing scan findings and field settings first."

    if st.session_state.synthetic_df is None:
        output_status = "Not generated"
        output_owner = "System"
        output_note = "Synthetic output will appear after the request is approved."
    elif has_stale_generation(metadata, controls):
        output_status = "Out of date"
        output_owner = "Data Analyst"
        output_note = "Settings changed after generation. Run generation again."
    elif st.session_state.results_shared_at:
        output_status = "Shared"
        output_owner = "Data Analyst"
        output_note = f"Marked shared by {st.session_state.results_shared_by} at {st.session_state.results_shared_at}."
    else:
        output_status = "Generated"
        output_owner = "Data Analyst"
        output_note = "Synthetic output is ready for final review and download."

    rows = [
        {
            "Work item": "Current request",
            "Status": request_status,
            "Current owner": request_owner,
            "Latest update": request_note,
        },
        {
            "Work item": "Review queue",
            "Status": review_status,
            "Current owner": review_owner,
            "Latest update": review_note,
        },
        {
            "Work item": "Final output",
            "Status": output_status,
            "Current owner": output_owner,
            "Latest update": output_note,
        },
    ]
    return pd.DataFrame(rows)


def build_role_access_frame() -> pd.DataFrame:
    rows = []
    for role, config in ROLE_CONFIGS.items():
        permissions = config["permissions"]
        rows.append(
            {
                "Stakeholder group": ROLE_TO_GROUP[role],
                "Role": role,
                "Visible workflow": ", ".join(STEP_CONFIG[index]["title"] for index in visible_steps_for_role(role)),
                "Raw data access": "Yes" if "view_raw" in permissions else "No",
                "Metadata edits": "Yes" if "edit_metadata" in permissions else "No",
                "Metadata approval": "Yes" if "approve_metadata" in permissions else "No",
                "Synthetic generation": "Yes" if "generate" in permissions else "No",
                "Controlled release review": "Yes" if "approve_release_policy" in permissions else "No",
                "Final export approval": "Yes" if "approve_export" in permissions else "No",
            }
        )
    return pd.DataFrame(rows)


def field_action_options(item: dict[str, Any]) -> list[str]:
    lowered = item["column"].lower()
    role = item["data_type"]
    if role == "identifier":
        return ["Tokenize", "Exclude"]
    if role == "date":
        return ["Date shift", "Month only", "Exclude"]
    if "postal" in lowered or "address" in lowered:
        return ["Coarse geography", "Exclude", "Preserve"]
    if "complaint" in lowered or "note" in lowered or "text" in lowered:
        return ["Group text", "Exclude", "Preserve"]
    if role == "numeric":
        return ["Preserve", "Clip extremes", "Exclude"]
    return ["Preserve", "Group rare categories", "Exclude"]


ALL_CONTROL_ACTIONS = [
    "Preserve",
    "Tokenize",
    "Date shift",
    "Month only",
    "Coarse geography",
    "Group text",
    "Group rare categories",
    "Clip extremes",
    "Exclude",
]


def sanitize_control_action(item: dict[str, Any], chosen_action: str) -> str:
    allowed = field_action_options(item)
    if chosen_action in allowed:
        return chosen_action
    return allowed[0]


def normalize_metadata_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    action = sanitize_control_action(normalized, normalized.get("control_action", "Preserve") or "Preserve")
    normalized["control_action"] = action

    if action == "Exclude":
        normalized["include"] = False
        normalized["notes"] = "Excluded from the synthetic package during metadata review."
        return normalized

    normalized["include"] = True
    if action == "Tokenize":
        normalized["data_type"] = "identifier"
        normalized["strategy"] = "new_token"
    elif action in {"Date shift", "Month only"}:
        normalized["data_type"] = "date"
        normalized["strategy"] = "sample_plus_jitter"
    elif action in {"Coarse geography", "Group text", "Group rare categories"}:
        normalized["strategy"] = "sample_category"
    elif action == "Clip extremes":
        normalized["strategy"] = "sample_plus_noise"

    return normalized


def normalize_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = [normalize_metadata_item(item) for item in editor_frame_to_metadata(frame)]
    normalized = metadata_to_editor_frame(metadata)
    normalized.index = frame.index
    return normalized


def apply_bulk_metadata_profile(mode: str) -> None:
    frame = st.session_state.metadata_editor_df.copy()
    for index, row in frame.iterrows():
        column_name = str(row["column"]).lower()
        role = str(row["data_type"])
        if mode == "tighten_phi":
            if role == "identifier":
                frame.at[index, "control_action"] = "Exclude"
            elif role == "date":
                frame.at[index, "control_action"] = "Month only"
            elif "postal" in column_name or "address" in column_name:
                frame.at[index, "control_action"] = "Coarse geography"
            elif "complaint" in column_name or "note" in column_name or "text" in column_name:
                frame.at[index, "control_action"] = "Group text"
        elif mode == "preserve_analytics":
            if role == "identifier":
                frame.at[index, "control_action"] = "Tokenize"
            elif role == "date":
                frame.at[index, "control_action"] = "Date shift"
            elif role == "numeric":
                frame.at[index, "control_action"] = "Preserve"
            elif "postal" in column_name or "address" in column_name:
                frame.at[index, "control_action"] = "Coarse geography"
            elif "complaint" in column_name or "note" in column_name or "text" in column_name:
                frame.at[index, "control_action"] = "Group text"
            else:
                frame.at[index, "control_action"] = "Preserve"
        elif mode == "reset_defaults":
            metadata_defaults = build_metadata(st.session_state.source_df, st.session_state.profile)
            st.session_state.metadata_editor_df = metadata_to_editor_frame(metadata_defaults)
            persist_shared_workspace_state()
            return

    st.session_state.metadata_editor_df = normalize_metadata_frame(frame)
    persist_shared_workspace_state()


def apply_generation_preset(controls: dict[str, Any], preset: str) -> dict[str, Any]:
    updated = dict(controls)
    config = GENERATION_PRESETS.get(preset)
    if not config:
        updated["generation_preset"] = "Custom"
        return updated

    for key, value in config.items():
        if key == "description":
            continue
        updated[key] = value
    updated["generation_preset"] = preset
    return updated


def sync_generation_preset_label(controls: dict[str, Any]) -> dict[str, Any]:
    updated = dict(controls)
    preset_name = updated.get("generation_preset", "Internal prototyping")
    if preset_name in GENERATION_PRESETS:
        preset = GENERATION_PRESETS[preset_name]
        matches = all(
            updated.get(key) == value
            for key, value in preset.items()
            if key != "description"
        )
        if not matches:
            updated["generation_preset"] = "Custom"
    return updated


def build_quick_controls_frame(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        rows.append(
            {
                "Field": item["column"],
                "Sensitivity": metadata_sensitivity(item),
                "Field type": item["data_type"].replace("_", " ").title(),
                "Field action": item.get("control_action", "Preserve"),
                "Included": bool(item["include"]),
                "What will happen": metadata_handling(item),
            }
        )
    quick_frame = pd.DataFrame(rows)
    sensitivity_order = {"Restricted": 0, "Sensitive": 1, "Operational": 2}
    quick_frame["_order"] = quick_frame["Sensitivity"].map(sensitivity_order).fillna(3)
    quick_frame = quick_frame.sort_values(by=["_order", "Field"]).drop(columns="_order")
    return quick_frame


STEP_PERMISSION_LABELS: dict[int, list[tuple[str, str]]] = {
    0: [
        ("upload", "Upload or replace the source dataset"),
        ("view_raw", "Inspect row-level raw source data"),
    ],
    1: [
        ("remediate", "Apply safe source-data fixes"),
    ],
    2: [
        ("edit_metadata", "Edit metadata rules and field handling"),
        ("generate", "Generate synthetic preview"),
        ("submit_metadata", "Submit the package for review"),
    ],
    3: [
        ("approve_metadata", "Approve the submitted package"),
        ("view_audit", "Review audit and submission history"),
    ],
    4: [
        ("request_export", "Request governed release"),
        ("download_results", "Download the approved synthetic package"),
        ("share_results", "Share release for controlled distribution"),
    ],
}


ROLE_PRIORITY_NOTES: dict[str, str] = {
    "Clinician / Clinical Lead": "Focus on whether the synthetic dataset still reflects clinically plausible patterns without exposing raw patient detail.",
    "Data Analyst": "Focus on data quality, metadata choices, column behavior, and whether the synthetic output remains useful for analysis.",
    "Data Steward": "Focus on request scope, metadata completeness, field ownership, and whether the package remains useful for hospital analytics.",
    "Privacy Officer": "Focus on sensitive-field handling, privacy risk, and whether governance evidence is sufficient before use.",
    "IT / Security Admin": "Focus on export controls, audit traceability, and whether release stays scoped to approved channels.",
    "Executive Approver": "Focus on whether the package is ready for final governed release and whether the control record is complete.",
}


STEP_EXPLANATION_NOTES: dict[int, str] = {
    0: "This step captures the request, places the dataset in the workspace, and makes clear who can inspect source rows.",
    1: "This step explores the source dataset — shape, distributions, and field-level characteristics — before metadata design.",
    2: "This step configures metadata, generates a synthetic preview with quality metrics, and submits the complete package for reviewer sign-off.",
    3: "This step gives the reviewer a complete view of the submitted package including the synthetic preview and quality metrics before approval.",
    4: "This step delivers the approved synthetic package for controlled distribution with full audit trail.",
}


def current_owner_checkpoint(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> tuple[str, str]:
    if not has_active_dataset():
        return "Next step", "Upload a source dataset to open a new request."
    if not st.session_state.intake_confirmed:
        return "Next step", "Submit the uploaded dataset into the workspace."
    if not st.session_state.hygiene_reviewed:
        return "Next step", "Review the scan findings and confirm any safe fixes."
    if not st.session_state.settings_reviewed:
        return "Next step", "Finalize the data settings before sending the request for review."
    if st.session_state.metadata_status == "Changes Requested":
        note = st.session_state.metadata_review_note or "The reviewer sent this request back for revision."
        return "Next step", f"Revise the data settings and resubmit. {note}"
    if st.session_state.metadata_status == "Rejected":
        note = st.session_state.metadata_review_note or "The reviewer rejected this request."
        return "Next step", f"Update the request and submit a new version. {note}"
    if has_unsubmitted_metadata_changes(metadata):
        return "Next step", "A revised settings draft exists. Submit it again for review."
    if st.session_state.metadata_status == "Draft":
        return "Next step", "Submit the reviewed settings to the Manager / Reviewer."
    if st.session_state.metadata_status == "In Review":
        review_package = current_review_package_record()
        if review_package is not None:
            return "Next step", f"Manager / Reviewer should approve or reject package {review_package['package_id']}."
        return "Next step", "Manager / Reviewer should review the submitted request."
    if st.session_state.synthetic_df is None or has_stale_generation(metadata, controls):
        return "Next step", "Generate the synthetic dataset from the approved request."
    if not st.session_state.results_shared_at:
        return "Next step", "Download and share the synthetic results."
    return "Current state", "The current request has been generated and marked as shared."


def build_role_status_lists(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, list[str] | str]:
    available: list[str] = []
    waiting: list[str] = []
    completed: list[str] = []
    visible_labels = [STEP_CONFIG[index]["title"] for index in visible_steps_for_role()]

    if not has_active_dataset():
        if has_permission("upload"):
            if not st.session_state.get("project_purpose", "").strip():
                available.append("Add a short project purpose for the request.")
            available.append("Upload a source dataset to start the current request.")
        else:
            waiting.append("Waiting on the Data Analyst to upload a source dataset.")
        label, next_action = current_owner_checkpoint(metadata, controls)
        return {
            "label": label,
            "next_action": next_action,
            "visible_steps": visible_labels,
            "available": available or ["No direct action is required from this role at the moment."],
            "waiting": waiting or ["No external blockers are active right now."],
            "completed": ["No request has been started yet in this workspace."],
        }

    if st.session_state.intake_confirmed:
        completed.append("Request submitted.")
    else:
        if has_permission("upload"):
            if not st.session_state.get("project_purpose", "").strip():
                available.append("Add a short project purpose for this request.")
            available.append("Upload or replace the source dataset if needed.")
        available.append("Submit the uploaded dataset into the workflow.")

    if st.session_state.hygiene_reviewed:
        completed.append("System scan reviewed.")
    else:
        if has_permission("remediate"):
            available.append("Review scan findings and apply safe fixes.")
        else:
            waiting.append("Waiting on the Data Analyst to complete the scan review.")

    if st.session_state.settings_reviewed:
        completed.append("Data settings reviewed.")
    elif has_permission("edit_metadata"):
        available.append("Review field actions and finalize the data settings.")
    else:
        waiting.append("Waiting on the Data Analyst to finalize data settings.")

    if st.session_state.metadata_status == "Approved":
        completed.append(
            f"Metadata package approved{f' by {st.session_state.metadata_approved_by}' if st.session_state.metadata_approved_by else ''}."
        )
    elif st.session_state.metadata_status in {"Changes Requested", "Rejected"}:
        note = st.session_state.metadata_review_note or "The reviewer requested a revision."
        if has_permission("edit_metadata") or has_permission("submit_metadata"):
            available.append(f"Review the feedback and revise the request. {note}")
        else:
            waiting.append("Waiting on the Data Analyst to revise and resubmit the request.")
    elif st.session_state.metadata_status == "In Review":
        if has_permission("approve_metadata"):
            record = current_review_package_record() or active_metadata_package_record(metadata)
            available.append(
                f"Approve or reject request {record['package_id']}." if record else "Review the submitted request."
            )
        else:
            record = current_review_package_record()
            waiting.append(
                f"Waiting on the Manager / Reviewer to review package {record['package_id']}."
                if record
                else "Waiting on the Manager / Reviewer to review the submitted request."
            )
    elif st.session_state.settings_reviewed:
        if has_permission("edit_metadata"):
            available.append("Submit the reviewed settings for manager approval.")
        if has_permission("submit_metadata"):
            available.append("Submit the request for review.")

    if has_unsubmitted_metadata_changes(metadata):
        if has_permission("edit_metadata") or has_permission("submit_metadata"):
            available.append("Review the revised settings draft and resubmit the request.")
        waiting.append("The current draft no longer matches the last submitted or approved version.")

    if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
        completed.append("Synthetic dataset generated.")
    elif st.session_state.metadata_status == "Approved" and not has_unsubmitted_metadata_changes(metadata):
        if has_permission("generate"):
            available.append("Generate a new synthetic dataset from the approved package.")
        else:
            waiting.append("Waiting on the Data Analyst to generate synthetic data.")

    if st.session_state.synthetic_df is not None and not has_stale_generation(metadata, controls):
        if st.session_state.results_shared_at:
            completed.append(f"Results marked shared by {st.session_state.results_shared_by}.")
        elif has_permission("share_results"):
            available.append("Download the synthetic dataset and mark results shared.")
        else:
            waiting.append("Waiting on the Data Analyst to share the final results.")

    if not available:
        available = ["No direct action is required from this role at the moment."]
    if not waiting:
        waiting = ["No external blockers are active right now."]

    label, next_action = current_owner_checkpoint(metadata, controls)
    return {
        "label": label,
        "next_action": next_action,
        "visible_steps": visible_labels,
        "available": available,
        "waiting": waiting,
        "completed": completed or ["No workflow checkpoints have been completed yet in this session."],
    }


def build_role_guidance(role: str, step_index: int) -> dict[str, Any]:
    permissions = ROLE_CONFIGS[role]["permissions"]
    allowed = [label for permission, label in STEP_PERMISSION_LABELS.get(step_index, []) if permission in permissions]
    blocked = [label for permission, label in STEP_PERMISSION_LABELS.get(step_index, []) if permission not in permissions]

    if not allowed:
        allowed = ["Review the current state, summaries, and controls without changing workflow state."]

    if not blocked:
        blocked = ["No major restrictions at this step."]

    return {
        "role": role,
        "summary": ROLE_CONFIGS[role]["summary"],
        "priority": ROLE_PRIORITY_NOTES[role],
        "step_note": STEP_EXPLANATION_NOTES[step_index],
        "allowed": allowed,
        "blocked": blocked,
    }


def build_work_in_progress_cards(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()
    package_status = "No package submitted"
    package_detail = "Metadata is still in draft."
    package_owner = "Users"

    if not has_active_dataset():
        package_status = "Awaiting dataset upload"
        package_owner = "Users"
        package_detail = "Upload a CSV before a metadata package can enter the workflow."
    elif review_package is not None:
        package_status = f"{review_package['package_id']} · In Review"
        package_owner = "Governance"
        package_detail = (
            f"Submitted by {review_package['submitted_by']} at {review_package['submitted_at']}. "
            "This is the package currently waiting for approval."
        )
    elif active_package is not None and active_package["status"] in {"Changes Requested", "Rejected"}:
        package_status = f"{active_package['package_id']} · {active_package['status']}"
        package_owner = "Users"
        package_detail = active_package.get("review_note") or "Governance returned the package for revision."
    elif active_package is not None:
        package_status = f"{active_package['package_id']} · {active_package['status']}"
        package_owner = "Users"
        package_detail = (
            f"Approved by {active_package['approved_by']} at {active_package['approved_at']}. "
            "This is the latest approved package in the workflow."
        )

    if has_unsubmitted_metadata_changes(metadata):
        package_detail += " A revised draft exists and still needs to be resubmitted."

    release_owner = "Users"
    release_detail = effective_release_status(metadata, controls)
    if st.session_state.export_requested_by and st.session_state.export_policy_approved_by is None:
        release_owner = "Control"
    elif st.session_state.export_policy_approved_by and st.session_state.export_approved_by is None:
        release_owner = "Executive Approver"
    elif st.session_state.export_approved_by:
        release_owner = "Ready for approved recipients"

    return [
        {
            "title": "Metadata package in progress",
            "value": package_status,
            "detail": package_detail,
            "status": "Good" if active_package and active_package["status"] == "Approved" and not has_unsubmitted_metadata_changes(metadata) else "Warn",
        },
        {
            "title": "Current owner",
            "value": package_owner,
            "detail": "Who is expected to move the metadata package forward next.",
            "status": "Warn",
        },
        {
            "title": "Release gate",
            "value": release_detail,
            "detail": f"Next release owner: {release_owner}.",
            "status": "Good" if release_detail == "Approved" else "Warn",
        },
    ]


def current_workflow_stage(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> int:
    if not has_active_dataset() or not st.session_state.intake_confirmed:
        return 0
    if not st.session_state.hygiene_reviewed:
        return 1
    if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
        return 2 if st.session_state.metadata_status != "In Review" else 3
    return 4


def build_progress_tracker_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    inferred_stage = current_workflow_stage(metadata, controls)
    # Use the user's actual current page as the highlight anchor so the nav
    # stays in sync with what's rendered below.
    active_step = int(st.session_state.get("current_step", inferred_stage))
    all_complete = bool(st.session_state.results_shared_at)
    rows: list[dict[str, str]] = []
    status_labels = step_status_labels(metadata, controls)
    for index, step in enumerate(STEP_CONFIG):
        if index == active_step:
            state = "current"
        elif index < max(inferred_stage, active_step):
            state = "complete"
        elif all_complete:
            state = "complete"
        elif index == max(inferred_stage, active_step) + 1:
            state = "next"
        else:
            state = "future"
        marker = "✓" if state == "complete" else str(index + 1)
        meta = (
            "Complete"
            if state == "complete"
            else "Current step"
            if state == "current"
            else "Next"
            if state == "next"
            else "Locked"
        )
        rows.append(
            {
                "title": step["title"],
                "owner": step["owner"],
                "marker": marker,
                "meta": status_labels[index] if state in {"complete", "current"} else meta,
                "class": state,
            }
        )
    return rows


def build_primary_action(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    action_step = default_step_for_role(metadata, controls)
    step_zero_label = (
        "Upload Dataset"
        if not has_active_dataset()
        else "Submit Request"
        if submission_ready() and not st.session_state.intake_confirmed
        else "Complete Request Details"
    )
    label_map = {
        0: step_zero_label,
        1: "Explore Source Data",
        2: "Configure & Generate",
        3: "Review Submitted Package" if st.session_state.current_role == "Manager / Reviewer" else "Awaiting Reviewer",
        4: "Download Synthetic Package",
    }
    note_map = {
        0: "Add request details, upload a CSV, and submit when the request is ready.",
        1: "Explore the source dataset before designing metadata.",
        2: "Configure metadata, generate a synthetic preview, and submit the package for reviewer sign-off.",
        3: "Reviewer inspects the generated preview and metadata config before approving release.",
        4: "Download the approved synthetic package and record the release for audit.",
    }
    return {
        "step": action_step,
        "label": label_map[action_step],
        "headline": STEP_CONFIG[action_step]["title"],
        "note": note_map[action_step],
    }


def render_action_center(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    status_lists = build_role_status_lists(metadata, controls)
    status_html = "".join(
        f"<div class='status-line'><div class='status-line-label'>{label}</div><div class='status-line-value'>{value}</div></div>"
        for label, value in build_current_request_status_rows(metadata, controls)
    )
    waiting_items = [item for item in status_lists["waiting"] if "No external blockers" not in item]
    blocker_html = f"<div class='action-subtext' style='margin: 0.85rem 0 0 0;'>Blocked by: {waiting_items[0]}</div>" if waiting_items else ""
    st.markdown(
        f"""
        <div class="action-shell">
            <h4>Request status</h4>
            <div class="status-lines">{status_html}</div>
            {blocker_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_role_guidance_panel(step_index: int, compare_only: bool = False) -> None:
    default_index = list(ROLE_CONFIGS.keys()).index(st.session_state.current_role)
    selected_role = st.selectbox(
        "Compare another role",
        options=list(ROLE_CONFIGS.keys()),
        index=default_index,
        key=f"role_guidance_{step_index}_{'compare' if compare_only else 'inline'}",
    )
    guidance = build_role_guidance(selected_role, step_index)
    left_col, right_col = st.columns([1.05, 0.95], gap="large")
    with left_col:
        st.markdown(
            f"""
            <div class="note-card">
                <div class="mini-label">{guidance['role']}</div>
                <strong>What this role focuses on</strong><br/>
                {guidance['priority']} {guidance['step_note']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            f"""
            <div class="note-card">
                <strong>Can do in this step</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in guidance['allowed'])}
                </ul>
                <strong>Cannot do in this step</strong>
                <ul class="bullet-list">
                    {''.join(f"<li>{item}</li>" for item in guidance['blocked'])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_metadata_approval_rows() -> list[dict[str, str]]:
    submitted = st.session_state.metadata_submitted_by
    approved = st.session_state.metadata_approved_by
    submitted_at = st.session_state.metadata_submitted_at
    approved_at = st.session_state.metadata_approved_at
    reviewed_by = st.session_state.metadata_reviewed_by
    reviewed_at = st.session_state.metadata_reviewed_at
    status = st.session_state.metadata_status
    if status == "Changes Requested":
        final_status = "Changes requested"
        final_kind = "Warn"
        final_detail = f"Returned by: {reviewed_by or 'Governance'}{f' at {reviewed_at}' if reviewed_at else ''}"
    elif status == "Rejected":
        final_status = "Rejected"
        final_kind = "Bad"
        final_detail = f"Rejected by: {reviewed_by or 'Governance'}{f' at {reviewed_at}' if reviewed_at else ''}"
    else:
        final_status = "Approved" if approved else "Pending"
        final_kind = "Good" if approved else "Warn"
        final_detail = f"Approved by: {approved or 'Waiting for approval'}{f' at {approved_at}' if approved_at else ''}"
    return [
        {
            "level": "Level 1",
            "stage": "Package prepared",
            "owner": "Users",
            "status": "Complete" if st.session_state.metadata_status in {"Draft", "In Review", "Approved", "Changes Requested", "Rejected"} else "Pending",
            "kind": "Good" if st.session_state.metadata_status in {"Draft", "In Review", "Approved", "Changes Requested", "Rejected"} else "Warn",
            "detail": "Profiled source fields become editable metadata with owners, handling rules, and inclusion flags.",
        },
        {
            "level": "Level 2",
            "stage": "Review submitted",
            "owner": "Users",
            "status": "Submitted" if submitted else "Pending",
            "kind": "Good" if submitted else "Warn",
            "detail": f"Submitted by: {submitted or 'Waiting for submission'}{f' at {submitted_at}' if submitted_at else ''}",
        },
        {
            "level": "Level 3",
            "stage": "Metadata approved",
            "owner": "Governance",
            "status": final_status,
            "kind": final_kind,
            "detail": final_detail,
        },
    ]


def build_release_approval_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    validation_ready = st.session_state.validation is not None and not has_stale_generation(metadata, controls)
    requested_by = st.session_state.export_requested_by
    policy_by = st.session_state.export_policy_approved_by
    final_by = st.session_state.export_approved_by
    return [
        {
            "level": "Level 1",
            "stage": "Validation complete",
            "owner": "Users + Governance",
            "status": "Verified" if validation_ready else "Waiting",
            "kind": "Good" if validation_ready else "Warn",
            "detail": "A current validation run is required before the release chain can start.",
        },
        {
            "level": "Level 2",
            "stage": "Release requested",
            "owner": "Users",
            "status": "Requested" if requested_by else "Pending",
            "kind": "Good" if requested_by else "Warn",
            "detail": f"Requested by: {requested_by or 'Waiting for request'}",
        },
        {
            "level": "Level 3",
            "stage": "Control review",
            "owner": "IT / Security Admin",
            "status": "Approved" if policy_by else "Pending",
            "kind": "Good" if policy_by else "Warn",
            "detail": f"Approved by: {policy_by or 'Waiting for control review'}",
        },
        {
            "level": "Level 4",
            "stage": "Final export authorization",
            "owner": "Executive Approver",
            "status": "Approved" if final_by else "Locked",
            "kind": "Good" if final_by else "Warn",
            "detail": f"Authorized by: {final_by or 'Waiting for Executive Approver'}",
        },
    ]


def render_approval_hierarchy(rows: list[dict[str, str]], key_prefix: str) -> None:
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        chip_class = render_status_chip(row["kind"])
        col.markdown(
            f"""
            <div class="approval-card" id="{key_prefix}_{row['level']}">
                <div class="approval-level">{row['level']}</div>
                <h4>{row['stage']}</h4>
                <div class="approval-owner">{row['owner']}</div>
                <div class="{chip_class}">{row['status']}</div>
                <div class="approval-detail">{row['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _format_percent(series: pd.Series) -> pd.Series:
    return (series * 100).round(1)


def build_distribution_comparison(metadata: list[dict[str, Any]], column_name: str) -> dict[str, Any]:
    metadata_lookup = {item["column"]: item for item in metadata}
    column_meta = metadata_lookup[column_name]
    role = column_meta["data_type"]
    original = st.session_state.source_df[column_name]
    synthetic = st.session_state.synthetic_df[column_name]

    if role == "numeric":
        original_numeric = pd.to_numeric(original, errors="coerce").dropna()
        synthetic_numeric = pd.to_numeric(synthetic, errors="coerce").dropna()
        combined = pd.concat([original_numeric, synthetic_numeric], ignore_index=True)
        if combined.empty:
            return {"kind": "empty", "frame": pd.DataFrame(), "note": "No numeric values available for this column."}

        if float(combined.min()) == float(combined.max()):
            edges = np.array([float(combined.min()) - 0.5, float(combined.max()) + 0.5])
        else:
            edges = np.linspace(float(combined.min()), float(combined.max()), 13)

        source_hist, edges = np.histogram(original_numeric, bins=edges)
        synthetic_hist, _ = np.histogram(synthetic_numeric, bins=edges)
        labels = [f"{edges[index]:.1f} to {edges[index + 1]:.1f}" for index in range(len(edges) - 1)]
        frame = pd.DataFrame(
            {
                "Bucket": labels,
                "Source": _format_percent(pd.Series(source_hist) / max(source_hist.sum(), 1)),
                "Synthetic": _format_percent(pd.Series(synthetic_hist) / max(synthetic_hist.sum(), 1)),
            }
        ).set_index("Bucket")
        note = (
            f"Source median {original_numeric.median():.1f}, synthetic median {synthetic_numeric.median():.1f}. "
            f"Source mean {original_numeric.mean():.1f}, synthetic mean {synthetic_numeric.mean():.1f}."
        )
        return {"kind": "line", "frame": frame, "note": note}

    if role == "date":
        original_dates = pd.to_datetime(original, errors="coerce", format="mixed").dropna()
        synthetic_dates = pd.to_datetime(synthetic, errors="coerce", format="mixed").dropna()
        combined = pd.concat([original_dates, synthetic_dates], ignore_index=True)
        if combined.empty:
            return {"kind": "empty", "frame": pd.DataFrame(), "note": "No valid dates available for this column."}
        freq = "W" if (combined.max() - combined.min()).days > 45 else "D"
        source_counts = original_dates.dt.to_period(freq).value_counts(normalize=True).sort_index()
        synthetic_counts = synthetic_dates.dt.to_period(freq).value_counts(normalize=True).sort_index()
        periods = sorted(set(source_counts.index).union(set(synthetic_counts.index)))
        frame = pd.DataFrame(
            {
                "Period": [str(period) for period in periods],
                "Source": _format_percent(source_counts.reindex(periods, fill_value=0.0)),
                "Synthetic": _format_percent(synthetic_counts.reindex(periods, fill_value=0.0)),
            }
        ).set_index("Period")
        note = f"Timeline compared at {'weekly' if freq == 'W' else 'daily'} resolution."
        return {"kind": "line", "frame": frame, "note": note}

    source_distribution = original.fillna("Missing").astype(str).value_counts(normalize=True)
    synthetic_distribution = synthetic.fillna("Missing").astype(str).value_counts(normalize=True)
    ranked_categories = (
        _format_percent(source_distribution).add(_format_percent(synthetic_distribution), fill_value=0.0).sort_values(ascending=False).head(8).index
    )
    frame = pd.DataFrame(
        {
            "Category": ranked_categories,
            "Source": _format_percent(source_distribution.reindex(ranked_categories, fill_value=0.0)),
            "Synthetic": _format_percent(synthetic_distribution.reindex(ranked_categories, fill_value=0.0)),
        }
    ).set_index("Category")
    note = "Top categories shown by combined frequency across source and synthetic output."
    return {"kind": "bar", "frame": frame, "note": note}


def build_use_case_rows(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    indicators = {item["label"]: item["value"] for item in build_validation_dashboard(metadata, controls)}
    overall = st.session_state.validation["overall_score"]
    privacy = st.session_state.validation["privacy_score"]
    utility = indicators.get("Sandbox suitability", 0.0)
    fidelity = st.session_state.validation["fidelity_score"]
    wait_fields_present = any("wait" in item["column"].lower() and item["include"] for item in metadata)

    def status(value: str) -> str:
        return value

    return [
        {
            "Use case": "Operational workflow modeling",
            "Fit": status("Ready" if overall >= 75 else "Review"),
            "Why": "Key operational fields remain aligned enough to model service-line workflows without moving raw encounter rows.",
            "Guardrail": "Use synthetic output only. Do not back-infer individual patient events.",
        },
        {
            "Use case": "Patient-flow scenario analysis",
            "Fit": status("Ready" if wait_fields_present and fidelity >= 70 else "Limited"),
            "Why": "Arrival, acuity, wait, and disposition behavior can support queue and throughput experiments under controlled assumptions.",
            "Guardrail": "Treat scenario outputs as planning aids, not as exact forecasts.",
        },
        {
            "Use case": "Analytics pipeline development",
            "Fit": status("Ready" if utility >= 75 else "Review"),
            "Why": "Analysts can test joins, cohort logic, feature engineering, and notebooks before requesting sensitive data access.",
            "Guardrail": "Re-run verification after metadata changes before sharing derived work.",
        },
        {
            "Use case": "Vendor sandbox or integration testing",
            "Fit": status("Ready" if privacy >= 85 else "Review"),
            "Why": "Synthetic records can exercise file layouts, API contracts, and workflows without releasing direct identifiers.",
            "Guardrail": "Share only the generated synthetic output and keep the release readiness report attached.",
        },
        {
            "Use case": "Training and workflow rehearsal",
            "Fit": status("Ready" if overall >= 65 else "Limited"),
            "Why": "Teams can rehearse handoffs, reporting, and review processes using realistic but de-identified examples.",
            "Guardrail": "Not for clinical decision support or patient-specific intervention.",
        },
    ]


def build_validation_dashboard(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, Any]]:
    validation = st.session_state.validation
    synthetic_df = st.session_state.synthetic_df
    if validation is None or synthetic_df is None:
        return []

    active_columns = [item for item in metadata if item["include"]]
    active_names = [item["column"] for item in active_columns]
    schema_match = round(
        len([column for column in active_names if column in synthetic_df.columns]) / max(len(active_names), 1) * 100,
        1,
    )
    statistical_fidelity = round(validation["fidelity_score"] * 0.75 + schema_match * 0.25, 1)
    unresolved_hygiene_pressure = max(
        0.0,
        100.0
        - st.session_state.hygiene["severity_counts"]["High"] * 12
        - st.session_state.hygiene["severity_counts"]["Medium"] * 6,
    )
    downstream_utility = round(
        min(
            100.0,
            validation["overall_score"] * 0.62
            + schema_match * 0.18
            + unresolved_hygiene_pressure * 0.20,
        ),
        1,
    )
    return [
        {
            "label": "Schema preservation",
            "value": schema_match,
            "detail": "Approved source fields retained in the synthetic output.",
        },
        {
            "label": "Distribution alignment",
            "value": validation["fidelity_score"],
            "detail": "Operational similarity between source and synthetic columns.",
        },
        {
            "label": "Privacy boundary",
            "value": validation["privacy_score"],
            "detail": "Overlap and identifier reuse checks under the current posture.",
        },
        {
            "label": "Statistical fidelity",
            "value": statistical_fidelity,
            "detail": "Blends schema coverage with column-level fidelity.",
        },
        {
            "label": "Sandbox suitability",
            "value": downstream_utility,
            "detail": "Indicator for workflow modeling and sandbox use.",
        },
    ]


def build_comparison_table(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metadata:
        if not item["include"] or item["column"] not in synthetic_df.columns:
            continue
        column = item["column"]
        role = item["data_type"]
        if role == "identifier":
            continue
        if role == "numeric":
            original_numeric = pd.to_numeric(original_df[column], errors="coerce").dropna()
            synthetic_numeric = pd.to_numeric(synthetic_df[column], errors="coerce").dropna()
            if original_numeric.empty or synthetic_numeric.empty:
                continue
            rows.append(
                {
                    "Field": column,
                    "Source summary": f"mean {original_numeric.mean():.1f} | median {original_numeric.median():.1f}",
                    "Synthetic summary": f"mean {synthetic_numeric.mean():.1f} | median {synthetic_numeric.median():.1f}",
                }
            )
        else:
            source_mode = original_df[column].fillna("Missing").astype(str).value_counts(normalize=True).head(1)
            synthetic_mode = synthetic_df[column].fillna("Missing").astype(str).value_counts(normalize=True).head(1)
            if source_mode.empty or synthetic_mode.empty:
                continue
            rows.append(
                {
                    "Field": column,
                    "Source summary": f"{source_mode.index[0]} ({source_mode.iloc[0] * 100:.1f}%)",
                    "Synthetic summary": f"{synthetic_mode.index[0]} ({synthetic_mode.iloc[0] * 100:.1f}%)",
                }
            )
        if len(rows) >= 8:
            break
    return pd.DataFrame(rows)


def build_generation_control_rows(controls: dict[str, Any]) -> list[dict[str, str]]:
    locked_columns = controls.get("locked_columns", [])
    return [
        {
            "Control": "Release profile",
            "Setting": str(controls.get("generation_preset", "Internal prototyping")),
            "Effect": "High-level use-case preset that sets privacy, structure, and noise in one click.",
        },
        {
            "Control": "Synthetic row count",
            "Setting": str(controls.get("synthetic_rows", 0)),
            "Effect": "Sets how many synthetic records will be created in the current run.",
        },
        {
            "Control": "Privacy versus fidelity",
            "Setting": f"{controls.get('fidelity_priority', 0)} / 100",
            "Effect": "Balances closeness to source behavior against privacy smoothing.",
        },
        {
            "Control": "Lock key distributions",
            "Setting": ", ".join(locked_columns[:4]) + (" ..." if len(locked_columns) > 4 else "") if locked_columns else "None selected",
            "Effect": "Keeps selected columns closer to source frequency or value patterns.",
        },
        {
            "Control": "Correlation preservation",
            "Setting": f"{controls.get('correlation_preservation', 0)} / 100",
            "Effect": "Retains more row-to-row structure across fields when increased.",
        },
        {
            "Control": "Rare case retention",
            "Setting": f"{controls.get('rare_case_retention', 0)} / 100",
            "Effect": "Gives additional weight to rare categories and atypical rows.",
        },
        {
            "Control": "Noise level",
            "Setting": f"{controls.get('noise_level', 0)} / 100",
            "Effect": "Controls how much random variation is added during generation.",
        },
        {
            "Control": "Missingness pattern",
            "Setting": str(controls.get("missingness_pattern", "Preserve source pattern")),
            "Effect": "Determines whether gaps are preserved, reduced, or filled.",
        },
        {
            "Control": "Outlier strategy",
            "Setting": str(controls.get("outlier_strategy", "Preserve tails")),
            "Effect": "Defines whether extreme values are preserved, clipped, or smoothed.",
        },
    ]


def build_validation_report(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> str:
    validation = st.session_state.validation
    if validation is None:
        return "Verification is not available yet."
    lines = [
        "Synthetic Package Release Readiness Report",
        "",
        f"Dataset: {st.session_state.source_label}",
        f"Metadata status: {st.session_state.metadata_status}",
        f"Release recommendation: {effective_release_status(metadata, controls)}",
        "",
        "Supporting verification indicators (secondary):",
        f"- Fidelity indicator: {validation['fidelity_score']}",
        f"- Privacy indicator: {validation['privacy_score']}",
        f"- Overall indicator: {validation['overall_score']}",
        "",
        "Verification breakdown:",
    ]
    for metric in build_validation_dashboard(metadata, controls):
        lines.append(f"- {metric['label']}: {metric['value']} ({metric['detail']})")
    lines.extend(["", "Active generation controls:"])
    for row in build_generation_control_rows(controls):
        lines.append(f"- {row['Control']}: {row['Setting']} ({row['Effect']})")
    lines.extend(["", "Governance and privacy checks:"])
    for _, row in validation["privacy_checks"].iterrows():
        lines.append(f"- {row['check']}: {row['result']} ({row['interpretation']})")
    lines.extend(
        [
            "",
            "Audit status:",
            f"- Metadata approved by: {st.session_state.metadata_approved_by or 'Not approved'}",
            f"- Submitted by: {st.session_state.metadata_submitted_by or 'Not submitted'}",
            f"- Shared by: {st.session_state.results_shared_by or 'Not shared'}",
            f"- Shared at: {st.session_state.results_shared_at or 'Not shared'}",
        ]
    )
    lines.extend(["", "Recommended use cases:"])
    for row in build_use_case_rows(metadata, controls):
        lines.append(f"- {row['Use case']}: {row['Fit']} ({row['Guardrail']})")
    return "\n".join(lines)


def sync_metadata_workflow_state(metadata: list[dict[str, Any]]) -> None:
    st.session_state.metadata_has_unsubmitted_changes = has_unsubmitted_metadata_changes(metadata)
    reviewed_signature = st.session_state.get("settings_review_signature")
    if reviewed_signature and build_metadata_signature(metadata) != reviewed_signature:
        st.session_state.settings_reviewed = False


def has_stale_generation(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> bool:
    if st.session_state.synthetic_df is None:
        return False
    return st.session_state.last_generation_signature != build_generation_signature(metadata, controls)


def intake_visible_to_raw_rows() -> str:
    return "Full preview" if has_permission("view_raw") else "Summary only"


def build_operating_state_cards(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[dict[str, str]]:
    sensitive_fields = build_phi_detection_frame(st.session_state.profile, metadata) if has_active_dataset() else pd.DataFrame()
    review_fields = sum(1 for item in metadata if metadata_owner(item) in {"Data Steward", "Privacy Officer"} and item["include"])
    return [
        {
            "title": "Signed-in role",
            "value": st.session_state.current_role,
            "detail": current_role_group(),
            "status": "Good",
        },
        {
            "title": "Raw data visibility",
            "value": intake_visible_to_raw_rows(),
            "detail": "Row-level source access",
            "status": "Good" if has_permission("view_raw") else "Warn",
        },
        {
            "title": "PHI / sensitive fields",
            "value": str(len(sensitive_fields)),
            "detail": "Fields under active controls",
            "status": "Warn" if len(sensitive_fields) else "Good",
        },
        {
            "title": "Metadata status",
            "value": metadata_display_status(metadata),
            "detail": f"{review_fields} governance-owned fields",
            "status": "Good" if st.session_state.metadata_status == "Approved" and not has_unsubmitted_metadata_changes(metadata) else "Warn",
        },
        {
            "title": "Release gate",
            "value": effective_release_status(metadata, controls),
            "detail": "Current export state",
            "status": "Good" if effective_release_status(metadata, controls) == "Approved" else "Warn",
        },
    ]


def step_status_labels(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> list[str]:
    if not has_active_dataset():
        return ["Upload needed", "Locked", "Locked", "Locked", "Locked"]

    # Step 2 label: configure + generate state
    if st.session_state.metadata_status == "In Review":
        configure_label = "Submitted"
    elif st.session_state.metadata_status == "Approved":
        configure_label = "Approved"
    elif st.session_state.metadata_status in {"Changes Requested", "Rejected"}:
        configure_label = "Changes requested"
    elif st.session_state.synthetic_df is not None:
        configure_label = "Preview ready"
    else:
        configure_label = "In progress"

    # Step 3 label: manager review state
    if st.session_state.metadata_status == "In Review":
        manager_label = "Awaiting review"
    elif st.session_state.metadata_status == "Approved":
        manager_label = "Approved"
    elif st.session_state.metadata_status in {"Changes Requested", "Rejected"}:
        manager_label = "Returned"
    else:
        manager_label = "Waiting"

    # Step 4 label: release state
    if st.session_state.results_shared_at:
        release_label = "Released"
    elif st.session_state.metadata_status == "Approved":
        release_label = "Ready"
    else:
        release_label = "Locked"

    return [
        "Uploaded" if st.session_state.intake_confirmed else "Action needed",
        "Reviewed" if st.session_state.hygiene_reviewed else "Ready",
        configure_label,
        manager_label,
        release_label,
    ]


def max_unlocked_step(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> int:
    if not st.session_state.intake_confirmed:
        return 0
    if not st.session_state.hygiene_reviewed:
        return 1
    if not st.session_state.settings_reviewed:
        return 2
    if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
        # Analyst on step 2 (configure+generate+submit); manager on step 3 (review+approve)
        return 3 if st.session_state.metadata_status == "In Review" else 2
    return 4  # Release


def default_step_for_role(metadata: list[dict[str, Any]], controls: dict[str, Any], role: str | None = None) -> int:
    active_role = role or st.session_state.get("current_role")
    visible_steps = visible_steps_for_role(active_role)

    if not visible_steps:
        return 0

    if active_role == "Manager / Reviewer":
        if st.session_state.metadata_status == "In Review":
            return 3  # Manager Review & Approve page
        if st.session_state.synthetic_df is not None and st.session_state.metadata_status == "Approved":
            return 4  # Release page
        return visible_steps[0]

    if active_role == "Data Analyst":
        if not st.session_state.intake_confirmed:
            return 0
        if not st.session_state.hygiene_reviewed:
            return 1
        if st.session_state.metadata_status != "Approved" or has_unsubmitted_metadata_changes(metadata):
            return 2  # Configure & Generate
        return 4  # Release (approved)

    unlocked = max_unlocked_step(metadata, controls)
    for step_index in visible_steps:
        if step_index >= unlocked:
            return step_index
    return visible_steps[-1]


def ensure_role_step_visibility(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    visible_steps = visible_steps_for_role()
    if st.session_state.current_step not in visible_steps:
        st.session_state.current_step = default_step_for_role(metadata, controls)


def quick_sign_in(role: str) -> None:
    email_stub = "analyst" if role == "Data Analyst" else "reviewer"
    st.session_state.authenticated = True
    st.session_state.current_role = role
    st.session_state.current_user_email = f"{email_stub}@southlake.ca"
    record_audit_event("Quick access sign in", f"{st.session_state.current_user_email} entered as {role}.", status="Logged")
    ensure_dataset_loaded()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls
    st.session_state.current_step = default_step_for_role(metadata, controls, role)
    st.rerun()


def render_status_chip(kind: str) -> str:
    if kind == "Good":
        return "status-good"
    if kind == "Bad":
        return "status-bad"
    return "status-warn"


def build_stakeholder_group_overview_html() -> str:
    role_cards = {
        "Data Analyst": "Upload data, review scan findings, adjust settings, submit requests, and download final results.",
        "Manager / Reviewer": "Approve or reject submitted requests and review the final synthetic output.",
    }
    cards = []
    for role_name, summary in role_cards.items():
        cards.append(
            f"""
            <div class="group-chip">
                <div class="group-chip-label">{role_name}</div>
                <div class="group-chip-value">{ROLE_TO_GROUP.get(role_name, '')}</div>
                <div class="group-chip-meta">{summary}</div>
            </div>
            """
        )
    return f"<div class='group-strip' style='grid-template-columns: repeat(2, minmax(0, 1fr));'>{''.join(cards)}</div>"


def render_stakeholder_group_overview() -> None:
    st.markdown(build_stakeholder_group_overview_html(), unsafe_allow_html=True)


def render_login_screen() -> None:
    logo_uri = load_logo_data_uri()

    # Page-level CSS
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem !important; }

            /* Left panel — dark brand */
            .login-left-panel {
                background: linear-gradient(140deg, #0B3A6B 0%, #08467D 40%, #0B5EA8 100%);
                border-radius: 20px;
                padding: 2.5rem 2.5rem;
                min-height: 620px;
                color: #FFFFFF;
                position: relative;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .login-left-panel::before {
                content: "";
                position: absolute;
                top: -150px;
                right: -150px;
                width: 400px;
                height: 400px;
                background: radial-gradient(circle, rgba(25,203,197,0.12) 0%, transparent 70%);
                border-radius: 50%;
                z-index: 0;
            }
            .login-left-inner { position: relative; z-index: 1; }
            .login-brand-logo {
                width: 280px;
                max-width: 80%;
                height: auto;
                margin-bottom: 1.8rem;
                background: rgba(255,255,255,0.98);
                padding: 0.7rem 1rem;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            .login-brand-kicker {
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: rgba(255,255,255,0.65);
                margin-bottom: 0.9rem;
            }
            .login-brand-headline {
                font-size: 2.2rem;
                font-weight: 600;
                line-height: 1.15;
                letter-spacing: -0.015em;
                margin: 0 0 1.1rem 0;
                color: #FFFFFF;
            }
            .login-brand-sub {
                font-size: 0.98rem;
                line-height: 1.6;
                color: rgba(255,255,255,0.78);
                margin: 0 0 2.2rem 0;
            }
            .login-feature {
                display: flex;
                gap: 0.85rem;
                align-items: flex-start;
                margin-bottom: 0.9rem;
            }
            .login-feature-icon {
                flex: 0 0 32px;
                height: 32px;
                border-radius: 8px;
                background: #19CBC5;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #FFFFFF;
                font-weight: 900;
                font-size: 1rem;
                box-shadow: 0 2px 6px rgba(25,203,197,0.35);
            }
            .login-feature-title {
                font-size: 0.92rem;
                font-weight: 600;
                color: #FFFFFF;
            }
            .login-feature-desc {
                font-size: 0.82rem;
                color: rgba(255,255,255,0.72);
                margin-top: 0.1rem;
                line-height: 1.5;
            }

            /* Right panel — form */
            .login-right-panel {
                padding: 2.5rem 1rem 2.5rem 1.5rem;
            }
            .login-form-title {
                font-size: 1.75rem;
                font-weight: 600;
                color: #0F172A;
                letter-spacing: -0.015em;
                margin: 0 0 0.35rem 0;
                line-height: 1.2;
            }
            .login-form-sub {
                font-size: 0.92rem;
                color: #64748B;
                line-height: 1.5;
                margin-bottom: 1.8rem;
            }

            /* Input styling — matches demo button shape for visual consistency */
            .login-right-panel [data-baseweb="input"] {
                border-radius: 12px !important;
                border: 1.5px solid #CBD5E1 !important;
                min-height: 52px !important;
                box-sizing: border-box !important;
                background: #FFFFFF !important;
            }
            .login-right-panel [data-baseweb="input"]:hover {
                border-color: #94A3B8 !important;
            }
            .login-right-panel [data-baseweb="input"]:focus-within {
                border-color: #0B5EA8 !important;
                box-shadow: 0 0 0 3px rgba(11,94,168,0.12) !important;
            }
            .login-right-panel [data-baseweb="input"] input,
            .login-right-panel [data-baseweb="base-input"] input {
                padding: 0.85rem 1rem !important;
                font-size: 0.95rem !important;
                line-height: 1.5 !important;
                color: #0F172A !important;
                background: transparent !important;
            }
            .login-right-panel .stTextInput label p,
            .login-right-panel .stTextInput label {
                font-size: 0.87rem !important;
                font-weight: 500 !important;
                color: #334155 !important;
                margin-bottom: 0.3rem !important;
            }
            .login-right-panel .stTextInput {
                margin-bottom: 0.85rem !important;
            }
            .login-divider {
                display: flex;
                align-items: center;
                gap: 0.8rem;
                margin: 1.5rem 0 1rem 0;
            }
            .login-divider-line { flex: 1; height: 1px; background: #E2E8F0; }
            .login-divider-text { font-size: 0.76rem; color: #94A3B8; font-weight: 500; }
            .login-footer {
                margin-top: 1.6rem;
                padding-top: 1.1rem;
                border-top: 1px solid #F1F5F9;
                font-size: 0.8rem;
                color: #94A3B8;
                line-height: 1.55;
                text-align: left;
            }
            .login-footer a {
                color: #0B5EA8;
                text-decoration: none;
                font-weight: 500;
                margin-right: 1.2rem;
            }
            .login-footer code {
                background: #F1F5F9;
                padding: 0.1rem 0.4rem;
                border-radius: 4px;
                font-size: 0.76rem;
                color: #475569;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1], gap="medium")

    # Global placeholder styling for login form inputs — targets stTextInput directly
    # because Streamlit widgets are NOT children of custom markdown wrappers in the DOM.
    st.markdown(
        """
        <style>
            .stTextInput input {
                color: #0F172A !important;
            }
            .stTextInput input::placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #94A3B8 !important;
            }
            .stTextInput input::-webkit-input-placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #94A3B8 !important;
            }
            .stTextInput input::-moz-placeholder {
                color: #94A3B8 !important;
                opacity: 1 !important;
            }
            .stTextInput input:-ms-input-placeholder {
                color: #94A3B8 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── LEFT: brand panel (single self-contained HTML block) ──
    logo_html = (
        f'<img src="{logo_uri}" alt="Southlake Health" class="login-brand-logo" />'
        if logo_uri else ''
    )
    with left_col:
        st.markdown(
            f"""
            <div class="login-left-panel">
                <div class="login-left-inner">
                    {logo_html}
                    <div class="login-brand-kicker">Agentic Synthetic Data Workspace</div>
                    <h1 class="login-brand-headline">Governed synthetic data for hospital teams.</h1>
                    <p class="login-brand-sub">
                        Create synthetic healthcare datasets with metadata transparency,
                        reviewer sign-off, and controlled release &mdash; without moving source data.
                    </p>
                </div>
                <div class="login-left-inner">
                    <div class="login-feature">
                        <div class="login-feature-icon">&#10003;</div>
                        <div>
                            <div class="login-feature-title">Role-based access</div>
                            <div class="login-feature-desc">Data Analyst uploads. Manager signs off. Auditable throughout.</div>
                        </div>
                    </div>
                    <div class="login-feature">
                        <div class="login-feature-icon">&#10003;</div>
                        <div>
                            <div class="login-feature-title">Metadata-only transformation</div>
                            <div class="login-feature-desc">Source records never leave the governed boundary.</div>
                        </div>
                    </div>
                    <div class="login-feature">
                        <div class="login-feature-icon">&#10003;</div>
                        <div>
                            <div class="login-feature-title">Controlled release</div>
                            <div class="login-feature-desc">Verified synthetic packages released for internal modeling only.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── RIGHT: form panel — use pure Streamlit widgets inside a styled wrapper ──
    with right_col:
        # We use a class on the column area via a marker div — all widgets below inherit the CSS
        st.markdown('<div class="login-right-panel">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="login-form-title">Sign in</div>
            <div class="login-form-sub">Use your hospital workspace credentials to continue.</div>
            """,
            unsafe_allow_html=True,
        )

        work_email = st.text_input(
            "Work Email",
            placeholder="name@southlake.ca",
            key="login_email_input",
        )
        password = st.text_input(
            "Password",
            type="password",
            key="login_password_input",
        )

        submitted = st.button(
            "Sign In",
            type="primary",
            use_container_width=True,
            key="login_submit_btn",
        )

        if submitted:
            email_value = work_email.strip().lower()
            email_valid = "@" in email_value and "." in email_value.split("@")[-1] if "@" in email_value else False
            role = "Data Analyst"
            if not email_value:
                st.error("Enter your work email to continue.")
            elif not email_valid:
                st.error("Enter a valid work email address.")
            elif password != ROLE_CONFIGS[role]["password"]:
                st.error("Password did not match your access profile.")
            else:
                st.session_state.authenticated = True
                st.session_state.current_role = role
                st.session_state.current_user_email = email_value
                record_audit_event("User signed in", f"{email_value} signed in as {role}.", status="Logged")
                ensure_dataset_loaded()
                metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
                controls = st.session_state.controls
                st.session_state.current_step = default_step_for_role(metadata, controls, role)
                st.rerun()

        st.markdown(
            """
            <div style="text-align:center;margin:0.9rem 0 0.3rem 0;">
                <a href="mailto:itsupport@southlake.ca?subject=Password%20Reset%20Request"
                    style="color:#0B5EA8;text-decoration:none;font-weight:500;font-size:0.85rem;margin:0 0.8rem;">Forgot password?</a>
                <a href="mailto:accessrequests@southlake.ca?subject=Workspace%20Access"
                    style="color:#0B5EA8;text-decoration:none;font-weight:500;font-size:0.85rem;margin:0 0.8rem;">Request access</a>
            </div>
            <div class="login-divider">
                <div class="login-divider-line"></div>
                <div class="login-divider-text">or continue as demo</div>
                <div class="login-divider-line"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Data Analyst", use_container_width=True, key="login_demo_analyst"):
            quick_sign_in("Data Analyst")
        if st.button("Manager / Reviewer", use_container_width=True, key="login_demo_manager"):
            quick_sign_in("Manager / Reviewer")

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full-width bottom footer — demo credential + security notice only
    st.markdown(
        """
        <div style="margin-top:2rem;padding:1.5rem 1rem 1rem 1rem;border-top:1px solid #E2E8F0;text-align:center;">
            <div style="font-size:0.82rem;color:#64748B;margin-bottom:0.4rem;">
                Demo credential for all access profiles:
                <code style="background:#F1F5F9;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.78rem;color:#475569;margin-left:0.3rem;">test</code>
            </div>
            <div style="font-size:0.76rem;color:#94A3B8;">
                Authorized users only. Activity is logged for compliance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    st.sidebar.markdown("## Session")
    st.sidebar.markdown(f"**Role**  \n{st.session_state.current_role}")
    st.sidebar.caption(current_role_summary())
    st.sidebar.metric("Current step", f"{st.session_state.current_step + 1}/{len(STEP_CONFIG)}")
    st.sidebar.metric("Release gate", effective_release_status(metadata, controls))
    st.sidebar.metric("Audit events", len(st.session_state.audit_events))

    st.sidebar.markdown("## Access in this role")
    for permission in sorted(ROLE_CONFIGS[st.session_state.current_role]["permissions"]):
        st.sidebar.markdown(f"- {permission.replace('_', ' ').title()}")

    if st.sidebar.button("Switch role", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_role = None
        st.session_state.current_user_email = None
        st.session_state.current_step = 0
        st.rerun()


def render_header(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    logo_uri = load_logo_data_uri()
    dataset_status, dataset_meta = dataset_status_summary()
    workflow_step = build_primary_action(metadata, controls)["step"]
    request_status = request_status_from_snapshot(capture_workflow_snapshot())
    st.markdown(
        f"""
            <div class="topbar-shell">
                <div class="brand-lockup">
                    {'<img src="' + logo_uri + '" class="brand-logo" alt="Southlake Health logo" />' if logo_uri else ''}
                    <div class="brand-copy">
                        <div class="brand-title">Southlake Health — Agentic Synthetic Data Workspace</div>
                        <div class="brand-subtitle">A governed, metadata-driven workflow for synthetic healthcare data creation, review, and release readiness</div>
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="summary-grid compact">
            <div class="summary-tile slim">
                <div class="summary-label">Current Role</div>
                <div class="summary-value">{st.session_state.current_role}</div>
                <div class="summary-meta">{ROLE_CONFIGS[st.session_state.current_role]['summary']}</div>
            </div>
            <div class="summary-tile slim">
                <div class="summary-label">Request Status</div>
                <div class="summary-value">{request_status}</div>
                <div class="summary-meta">{st.session_state.active_request_id or "No active request selected"}</div>
            </div>
            <div class="summary-tile slim">
                <div class="summary-label">Dataset Status</div>
                <div class="summary-value">{dataset_status}</div>
                <div class="summary-meta">{dataset_meta}</div>
            </div>
            <div class="summary-tile slim">
                <div class="summary-label">Current Step</div>
                <div class="summary-value">Step {workflow_step + 1}</div>
                <div class="summary-meta">{STEP_CONFIG[workflow_step]["title"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    queue_cols = st.columns([1.05, 0.45, 0.5], gap="large")
    with queue_cols[0]:
        request_ids = [record["request_id"] for record in st.session_state.get("request_registry", [])]
        active_id = st.session_state.get("active_request_id")
        if active_id is None and request_ids:
            active_id = request_ids[0]
        pending_selector = st.session_state.get("pending_active_request_selector")
        if pending_selector in request_ids:
            st.session_state.active_request_selector = pending_selector
            st.session_state.pending_active_request_selector = None
            active_id = pending_selector
        selector_value = st.session_state.get("active_request_selector")
        if selector_value not in request_ids and active_id in request_ids:
            st.session_state.active_request_selector = active_id
        selected_request = st.selectbox(
            "Request queue",
            options=request_ids,
            index=request_ids.index(active_id) if active_id in request_ids else 0,
            format_func=request_display_label,
            key="active_request_selector",
        )
        if selected_request != st.session_state.get("active_request_id"):
            sync_active_request_snapshot()
            restore_request_workspace(selected_request)
            persist_shared_workspace_state()
            st.rerun()
    with queue_cols[1]:
        # Invisible spacer label so button aligns vertically with selectbox
        st.markdown("<div style='font-size:0.875rem;margin-bottom:0.25rem;visibility:hidden;'>&nbsp;</div>", unsafe_allow_html=True)
        if has_permission("upload") and st.button("New request", use_container_width=True):
            sync_active_request_snapshot()
            create_blank_request()
            st.session_state.current_step = 0
            st.rerun()
    with queue_cols[2]:
        st.markdown("<div style='font-size:0.875rem;margin-bottom:0.25rem;visibility:hidden;'>&nbsp;</div>", unsafe_allow_html=True)
        clear_col, switch_col = st.columns(2, gap="small")
        if clear_col.button("Clear queue", use_container_width=True):
            schedule_request_queue_clear()
        if switch_col.button("Switch role", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_role = None
            st.session_state.current_user_email = None
            st.session_state.current_step = 0
            st.rerun()


def render_step_navigation(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    st.markdown("**Workflow**")
    progress_html = "".join(
        (
            f'<div class="workflow-progress-step {row["class"]}">'
            f'<div class="workflow-progress-marker">{row["marker"]}</div>'
            f'<div class="workflow-progress-owner">{row["owner"]}</div>'
            f'<div class="workflow-progress-title">{row["title"]}</div>'
            f'<div class="workflow-progress-state">{row["meta"]}</div>'
            "</div>"
        )
        for row in build_progress_tracker_rows(metadata, controls)
    )
    st.markdown(f"<div class='workflow-progress'>{progress_html}</div>", unsafe_allow_html=True)


def render_section_header(step_index: int, checkpoint: str) -> None:
    step = STEP_CONFIG[step_index]
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">Step {step_index + 1} · Owner {step['owner']}</div>
            <h3>{step['title']}</h3>
            <div class="state-text">{checkpoint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_previous_step_control(step_index: int) -> None:
    visible_steps = visible_steps_for_role()
    if step_index not in visible_steps:
        return
    position = visible_steps.index(step_index)
    if position == 0:
        return
    previous_step = visible_steps[position - 1]
    nav_cols = st.columns([0.34, 0.66], gap="large")
    if nav_cols[0].button(
        f"Back to {STEP_CONFIG[previous_step]['title']}",
        key=f"previous_step_{step_index}",
        use_container_width=True,
    ):
        st.session_state.current_step = previous_step
        st.rerun()


def render_role_restriction(message: str) -> None:
    st.info(message)


def render_step_one(metadata: list[dict[str, Any]]) -> None:
    render_section_header(0, "Register the source dataset into the governed workflow.")

    snapshot = capture_workflow_snapshot()
    status_value = request_status_from_snapshot(snapshot)
    has_data = has_active_dataset()
    active_request = st.session_state.active_request_id or "Not yet created"

    # Status chip color based on state
    if has_data:
        status_chip_color = "#136B48"; status_chip_bg = "#EDF9F3"
    else:
        status_chip_color = "#9C6A17"; status_chip_bg = "#FFF6E3"

    # ─────────────────────────────────────────────────────────────
    # A. STATUS STRIP — matches section-shell styling (white bg, --line border, --text color, --brand kickers)
    # ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.2rem;padding:0.8rem 1.15rem;background:#ffffff;border:1px solid #d6e2ec;border-radius:20px;box-shadow:0 10px 24px rgba(8,70,125,0.08);margin-bottom:1rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Request</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;font-family:ui-monospace,monospace;">{active_request}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Role</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">{st.session_state.current_role}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Step</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">1 of {len(STEP_CONFIG)}</span>
            </div>
            <div style="margin-left:auto;display:inline-flex;align-items:center;gap:0.4rem;padding:0.28rem 0.7rem;background:{status_chip_bg};border-radius:999px;">
                <span style="width:6px;height:6px;border-radius:50%;background:{status_chip_color};"></span>
                <span style="font-size:0.8rem;color:{status_chip_color};font-weight:700;">{status_value}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────
    # B. REQUEST DETAILS
    # ─────────────────────────────────────────────────────────────
    with st.container(border=True, key="request_details_panel"):
        st.markdown(
            """
            <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Request details</div>
            <div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Project purpose</div>
            """,
            unsafe_allow_html=True,
        )
        if has_permission("upload"):
            st.text_input(
                "Project purpose",
                key="project_purpose",
                placeholder="Example: ED operational workflow modeling",
                label_visibility="collapsed",
            )
            st.caption("Describe how the synthetic data will be used. Required for submission.")
        else:
            render_role_restriction("This role can view the request summary but cannot edit request details.")

    # ─────────────────────────────────────────────────────────────
    # C. TWO-COLUMN TASK AREA — both columns in parallel bordered containers
    # ─────────────────────────────────────────────────────────────
    task_cols = st.columns([1.35, 1], gap="large")

    # ─── LEFT: Upload zone ───
    with task_cols[0]:
        with st.container(border=True, key="upload_task_panel"):
            if not has_data:
                st.markdown(
                    """
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Primary action</div>
                    <div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.35rem;line-height:1.3;">
                        Start by uploading your source dataset
                    </div>
                    <p style="font-size:0.9rem;color:#668097;line-height:1.55;margin:0 0 1rem 0;">
                        Drop a CSV file from Southlake\'s operational system, or click to browse.
                        The workspace will extract metadata and flag sensitive fields automatically.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Source dataset</div>
                    <div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Dataset loaded</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div style="background:#EDF9F3;border:1px solid #B8E3CC;border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:0.9rem;display:flex;align-items:center;gap:0.9rem;">
                        <div style="flex:0 0 34px;height:34px;border-radius:9px;background:#136B48;color:#FFFFFF;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem;">&#10003;</div>
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:0.9rem;color:#17324d;font-weight:600;">{current_dataset_label()}</div>
                            <div style="font-size:0.78rem;color:#668097;margin-top:0.15rem;">
                                {format_file_size(st.session_state.get("source_file_size")) or "size unknown"}
                                &middot; {st.session_state.profile['summary']['rows']:,} rows
                                &middot; {st.session_state.profile['summary']['columns']} columns
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if has_permission("upload"):
                uploaded_file = st.file_uploader(
                    "Upload source dataset" if not has_data else "Replace source dataset",
                    type=["csv"],
                    help="CSV only. Headers required in first row.",
                    label_visibility="collapsed",
                )
                if uploaded_file is not None:
                    current_signature = (uploaded_file.name, uploaded_file.size)
                    if st.session_state.get("uploaded_signature") != current_signature:
                        st.session_state.source_file_size = uploaded_file.size
                        set_source_dataframe(load_csv_bytes(uploaded_file.getvalue()), f"Uploaded dataset * {uploaded_file.name}")
                        st.session_state.uploaded_signature = current_signature
                        persist_shared_workspace_state()
                        st.rerun()

                st.markdown(
                    """
                    <div style="display:flex;gap:1rem;margin-top:0.25rem;margin-bottom:0;font-size:0.78rem;color:#668097;flex-wrap:wrap;">
                        <span>&#128196; CSV format only</span>
                        <span>&#9881;&#65039; Headers in first row</span>
                        <span>&#128274; Source never leaves governance boundary</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                render_role_restriction("This role cannot upload or replace the source dataset.")

    # ─── RIGHT: Submission readiness ───
    with task_cols[1]:
        with st.container(border=True, key="submission_readiness_panel"):
            checklist = build_submission_checklist()
            total = len(checklist)
            completed_count = sum(1 for _, done in checklist if done)
            progress_pct = int((completed_count / total) * 100) if total else 0
            is_ready = submission_ready()

            if completed_count == total:
                progress_color = "#136B48"
            elif completed_count > 0:
                progress_color = "#0b5ea8"
            else:
                progress_color = "#94A3B8"

            checklist_items_html = ""
            for label, done in checklist:
                if done:
                    icon_html = '<span style="flex:0 0 18px;width:18px;height:18px;border-radius:50%;background:#136B48;color:#FFFFFF;display:inline-flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;">&#10003;</span>'
                    label_style = "color:#17324d;font-weight:500;"
                else:
                    icon_html = '<span style="flex:0 0 18px;width:18px;height:18px;border-radius:50%;background:#FFFFFF;border:1.5px solid #d6e2ec;display:inline-block;"></span>'
                    label_style = "color:#17324d;font-weight:500;"
                checklist_items_html += (
                    f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.55rem;">'
                    f'{icon_html}'
                    f'<span style="font-size:0.9rem;{label_style}">{label}</span>'
                    f'</div>'
                )

            missing = submission_missing_items()
            if is_ready:
                helper_msg = "All requirements met. You can submit this request."
                helper_color = "#136B48"
            elif missing:
                helper_msg = f"Complete {missing[0].lower()} to continue."
                helper_color = "#668097"
            else:
                helper_msg = "Upload required to continue."
                helper_color = "#9C6A17"

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem;">
                    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;">Submission readiness</div>
                    <div style="font-size:0.82rem;font-weight:600;color:{progress_color};">{completed_count} of {total}</div>
                </div>
                <div style="width:100%;height:6px;background:#f1f5f9;border-radius:999px;margin-bottom:1rem;overflow:hidden;">
                    <div style="width:{progress_pct}%;height:100%;background:{progress_color};border-radius:999px;transition:width 0.3s ease;"></div>
                </div>
                {checklist_items_html}
                <div style="font-size:0.82rem;color:{helper_color};margin-top:0.3rem;margin-bottom:0.8rem;line-height:1.5;">{helper_msg}</div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.intake_confirmed:
                st.success("Request entered the governed flow.")
                if st.button("Continue to Scan Source Data", type="primary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()
            else:
                submit_clicked = st.button(
                    "Submit request" if is_ready else "Submit request (blocked)",
                    type="primary",
                    use_container_width=True,
                    disabled=not is_ready,
                    help=None if is_ready else f"Blocked until: {', '.join(submission_missing_items())}",
                )
                if submit_clicked:
                    st.session_state.intake_confirmed = True
                    record_audit_event("Request submitted", "Source workspace request was acknowledged and entered into the governed flow.", status="Completed")
                    st.session_state.current_step = 1
                    st.rerun()

    # ─────────────────────────────────────────────────────────────
    # D. INTAKE SUMMARY (only after upload) — wrapped in matching container
    # ─────────────────────────────────────────────────────────────
    if has_data:
        sensitive_count = len(build_phi_detection_frame(st.session_state.profile, metadata))
        if sensitive_count == 0:
            sens_accent = "#136B48"; sens_bg = "#EDF9F3"
        else:
            sens_accent = "#9C6A17"; sens_bg = "#FFF6E3"

        def _stat_capsule(kicker: str, value: str, detail: str, accent: str = "#0b5ea8", bg: str = "#f5f9fc") -> str:
            return (
                f'<div style="flex:1;min-width:140px;padding:0.85rem 1rem;background:{bg};'
                f'border:1px solid #d6e2ec;border-radius:12px;margin-bottom:0.25rem;">'
                f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{accent};margin-bottom:0.25rem;">{kicker}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:#17324d;line-height:1.15;">{value}</div>'
                f'<div style="font-size:0.76rem;color:#668097;margin-top:0.2rem;line-height:1.35;">{detail}</div>'
                f'</div>'
            )

        with st.container(border=True, key="intake_summary_panel"):
            rows_val = f"{st.session_state.profile['summary']['rows']:,}"
            cols_val = str(st.session_state.profile['summary']['columns'])
            intake_html = (
                '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Dataset intake summary</div>'
                '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">Source package overview</div>'
                '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
                + _stat_capsule("Rows", rows_val, "Loaded from CSV")
                + _stat_capsule("Columns", cols_val, "Fields detected")
                + _stat_capsule(
                    "Sensitive fields",
                    str(sensitive_count),
                    "Flagged for review" if sensitive_count > 0 else "No PHI detected",
                    accent=sens_accent,
                    bg=sens_bg,
                )
                + _stat_capsule(
                    "File size",
                    format_file_size(st.session_state.get("source_file_size")) or "—",
                    "Source dataset",
                )
                + '</div>'
            )
            st.markdown(intake_html, unsafe_allow_html=True)


def render_step_two() -> None:
    # Simplified header — Step 2 is now "Explore Source Data" (visualization-first)
    step_cfg = STEP_CONFIG[1]
    st.markdown(
        f'''
        <div class="section-shell" style="margin-bottom:0.9rem;">
            <h3 style="margin:0;">{step_cfg["title"]}</h3>
            <div style="font-size:0.92rem;color:#668097;margin-top:0.3rem;line-height:1.55;">
                Inspect the cleaned source dataset before designing metadata. Review shape, distributions, and field-level characteristics.
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if not has_active_dataset():
        st.info("Upload and submit a source dataset before exploring.")
        return

    # ─────────────────────────────────────────────────────────────
    # A. STATUS STRIP
    # ─────────────────────────────────────────────────────────────
    active_request = st.session_state.active_request_id or "Not yet created"
    is_reviewed = st.session_state.get("hygiene_reviewed", False)

    if is_reviewed:
        status_chip_color = "#136B48"; status_chip_bg = "#EDF9F3"; status_label = "Data reviewed"
    else:
        status_chip_color = "#9C6A17"; status_chip_bg = "#FFF6E3"; status_label = "Ready for review"

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.2rem;padding:0.8rem 1.15rem;background:#ffffff;border:1px solid #d6e2ec;border-radius:20px;box-shadow:0 10px 24px rgba(8,70,125,0.08);margin-bottom:1rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Request</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;font-family:ui-monospace,monospace;">{active_request}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Role</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">{st.session_state.current_role}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Step</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">2 of {len(STEP_CONFIG)}</span>
            </div>
            <div style="margin-left:auto;display:inline-flex;align-items:center;gap:0.4rem;padding:0.28rem 0.7rem;background:{status_chip_bg};border-radius:999px;">
                <span style="width:6px;height:6px;border-radius:50%;background:{status_chip_color};"></span>
                <span style="font-size:0.8rem;color:{status_chip_color};font-weight:700;">{status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = st.session_state.profile
    source_df = st.session_state.source_df
    columns_profile = profile["columns"]

    # Categorize columns by semantic role
    role_columns: dict[str, list[str]] = {"numeric": [], "date": [], "categorical": [], "binary": [], "identifier": []}
    for col, details in columns_profile.items():
        role_columns.setdefault(details["semantic_role"], []).append(col)

    # ─────────────────────────────────────────────────────────────
    # B. DATASET SHAPE — capsule cards
    # ─────────────────────────────────────────────────────────────
    def _stat_capsule(kicker: str, value: str, detail: str, accent: str = "#0b5ea8", bg: str = "#f5f9fc") -> str:
        return (
            f'<div style="flex:1;min-width:140px;padding:0.85rem 1rem;background:{bg};'
            f'border:1px solid #d6e2ec;border-radius:12px;margin-bottom:0.25rem;">'
            f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{accent};margin-bottom:0.25rem;">{kicker}</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#17324d;line-height:1.15;">{value}</div>'
            f'<div style="font-size:0.76rem;color:#668097;margin-top:0.2rem;line-height:1.35;">{detail}</div>'
            f'</div>'
        )

    rows = profile["summary"]["rows"]
    total_cols = profile["summary"]["columns"]
    total_missing = sum(d["missing_count"] for d in columns_profile.values())
    total_cells = rows * total_cols
    missing_pct = (total_missing / total_cells * 100) if total_cells else 0

    role_counts = {k: len(v) for k, v in role_columns.items() if v}
    role_mix = ", ".join(f"{k}: {c}" for k, c in sorted(role_counts.items(), key=lambda x: -x[1]))

    with st.container(border=True, key="dataset_shape_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Dataset shape</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">Overview of the source data</div>'
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
            + _stat_capsule("Rows", f"{rows:,}", "Total records in dataset")
            + _stat_capsule("Columns", f"{total_cols}", "Fields per record")
            + _stat_capsule("Completeness", f"{100 - missing_pct:.1f}%", f"{total_missing:,} missing cell(s)")
            + _stat_capsule("Field types", str(sum(role_counts.values())), role_mix)
            + '</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────
    # C. DATA PREVIEW (first 10 rows)
    # ─────────────────────────────────────────────────────────────
    with st.container(border=True, key="data_preview_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Data preview</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.65rem;line-height:1.3;">First 10 rows of the source dataset</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(source_df.head(10), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────
    # D. MISSING DATA HEATMAP (one bar chart, only if any missing)
    # ─────────────────────────────────────────────────────────────
    missing_rows = [
        {"Field": col, "Missing %": details["missing_pct"]}
        for col, details in columns_profile.items()
        if details["missing_pct"] > 0
    ]
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows).sort_values("Missing %", ascending=False)
        with st.container(border=True, key="missing_heatmap_panel"):
            st.markdown(
                '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Missing data</div>'
                f'<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.65rem;line-height:1.3;">{len(missing_rows)} field(s) with missing values</div>',
                unsafe_allow_html=True,
            )
            # Pure HTML horizontal bar chart — no chart lib dependency
            max_missing = max(r["Missing %"] for r in missing_rows) if missing_rows else 100
            chart_html = '<div style="display:flex;flex-direction:column;gap:0.45rem;margin-top:0.2rem;">'
            for row in missing_df.to_dict("records"):
                label = str(row["Field"])
                value = float(row["Missing %"])
                pct_width = (value / max_missing * 100) if max_missing > 0 else 0
                # Color intensity by severity
                if value >= 20:
                    bar_color = "#9d2b3c"
                elif value >= 8:
                    bar_color = "#D68A00"
                else:
                    bar_color = "#0b5ea8"
                chart_html += (
                    f'<div style="display:flex;align-items:center;gap:0.7rem;">'
                    f'<div style="min-width:160px;max-width:220px;font-size:0.85rem;color:#17324d;font-weight:500;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{label}">{label}</div>'
                    f'<div style="flex:1;height:22px;background:#F1F5F9;border-radius:6px;overflow:hidden;position:relative;">'
                    f'<div style="width:{pct_width}%;height:100%;background:{bar_color};border-radius:6px;transition:width 0.3s;"></div>'
                    f'</div>'
                    f'<div style="min-width:60px;font-size:0.85rem;color:#475569;font-weight:600;text-align:right;">{value:.1f}%</div>'
                    f'</div>'
                )
            chart_html += '</div>'
            st.markdown(chart_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # E. FIELD DISTRIBUTIONS — tabbed view (Numeric / Categorical / Dates)
    # ─────────────────────────────────────────────────────────────
    cat_cols = role_columns["categorical"] + role_columns["binary"]
    date_cols = role_columns["date"]
    num_cols = role_columns["numeric"]

    if num_cols or cat_cols or date_cols:
        with st.container(border=True, key="field_distributions_panel"):
            st.markdown(
                '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Field distributions</div>'
                '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Drill into specific fields to inspect their shape</div>',
                unsafe_allow_html=True,
            )

            # Build tabs dynamically based on what column types exist
            tab_labels = []
            if num_cols:
                tab_labels.append(f"Numeric · {len(num_cols)}")
            if cat_cols:
                tab_labels.append(f"Categorical · {len(cat_cols)}")
            if date_cols:
                tab_labels.append(f"Dates · {len(date_cols)}")
            tabs = st.tabs(tab_labels)
            tab_idx = 0

            # ── Numeric tab ──
            if num_cols:
                with tabs[tab_idx]:
                    selected_numeric = st.selectbox(
                        "Select a numeric field",
                        options=num_cols,
                        key="num_field_selector",
                        label_visibility="collapsed",
                    )
                    details = columns_profile[selected_numeric]
                    numeric_values = pd.to_numeric(source_df[selected_numeric], errors="coerce").dropna()

                    stats_html = '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.6rem 0 0.8rem 0;">'
                    stats_html += _stat_capsule("Min", f"{details.get('min', 0):g}", "Lowest value")
                    stats_html += _stat_capsule("Median", f"{details.get('median', 0):g}", f"Q1: {details.get('q1', 0):g} · Q3: {details.get('q3', 0):g}")
                    stats_html += _stat_capsule("Mean", f"{details.get('mean', 0):g}", f"Std: {details.get('std', 0):g}")
                    stats_html += _stat_capsule("Max", f"{details.get('max', 0):g}", "Highest value")
                    stats_html += '</div>'
                    st.markdown(stats_html, unsafe_allow_html=True)

                    if not numeric_values.empty:
                        bin_count = min(20, max(8, int(len(numeric_values) ** 0.5)))
                        try:
                            bins = pd.cut(numeric_values, bins=bin_count)
                            hist = bins.value_counts().sort_index()
                            max_count = int(hist.max()) if len(hist) > 0 else 1

                            # Pure HTML vertical bars with horizontal bottom labels
                            bars_html = '<div style="margin-top:0.2rem;">'
                            bars_html += '<div style="display:flex;align-items:flex-end;gap:3px;height:220px;padding:0 0.3rem;border-bottom:1px solid #E5EDF5;">'
                            for interval, count in hist.items():
                                bar_height_pct = (count / max_count * 100) if max_count > 0 else 0
                                midpoint = round((interval.left + interval.right) / 2, 2)
                                bars_html += (
                                    f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;" '
                                    f'title="{interval.left:.2f} to {interval.right:.2f}: {int(count)}">'
                                    f'<div style="font-size:0.68rem;color:#668097;margin-bottom:2px;">{int(count) if count > 0 else ""}</div>'
                                    f'<div style="width:100%;height:{bar_height_pct}%;background:#0b5ea8;border-radius:3px 3px 0 0;min-height:2px;"></div>'
                                    f'</div>'
                                )
                            bars_html += '</div>'
                            # Bottom axis labels — show every few bins to avoid crowding
                            step = max(1, len(hist) // 8)
                            bars_html += '<div style="display:flex;gap:3px;padding:0.4rem 0.3rem 0 0.3rem;">'
                            for i, interval in enumerate(hist.index):
                                midpoint = round((interval.left + interval.right) / 2, 2)
                                label = f"{midpoint:g}" if (i % step == 0 or i == len(hist) - 1) else ""
                                bars_html += f'<div style="flex:1;font-size:0.72rem;color:#475569;text-align:center;">{label}</div>'
                            bars_html += '</div>'
                            bars_html += f'<div style="text-align:center;font-size:0.78rem;color:#668097;margin-top:0.4rem;">{selected_numeric}</div>'
                            bars_html += '</div>'
                            st.markdown(bars_html, unsafe_allow_html=True)
                        except Exception:
                            st.info("Unable to compute histogram for this field.")
                tab_idx += 1

            # ── Categorical tab ──
            if cat_cols:
                with tabs[tab_idx]:
                    selected_cat = st.selectbox(
                        "Select a categorical field",
                        options=cat_cols,
                        key="cat_field_selector",
                        label_visibility="collapsed",
                    )
                    details = columns_profile[selected_cat]
                    top_values = details.get("top_values", {})

                    if top_values:
                        stats_html = '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.6rem 0 0.8rem 0;">'
                        stats_html += _stat_capsule("Unique values", str(details["unique_count"]), f"Completeness: {details['completeness_score']:.1f}%")
                        stats_html += _stat_capsule("Top value", list(top_values.keys())[0] if top_values else "—", f"{list(top_values.values())[0]:.1f}%" if top_values else "—")
                        stats_html += _stat_capsule("Examples", ", ".join(details.get("examples", [])[:2]) or "—", "Sampled from non-null values")
                        stats_html += '</div>'
                        st.markdown(stats_html, unsafe_allow_html=True)

                        cat_df = pd.DataFrame(
                            [{"Value": str(k), "Share (%)": v} for k, v in top_values.items()]
                        ).sort_values("Share (%)", ascending=False)
                        max_share = cat_df["Share (%)"].max() if not cat_df.empty else 100
                        cat_html = '<div style="display:flex;flex-direction:column;gap:0.45rem;margin-top:0.2rem;">'
                        for row in cat_df.to_dict("records"):
                            label = str(row["Value"])
                            value = float(row["Share (%)"])
                            pct_width = (value / max_share * 100) if max_share > 0 else 0
                            cat_html += (
                                f'<div style="display:flex;align-items:center;gap:0.7rem;">'
                                f'<div style="min-width:160px;max-width:220px;font-size:0.85rem;color:#17324d;font-weight:500;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{label}">{label}</div>'
                                f'<div style="flex:1;height:22px;background:#F1F5F9;border-radius:6px;overflow:hidden;">'
                                f'<div style="width:{pct_width}%;height:100%;background:#0b5ea8;border-radius:6px;transition:width 0.3s;"></div>'
                                f'</div>'
                                f'<div style="min-width:60px;font-size:0.85rem;color:#475569;font-weight:600;text-align:right;">{value:.1f}%</div>'
                                f'</div>'
                            )
                        cat_html += '</div>'
                        st.markdown(cat_html, unsafe_allow_html=True)
                    else:
                        st.info("No value distribution available for this field.")
                tab_idx += 1

            # ── Dates tab ──
            if date_cols:
                with tabs[tab_idx]:
                    date_rows = []
                    for col in date_cols:
                        d = columns_profile[col]
                        date_rows.append({
                            "Field": col,
                            "Earliest": d.get("min", "—"),
                            "Latest": d.get("max", "—"),
                            "Completeness %": d["completeness_score"],
                            "Unique dates": d["unique_count"],
                        })
                    st.dataframe(pd.DataFrame(date_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────
    # F. ALL FIELDS REFERENCE (collapsible, lightweight)
    # ─────────────────────────────────────────────────────────────
    profile_rows = []
    for column, details in columns_profile.items():
        profile_rows.append({
            "Field": column,
            "Type": details["semantic_role"],
            "Missing %": details["missing_pct"],
            "Completeness %": details["completeness_score"],
            "Unique": details["unique_count"],
            "Examples": ", ".join(details.get("examples", [])[:3]),
        })
    with st.expander(f"All fields reference table · {total_cols} fields", expanded=False):
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────
    # I. BOTTOM NAV
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
    nav_cols = st.columns([1, 1], gap="small")
    if nav_cols[0].button(
        "Mark data review complete",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.hygiene_reviewed,
    ):
        st.session_state.hygiene_reviewed = True
        record_audit_event("Data exploration completed", "Source data was reviewed before metadata design.", status="Completed")
        st.session_state.current_step = 2
        st.rerun()
    if nav_cols[1].button(
        f"← Back to {STEP_CONFIG[0]['title']}",
        use_container_width=True,
        key="step2_back_to_step1",
    ):
        st.session_state.current_step = 0
        st.rerun()


def _render_status_strip_v2(step_index: int, status_label: str, status_color: str, status_bg: str) -> None:
    """Shared status strip component."""
    active_request = st.session_state.active_request_id or "Not yet created"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.2rem;padding:0.8rem 1.15rem;background:#ffffff;border:1px solid #d6e2ec;border-radius:20px;box-shadow:0 10px 24px rgba(8,70,125,0.08);margin-bottom:1rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Request</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;font-family:ui-monospace,monospace;">{active_request}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Role</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">{st.session_state.current_role}</span>
            </div>
            <div style="width:1px;height:18px;background:#d6e2ec;"></div>
            <div style="display:flex;align-items:baseline;gap:0.45rem;">
                <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;font-weight:700;">Step</span>
                <span style="font-size:0.9rem;color:#17324d;font-weight:600;">{step_index + 1} of {len(STEP_CONFIG)}</span>
            </div>
            <div style="margin-left:auto;display:inline-flex;align-items:center;gap:0.4rem;padding:0.28rem 0.7rem;background:{status_bg};border-radius:999px;">
                <span style="width:6px;height:6px;border-radius:50%;background:{status_color};"></span>
                <span style="font-size:0.8rem;color:{status_color};font-weight:700;">{status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stat_capsule_v2(kicker: str, value: str, detail: str, accent: str = "#0b5ea8", bg: str = "#f5f9fc") -> str:
    return (
        f'<div style="flex:1;min-width:140px;padding:0.85rem 1rem;background:{bg};'
        f'border:1px solid #d6e2ec;border-radius:12px;margin-bottom:0.25rem;">'
        f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{accent};margin-bottom:0.25rem;">{kicker}</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#17324d;line-height:1.15;">{value}</div>'
        f'<div style="font-size:0.76rem;color:#668097;margin-top:0.2rem;line-height:1.35;">{detail}</div>'
        f'</div>'
    )


def build_expected_outcome(controls: dict[str, Any], metadata: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Produce a 5-row forecast of the generation's likely profile, from the
    current controls + metadata. Each row is (label, value, tone).
    Tone is one of 'good' / 'warn' / 'strong' / 'neutral'.
    """
    # Privacy posture — from privacy_preset / epsilon
    preset = str(controls.get("privacy_preset", "Balanced"))
    eps = float(controls.get("privacy_epsilon", 2.0) or 2.0)
    if preset == "Maximum fidelity" or eps >= 8:
        privacy_posture, privacy_tone = "Fidelity-first (low noise)", "warn"
    elif preset == "Balanced" or 1.5 <= eps < 8:
        privacy_posture, privacy_tone = "Balanced", "neutral"
    elif preset == "Strong privacy" or 0.3 <= eps < 1.5:
        privacy_posture, privacy_tone = "Strong (heavy Laplace noise)", "good"
    else:
        privacy_posture, privacy_tone = "Maximum (privacy dominates)", "strong"

    # Structural realism — from use_copula + copula_strength + correlation_preservation
    use_copula = bool(controls.get("use_copula", False))
    copula_str = int(controls.get("copula_strength", 80) or 0)
    corr_pres = int(controls.get("correlation_preservation", 65) or 0)
    if use_copula and copula_str >= 75 and corr_pres >= 60:
        realism, realism_tone = "High", "good"
    elif (use_copula and copula_str >= 50) or corr_pres >= 55:
        realism, realism_tone = "Medium–High", "good"
    elif use_copula or corr_pres >= 30:
        realism, realism_tone = "Medium", "neutral"
    else:
        realism, realism_tone = "Low (fields independent)", "warn"

    # Rare-case retention
    rare = int(controls.get("rare_case_retention", 30) or 0)
    if rare >= 65:
        rare_label, rare_tone = "Full (rare rows over-sampled)", "good"
    elif rare >= 35:
        rare_label, rare_tone = "Strong", "good"
    elif rare >= 15:
        rare_label, rare_tone = "Moderate", "neutral"
    else:
        rare_label, rare_tone = "Low (tail events suppressed)", "warn"

    # Date sensitivity protection
    included = [m for m in metadata if m.get("include")]
    date_meta = [m for m in included if m.get("data_type") == "date"]
    date_total = len(date_meta)
    month_only = sum(1 for m in date_meta if m.get("control_action") == "Month only")
    noise_lvl = int(controls.get("noise_level", 45) or 0)
    if date_total == 0:
        date_label, date_tone = "n/a (no date fields)", "neutral"
    elif month_only and month_only == date_total:
        date_label, date_tone = "High (all dates Month-only)", "good"
    elif month_only:
        date_label, date_tone = f"Mixed ({month_only}/{date_total} Month-only)", "neutral"
    elif noise_lvl >= 60:
        date_label, date_tone = "Strong jitter", "good"
    elif noise_lvl >= 35:
        date_label, date_tone = "Moderate jitter", "neutral"
    else:
        date_label, date_tone = "Light jitter", "warn"

    # Review recommendation — composite
    privacy_strong = preset in {"Strong privacy", "Maximum privacy"} or eps < 1.5
    fidelity_high = preset == "Maximum fidelity" or eps >= 8
    if privacy_strong and realism in {"Medium", "Medium–High", "High"}:
        reco = "Suitable for external / compliance review with audit sign-off"
        reco_tone = "strong"
    elif privacy_strong:
        reco = "Suitable for privacy audit and internal review"
        reco_tone = "good"
    elif fidelity_high and realism in {"Medium–High", "High"}:
        reco = "Suitable for analytics sandbox inside governance boundary"
        reco_tone = "good"
    else:
        reco = "Suitable for internal sandbox use"
        reco_tone = "neutral"

    return [
        ("Privacy posture", privacy_posture, privacy_tone),
        ("Structural realism", realism, realism_tone),
        ("Rare-case retention", rare_label, rare_tone),
        ("Date sensitivity protection", date_label, date_tone),
        ("Review recommendation", reco, reco_tone),
    ]


def _render_expected_outcome_card(controls: dict[str, Any], metadata: list[dict[str, Any]]) -> None:
    rows = build_expected_outcome(controls, metadata)
    tone_styles = {
        "good":    ("#136B48", "#EDF9F3"),
        "strong":  ("#08467d", "#EBF1F7"),
        "warn":    ("#9C6A17", "#FFF6E3"),
        "neutral": ("#475569", "#F1F5F9"),
    }
    items_html = ""
    for label, value, tone in rows:
        color, bg = tone_styles.get(tone, tone_styles["neutral"])
        items_html += (
            f'<div style="display:flex;align-items:center;gap:0.7rem;padding:0.55rem 0.75rem;'
            f'background:#ffffff;border:1px solid #d6e2ec;border-radius:10px;margin-bottom:0.4rem;">'
            f'<div style="flex:0 0 180px;font-size:0.8rem;color:#475569;font-weight:600;">{label}</div>'
            f'<div style="flex:1;display:inline-flex;align-items:center;gap:0.4rem;padding:0.18rem 0.55rem;'
            f'background:{bg};border-radius:999px;width:fit-content;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{color};"></span>'
            f'<span style="font-size:0.82rem;color:{color};font-weight:700;">{value}</span>'
            f'</div></div>'
        )
    st.markdown(
        '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Expected outcome</div>'
        '<div style="font-size:1.1rem;font-weight:600;color:#17324d;margin-bottom:0.1rem;line-height:1.3;">Agent forecast from the current configuration</div>'
        '<div style="font-size:0.82rem;color:#668097;margin-bottom:0.75rem;line-height:1.5;">Updates live as you change metadata actions or algorithm controls. Click Generate to confirm with real metrics.</div>'
        + items_html,
        unsafe_allow_html=True,
    )


# ── Field-level metadata helpers (Step 2 table + detail panel) ────────────

SENSITIVITY_BADGES = {
    "Restricted":  ("🔴", "#9d2b3c", "#fff1f3"),
    "Sensitive":   ("🟡", "#9C6A17", "#FFF6E3"),
    "Operational": ("⚪", "#475569", "#F1F5F9"),
}

SENSITIVITY_ORDER = {"Restricted": 0, "Sensitive": 1, "Operational": 2}


def sensitivity_display_label(name: str) -> str:
    """Return an emoji-prefixed sensitivity label for the main table."""
    icon, _, _ = SENSITIVITY_BADGES.get(name, ("⚪", "#475569", "#F1F5F9"))
    return f"{icon} {name}"


def resolve_generation_method(item: dict[str, Any], controls: dict[str, Any]) -> str:
    """Map the (role, strategy, use_copula) tuple to a human-readable method name.

    Used to display what 'auto' actually resolves to — the table should never
    show 'auto' to the reviewer, only the effective generation method.
    """
    role = str(item.get("data_type", "")).lower()
    strat_raw = str(item.get("strategy", "auto")).lower()
    legacy = {"", "auto", "sample_plus_noise", "sample_category", "sample_plus_jitter", "new_token"}
    is_auto = strat_raw in legacy
    use_copula = bool(controls.get("use_copula", True))

    # Identifier role or explicit identifier strategy always wins
    if role == "identifier" or strat_raw == "identifier":
        return "Surrogate token"

    # Explicit overrides first
    if not is_auto:
        if strat_raw == "copula":
            return "Gaussian Copula" if (role == "numeric" and use_copula) else "Empirical + Gaussian noise"
        if strat_raw == "kde":
            return "Kernel density (smooth)"
        if strat_raw == "dp_laplace":
            return "Differential privacy (Laplace)"
        if strat_raw == "empirical":
            return "Empirical + Gaussian noise"

    # Auto resolution by role
    if role == "numeric":
        return "Gaussian Copula" if use_copula else "Empirical + Gaussian noise"
    if role == "date":
        return "Resample + jitter"
    if role in {"categorical", "binary"}:
        return "Distribution sampling"
    return "Distribution sampling"


def build_field_rationale(item: dict[str, Any], controls: dict[str, Any]) -> str:
    """Prose explanation: why this field is handled this way."""
    role = str(item.get("data_type", "")).lower()
    action = str(item.get("control_action", "Preserve"))
    method = resolve_generation_method(item, controls)
    epsilon = controls.get("privacy_epsilon", 2.0)

    if role == "identifier":
        return (
            f"Source identifiers are never exported. A surrogate token with a "
            f"column-derived prefix is generated for each synthetic row. There is "
            f"zero risk of source-ID leakage regardless of other controls."
        )
    if role == "date":
        if action == "Month only":
            return (
                f"Dates are coarsened to year-month granularity. Exact encounter "
                f"timing is removed; only seasonal / monthly patterns remain."
            )
        if action == "Date shift":
            return (
                f"Dates are resampled from the source and jittered by a few days. "
                f"Exact encounter timestamps are obscured while weekly / seasonal "
                f"patterns are retained."
            )
        return (
            f"Dates are resampled with bounded jitter. Current action = {action}."
        )
    if role == "numeric":
        if method == "Gaussian Copula":
            return (
                f"This field participates in the Gaussian Copula fit across all "
                f"included numeric fields. Marginal distribution is preserved via "
                f"inverse empirical CDF; joint relationships with other numerics "
                f"(e.g. ctas_level vs length_of_stay_hr) are retained through the "
                f"fitted correlation matrix."
            )
        if method == "Kernel density (smooth)":
            return (
                f"Values are sampled from a Gaussian KDE fitted with Silverman's "
                f"bandwidth. This smooths out individual source records — the "
                f"synthetic value never directly equals a source record."
            )
        if method == "Differential privacy (Laplace)":
            return (
                f"Base empirical sampling is followed by Laplace(0, sensitivity/ε) "
                f"noise with ε = {epsilon}. Stronger formal privacy at the cost "
                f"of per-field fidelity."
            )
        return (
            f"Empirical resampling with calibrated Gaussian noise. Noise scale "
            f"is proportional to source std and the noise_level control. "
            f"Integer-preserving for fields that are naturally integer-valued."
        )
    if role in {"categorical", "binary"}:
        if action == "Group rare categories":
            return (
                f"Rare categories are collapsed into 'Other' before frequency "
                f"sampling. Protects low-count subgroups (e.g. rare diagnoses) "
                f"from re-identification."
            )
        if action == "Group text":
            return (
                f"Free-text-style values are collapsed to the top 8 most common, "
                f"plus 'Other'. Reduces chance of memorable source strings leaking."
            )
        if action == "Coarse geography":
            return (
                f"Geographic field coarsened to 3-char prefix (e.g. full postal → "
                f"FSA). Matches PIPEDA guidance for de-identified datasets."
            )
        return (
            f"Category frequencies are sampled with smoothing to avoid exact "
            f"replication of the source distribution, with optional rare-case "
            f"boost controlled by rare_case_retention."
        )
    return f"Handling: {action}. Generation: {method}."


def build_field_impact(item: dict[str, Any], controls: dict[str, Any]) -> tuple[str, str]:
    """Return (impact_label, impact_tone). Tone: good / warn / neutral / strong."""
    included = bool(item.get("include"))
    action = str(item.get("control_action", "Preserve"))
    sens = metadata_sensitivity(item)

    if not included or action == "Exclude":
        return ("Will NOT appear in synthetic output", "neutral")
    if sens == "Restricted":
        if action in {"Tokenize", "Exclude", "Month only", "Coarse geography", "Group text"}:
            return ("Restricted field protected by strong handling", "good")
        return ("Restricted field with lighter handling — review carefully", "warn")
    if sens == "Sensitive":
        if action in {"Month only", "Coarse geography", "Group rare categories", "Group text", "Clip extremes"}:
            return ("Sensitive field with appropriate generalization", "good")
        return ("Sensitive field preserved as-is — consider tighter action", "warn")
    return ("Operational field preserved for analytical use", "neutral")


def _render_sensitivity_badge(name: str) -> str:
    icon, color, bg = SENSITIVITY_BADGES.get(name, ("⚪", "#475569", "#F1F5F9"))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
        f'padding:0.2rem 0.55rem;background:{bg};border-radius:999px;'
        f'font-size:0.78rem;font-weight:700;color:{color};">'
        f'{icon} {name}</span>'
    )


def _render_field_detail_panel(
    item: dict[str, Any],
    controls: dict[str, Any],
    can_edit: bool,
) -> None:
    """Inspect-a-field panel: shown below the main table after the user picks
    a row. Replaces the old Notes column."""
    column = str(item.get("column", ""))
    role = str(item.get("data_type", "")).lower()
    action = str(item.get("control_action", "Preserve"))
    method = resolve_generation_method(item, controls)
    sensitivity = metadata_sensitivity(item)
    badge_html = _render_sensitivity_badge(sensitivity)
    impact_text, impact_tone = build_field_impact(item, controls)
    rationale = build_field_rationale(item, controls)

    tone_styles = {
        "good":    ("#136B48", "#EDF9F3"),
        "strong":  ("#08467d", "#EBF1F7"),
        "warn":    ("#9C6A17", "#FFF6E3"),
        "neutral": ("#475569", "#F1F5F9"),
    }
    impact_color, impact_bg = tone_styles.get(impact_tone, tone_styles["neutral"])

    st.markdown(
        '<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#0b5ea8;margin-bottom:0.25rem;">Field inspector</div>'
        f'<div style="display:flex;align-items:center;gap:0.7rem;flex-wrap:wrap;margin-bottom:0.3rem;">'
        f'<div style="font-size:1.3rem;font-weight:700;color:#17324d;font-family:ui-monospace,monospace;">{column}</div>'
        f'{badge_html}'
        f'<span style="padding:0.15rem 0.55rem;background:#F1F5F9;border-radius:999px;font-size:0.76rem;color:#475569;font-weight:600;">{role}</span>'
        f'<span style="padding:0.15rem 0.55rem;background:{"#EDF9F3" if item.get("include") else "#F1F5F9"};border-radius:999px;font-size:0.76rem;color:{"#136B48" if item.get("include") else "#668097"};font-weight:600;">{"Included" if item.get("include") else "Excluded"}</span>'
        f'</div>'
        f'<div style="display:flex;gap:0.5rem;margin-bottom:0.7rem;flex-wrap:wrap;">'
        f'<div style="padding:0.45rem 0.7rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:8px;font-size:0.78rem;color:#475569;">'
        f'<span style="color:#0b5ea8;font-weight:700;">Privacy handling:</span> {action}</div>'
        f'<div style="padding:0.45rem 0.7rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:8px;font-size:0.78rem;color:#475569;">'
        f'<span style="color:#0b5ea8;font-weight:700;">Generation method:</span> {method}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#0b5ea8;margin-bottom:0.25rem;">Rationale</div>'
        f'<div style="font-size:0.88rem;color:#17324d;line-height:1.55;margin-bottom:0.7rem;">{rationale}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#0b5ea8;margin-bottom:0.25rem;">Expected impact</div>'
        f'<div style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.22rem 0.6rem;background:{impact_bg};border-radius:999px;font-size:0.82rem;color:{impact_color};font-weight:700;margin-bottom:0.7rem;">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{impact_color};"></span>{impact_text}</div>',
        unsafe_allow_html=True,
    )

    if can_edit:
        current_strategy = str(item.get("strategy", "auto"))
        strategy_opts = ["auto", "empirical", "kde", "copula", "dp_laplace", "identifier"]
        if current_strategy not in strategy_opts:
            current_strategy = "auto"
        strat_friendly = {
            "auto":        "Auto (agent picks based on role)",
            "empirical":   "Empirical + Gaussian noise",
            "kde":         "Kernel density (smooth)",
            "copula":      "Gaussian Copula (multivariate)",
            "dp_laplace":  "Differential privacy (Laplace)",
            "identifier":  "Surrogate token",
        }
        col_a, col_b = st.columns([1, 1], gap="medium")
        with col_a:
            override_choice = st.selectbox(
                "Generation-method override",
                options=strategy_opts,
                index=strategy_opts.index(current_strategy),
                format_func=lambda s: strat_friendly.get(s, s),
                key=f"strategy_override_{column}",
                help="Advanced: force a specific sampling algorithm for this field. Leave as Auto for role-based dispatch.",
            )
        with col_b:
            notes_value = st.text_area(
                "Notes",
                value=str(item.get("notes", "")),
                key=f"notes_{column}",
                height=90,
                help="Free-text annotation for the governance record.",
            )
        # Persist edits if changed
        frame = st.session_state.metadata_editor_df
        mask = frame["column"] == column
        if mask.any():
            changed = False
            if str(frame.loc[mask, "strategy"].iloc[0]) != override_choice:
                frame.loc[mask, "strategy"] = override_choice
                changed = True
            if str(frame.loc[mask, "notes"].iloc[0]) != notes_value:
                frame.loc[mask, "notes"] = notes_value
                changed = True
            if changed:
                st.session_state.metadata_editor_df = normalize_metadata_frame(frame)
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
    else:
        notes_value = str(item.get("notes", "")).strip()
        if notes_value:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#0b5ea8;margin-bottom:0.25rem;">Notes</div>'
                f'<div style="font-size:0.85rem;color:#475569;line-height:1.55;white-space:pre-wrap;">{notes_value}</div>',
                unsafe_allow_html=True,
            )


def _render_dual_dist_bars(original: dict, synthetic: dict, title: str) -> None:
    """Side-by-side original vs synthetic distribution bars."""
    all_labels = list(dict.fromkeys(list(original.keys()) + list(synthetic.keys())))
    if not all_labels:
        return
    max_v = max(
        max((original.get(lbl, 0) for lbl in all_labels), default=0),
        max((synthetic.get(lbl, 0) for lbl in all_labels), default=0),
        1.0,
    )
    st.markdown(
        f'<div style="font-size:0.82rem;font-weight:600;color:#17324d;margin:0.5rem 0 0.35rem 0;">{title}</div>'
        f'<div style="display:flex;gap:0.8rem;font-size:0.72rem;color:#668097;margin-bottom:0.3rem;">'
        f'<div style="display:inline-flex;align-items:center;gap:0.3rem;"><span style="width:10px;height:10px;background:#0b5ea8;border-radius:2px;"></span>Original</div>'
        f'<div style="display:inline-flex;align-items:center;gap:0.3rem;"><span style="width:10px;height:10px;background:#19CBC5;border-radius:2px;"></span>Synthetic</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    html = '<div style="display:flex;flex-direction:column;gap:0.3rem;">'
    for label in all_labels[:10]:
        o_val = float(original.get(label, 0.0))
        s_val = float(synthetic.get(label, 0.0))
        o_width = (o_val / max_v * 100)
        s_width = (s_val / max_v * 100)
        html += (
            f'<div style="display:flex;align-items:center;gap:0.6rem;">'
            f'<div style="min-width:120px;max-width:160px;font-size:0.78rem;color:#17324d;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{label}">{label}</div>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:2px;">'
            f'<div style="height:10px;background:#F1F5F9;border-radius:3px;overflow:hidden;"><div style="width:{o_width}%;height:100%;background:#0b5ea8;"></div></div>'
            f'<div style="height:10px;background:#F1F5F9;border-radius:3px;overflow:hidden;"><div style="width:{s_width}%;height:100%;background:#19CBC5;"></div></div>'
            f'</div>'
            f'<div style="min-width:90px;font-size:0.72rem;color:#475569;text-align:right;">{o_val:.1f}% / {s_val:.1f}%</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _field_distribution(series: pd.Series, role: str) -> dict:
    """Return distribution dict: label -> share %."""
    if role == "numeric":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {}
        try:
            bins = pd.cut(numeric, bins=5, duplicates="drop")
            counts = bins.value_counts(normalize=True).sort_index().mul(100)
            return {f"{interval.left:.1f}-{interval.right:.1f}": round(float(v), 2) for interval, v in counts.items()}
        except Exception:
            return {}
    elif role == "date":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
        if parsed.empty:
            return {}
        try:
            by_month = parsed.dt.to_period("M").value_counts(normalize=True).sort_index().mul(100)
            return {str(k): round(float(v), 2) for k, v in list(by_month.items())[:10]}
        except Exception:
            return {}
    else:
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return {}
        counts = non_null.value_counts(normalize=True).mul(100).head(10)
        return {str(k): round(float(v), 2) for k, v in counts.items()}


def _render_preview_panel(metadata, controls, read_only=False):
    """Shared preview rendering used by both analyst (Step 3) and manager (Step 4)."""
    if st.session_state.synthetic_df is None:
        return

    synthetic_df = st.session_state.synthetic_df
    source_df = st.session_state.source_df
    validation = st.session_state.validation or {}
    summary = st.session_state.generation_summary or {}

    with st.container(border=True, key="synthetic_preview_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Synthetic preview &amp; quality metrics</div>'
            f'<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.7rem;line-height:1.3;">{len(synthetic_df)} synthetic rows across {len(synthetic_df.columns)} fields</div>',
            unsafe_allow_html=True,
        )

        # Quality metric capsules
        fidelity = validation.get("fidelity_score", 0)
        privacy = validation.get("privacy_score", 0)
        correlation = validation.get("correlation_score", 0)
        overall = validation.get("overall_score", 0)

        def _score_accent(score):
            if score >= 80: return "#136B48", "#EDF9F3"
            if score >= 60: return "#9C6A17", "#FFF6E3"
            return "#9d2b3c", "#fff1f3"

        f_c, f_bg = _score_accent(fidelity)
        p_c, p_bg = _score_accent(privacy)
        c_c, c_bg = _score_accent(correlation)
        o_c, o_bg = _score_accent(overall)

        st.markdown(
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;margin-bottom:0.9rem;">'
            + _stat_capsule_v2("Overall quality", f"{overall:.1f}", "Combined fidelity + privacy", accent=o_c, bg=o_bg)
            + _stat_capsule_v2("Fidelity", f"{fidelity:.1f}", "How well distributions match", accent=f_c, bg=f_bg)
            + _stat_capsule_v2("Privacy", f"{privacy:.1f}", f"epsilon = {summary.get('privacy_epsilon', 'n/a')}", accent=p_c, bg=p_bg)
            + _stat_capsule_v2("Correlation", f"{correlation:.1f}", "Joint relationship preservation", accent=c_c, bg=c_bg)
            + '</div>',
            unsafe_allow_html=True,
        )

        # Algorithm transparency
        strategy_log = summary.get("strategy_log", [])
        copula_cols = summary.get("copula_columns", [])
        constraints_list = summary.get("detected_constraints", [])
        repairs = summary.get("constraint_repairs", [])

        algo_html = '<div style="display:flex;gap:1.2rem;padding:0.7rem 0.9rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:10px;margin-bottom:0.8rem;font-size:0.82rem;flex-wrap:wrap;">'
        algo_html += f'<div><span style="font-weight:700;color:#0b5ea8;">Algorithm:</span> Gaussian Copula ({len(copula_cols)} fields), DP Laplace (epsilon = {summary.get("privacy_epsilon", "n/a")})</div>'
        if constraints_list:
            algo_html += f'<div><span style="font-weight:700;color:#0b5ea8;">Constraints detected:</span> {len(constraints_list)}</div>'
        if repairs:
            total_repairs = sum(r.get("rows_repaired", 0) for r in repairs)
            algo_html += f'<div><span style="font-weight:700;color:#9C6A17;">Repairs applied:</span> {total_repairs} row(s) across {len(repairs)} rule(s)</div>'
        algo_html += '</div>'
        st.markdown(algo_html, unsafe_allow_html=True)

        preview_tabs = st.tabs(["Sample rows", "Per-field comparison", "Constraints", "Strategy log"])

        with preview_tabs[0]:
            st.dataframe(synthetic_df.head(20), use_container_width=True, hide_index=True)

        with preview_tabs[1]:
            comparable = [m for m in metadata if m.get("include") and m["column"] in synthetic_df.columns and m["column"] in source_df.columns and m.get("data_type") != "identifier"]
            if not comparable:
                st.info("No comparable fields available.")
            else:
                picked = st.selectbox(
                    "Select a field to compare distributions",
                    options=[m["column"] for m in comparable],
                    key="preview_field_selector_mgr" if read_only else "preview_field_selector",
                )
                picked_meta = next((m for m in comparable if m["column"] == picked), None)
                if picked_meta:
                    role = picked_meta["data_type"]
                    orig_dist = _field_distribution(source_df[picked], role)
                    syn_dist = _field_distribution(synthetic_df[picked], role)
                    _render_dual_dist_bars(orig_dist, syn_dist, f"{picked} ({role})")

        with preview_tabs[2]:
            if constraints_list:
                st.markdown("**Auto-detected logical constraints in source data**")
                cons_df = pd.DataFrame([
                    {"Rule": c["rule"], "Type": c["kind"], "Confidence": f"{c['confidence']*100:.1f}%"}
                    for c in constraints_list
                ])
                st.dataframe(cons_df, use_container_width=True, hide_index=True)
            else:
                st.info("No logical constraints detected.")
            if repairs:
                st.markdown("**Constraint repairs applied to synthetic output**")
                rep_df = pd.DataFrame(repairs)
                st.dataframe(rep_df, use_container_width=True, hide_index=True)

        with preview_tabs[3]:
            if strategy_log:
                strat_df = pd.DataFrame(strategy_log)
                st.dataframe(strat_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strategy log.")

            drift_pairs = validation.get("correlation_details", {}).get("drift_pairs", [])
            if drift_pairs:
                st.markdown("**Top correlation drift pairs** (original vs synthetic)")
                drift_df = pd.DataFrame(drift_pairs[:6])
                st.dataframe(drift_df, use_container_width=True, hide_index=True)


def render_step_three() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Step 3 — Configure & Generate Package (analyst)."""
    step_cfg = STEP_CONFIG[2]
    st.markdown(
        f"""
        <div class="section-shell" style="margin-bottom:0.9rem;">
            <h3 style="margin:0;">{step_cfg["title"]}</h3>
            <div style="font-size:0.92rem;color:#668097;margin-top:0.3rem;line-height:1.55;">
                Configure metadata, generate a synthetic preview, iterate on the algorithm, then submit the complete package for reviewer sign-off.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_active_dataset():
        st.info("Upload a dataset to configure the synthetic package.")
        return [], st.session_state.controls.copy()

    controls = st.session_state.controls.copy()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)

    active_package = active_metadata_package_record(metadata)
    status = st.session_state.metadata_status

    if status == "Approved":
        s_color, s_bg, s_label = "#136B48", "#EDF9F3", "Approved"
    elif status == "In Review":
        s_color, s_bg, s_label = "#0b5ea8", "#EBF1F7", "In review"
    elif status == "Changes Requested":
        s_color, s_bg, s_label = "#9C6A17", "#FFF6E3", "Changes requested"
    elif status == "Rejected":
        s_color, s_bg, s_label = "#9d2b3c", "#fff1f3", "Rejected"
    else:
        s_color, s_bg, s_label = "#668097", "#F1F5F9", "Draft"

    _render_status_strip_v2(2, s_label, s_color, s_bg)

    # Callout for reviewer feedback
    if status == "Changes Requested" and active_package is not None:
        note = active_package.get("review_note") or "Reviewer requested changes."
        reviewer = active_package.get("reviewed_by") or "Reviewer"
        reviewed_at = active_package.get("reviewed_at") or ""
        st.markdown(
            f'<div style="padding:1rem 1.15rem;background:#FFF6E3;border:1px solid #F3DBA6;border-radius:14px;margin-bottom:0.9rem;">'
            f'<div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#9C6A17;margin-bottom:0.25rem;">Changes requested by reviewer</div>'
            f'<div style="font-size:0.9rem;color:#17324d;line-height:1.5;margin-bottom:0.25rem;"><strong>{reviewer}</strong>{" at " + reviewed_at if reviewed_at else ""}</div>'
            f'<div style="font-size:0.88rem;color:#475569;line-height:1.55;">{note}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    metadata_frame = build_metadata_review_frame(metadata)
    included = int(st.session_state.metadata_editor_df["include"].sum())
    total = len(st.session_state.metadata_editor_df)
    excluded = total - included
    restricted = int(metadata_frame["Sensitivity"].eq("Restricted").sum())
    sensitive = int(metadata_frame["Sensitivity"].eq("Sensitive").sum())

    has_preview = st.session_state.synthetic_df is not None

    # Package summary
    with st.container(border=True, key="package_summary_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Package summary</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">Overview of what will be submitted</div>'
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
            + _stat_capsule_v2("Fields included", f"{included}/{total}", f"{excluded} excluded from synthesis")
            + _stat_capsule_v2("Sensitive fields", str(restricted + sensitive),
                           f"{restricted} restricted, {sensitive} sensitive" if (restricted + sensitive) > 0 else "None flagged",
                           accent=("#9d2b3c" if restricted > 0 else "#9C6A17" if sensitive > 0 else "#0b5ea8"),
                           bg=("#fff1f3" if restricted > 0 else "#FFF6E3" if sensitive > 0 else "#f5f9fc"))
            + _stat_capsule_v2("Output rows", str(int(controls.get("synthetic_rows", 500))), "Synthetic dataset size")
            + _stat_capsule_v2("Preview status", "Generated" if has_preview else "Not yet generated",
                           "Click Generate below" if not has_preview else f"{len(st.session_state.synthetic_df)} rows ready",
                           accent=("#136B48" if has_preview else "#9C6A17"),
                           bg=("#EDF9F3" if has_preview else "#FFF6E3"))
            + '</div>',
            unsafe_allow_html=True,
        )

    can_edit = has_permission("edit_metadata") and status not in {"In Review", "Approved"}

    # Field settings
    with st.container(border=True, key="field_settings_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Field settings</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.3rem;line-height:1.3;">Per-field privacy handling &amp; generation method</div>'
            '<div style="font-size:0.82rem;color:#668097;margin-bottom:0.7rem;line-height:1.5;">Restricted fields first. <strong>Privacy handling</strong> controls how the source value is generalised before synthesis; <strong>Generation method</strong> is the algorithm used to produce each synthetic value.</div>',
            unsafe_allow_html=True,
        )

        if can_edit:
            bulk_cols = st.columns(3)
            bulk_defs = [
                ("Apply stricter privacy controls", "tighten_phi",
                 "More grouping, coarser dates / geography, fewer high-risk fields preserved."),
                ("Preserve more analytical detail", "preserve_analytics",
                 "Retain more fields with lighter transformations. Best for flow modelling."),
                ("Reset field settings", "reset_defaults",
                 "Rebuild from profiler defaults (identifiers tokenised, dates date-shifted)."),
            ]
            for idx, (label, mode, caption) in enumerate(bulk_defs):
                with bulk_cols[idx]:
                    if st.button(label, use_container_width=True, key=f"bulk_profile_{mode}"):
                        apply_bulk_metadata_profile(mode)
                        st.session_state.settings_reviewed = False
                        st.session_state.settings_review_signature = None
                        record_audit_event("Bulk metadata profile applied",
                            f"Profile: {label}.", status="Updated")
                        rerun_with_persist()
                    st.caption(caption)
            st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)

        strategy_opts = ["auto", "empirical", "kde", "copula", "dp_laplace", "identifier"]
        editor_source = st.session_state.metadata_editor_df.copy()
        metadata_items_for_display = editor_frame_to_metadata(editor_source)
        editor_source["Sensitivity"] = [metadata_sensitivity(item) for item in metadata_items_for_display]
        editor_source["strategy"] = editor_source["strategy"].apply(lambda s: s if s in strategy_opts else "auto")
        # Compute read-only "Generation method" column from the resolved strategy
        editor_source["generation_method"] = [
            resolve_generation_method(item, controls) for item in metadata_items_for_display
        ]
        # Sort: Restricted → Sensitive → Operational, then column name
        editor_source["_sens_order"] = editor_source["Sensitivity"].map(SENSITIVITY_ORDER).fillna(99)
        editor_source = editor_source.sort_values(by=["_sens_order", "column"]).drop(columns=["_sens_order"])
        # Decorate sensitivity with an emoji for the main table
        editor_source["Sensitivity"] = editor_source["Sensitivity"].map(sensitivity_display_label)
        # Final column order shown to user (no strategy / notes — those move to detail panel)
        display_cols = ["column", "Sensitivity", "data_type", "include", "control_action", "generation_method"]
        editor_source_display = editor_source[display_cols]

        if can_edit:
            edited = st.data_editor(
                editor_source_display,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key="metadata_editor_unified_v2",
                column_config={
                    "column": st.column_config.TextColumn("Field", disabled=True, width="medium"),
                    "Sensitivity": st.column_config.TextColumn("Sensitivity", disabled=True, width="small",
                        help="Restricted: identifiers. Sensitive: dates, postal, free-text. Operational: everything else."),
                    "data_type": st.column_config.TextColumn("Type", disabled=True, width="small"),
                    "include": st.column_config.CheckboxColumn("Include", width="small"),
                    "control_action": st.column_config.SelectboxColumn("Privacy handling",
                        options=ALL_CONTROL_ACTIONS, width="medium",
                        help="Generalisation applied to the source value BEFORE synthesis (e.g. Tokenize, Month only, Coarse geography, Group text). 'Exclude' drops the field from synthetic output."),
                    "generation_method": st.column_config.TextColumn("Generation method",
                        disabled=True, width="medium",
                        help="Algorithm used to produce each synthetic value. Resolved from data type + Gaussian Copula toggle. Override per-field in the inspector below."),
                },
            )
            updated_frame = st.session_state.metadata_editor_df.copy()
            for _, row in edited.iterrows():
                mask = updated_frame["column"] == row["column"]
                if not mask.any(): continue
                base_item = editor_frame_to_metadata(updated_frame.loc[mask].head(1))[0]
                chosen_action = sanitize_control_action(base_item, str(row["control_action"]))
                updated_frame.loc[mask, "control_action"] = chosen_action
                updated_frame.loc[mask, "include"] = bool(row["include"]) and chosen_action != "Exclude"
            updated_frame = normalize_metadata_frame(updated_frame)
            if not updated_frame.equals(st.session_state.metadata_editor_df):
                st.session_state.metadata_editor_df = updated_frame
                st.session_state.settings_reviewed = False
                st.session_state.settings_review_signature = None
        else:
            st.dataframe(editor_source_display, use_container_width=True, hide_index=True)

        # ── Field inspector (detail panel, replaces Notes column) ──
        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
        inspect_options = editor_source["column"].tolist()
        if inspect_options:
            default_field = st.session_state.get("field_inspector_selection") or inspect_options[0]
            if default_field not in inspect_options:
                default_field = inspect_options[0]
            inspect_choice = st.selectbox(
                "Inspect field",
                options=inspect_options,
                index=inspect_options.index(default_field),
                key="field_inspector_selection",
                help="Click a field to see rationale, expected impact, and override the generation method.",
            )
            inspect_item = next((m for m in editor_frame_to_metadata(st.session_state.metadata_editor_df)
                                 if m["column"] == inspect_choice), None)
            if inspect_item is not None:
                with st.container(border=True, key=f"field_inspector_body"):
                    _render_field_detail_panel(inspect_item, controls, can_edit=can_edit)

    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)

    source_rows = len(st.session_state.source_df) if st.session_state.source_df is not None else 0

    # Generation strategy
    with st.container(border=True, key="generation_strategy_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Generation algorithm</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Release profile &amp; fine-tune controls</div>',
            unsafe_allow_html=True,
        )

        if can_edit:
            # ── Release profile (high-level use case mode) ────────────
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#0b5ea8;margin-bottom:0.25rem;">Release profile</div>'
                '<div style="font-size:0.85rem;color:#475569;margin-bottom:0.5rem;line-height:1.5;">Pick a use case — the agent sets privacy, structure, noise, and outlier handling in one click. You can still fine-tune below.</div>',
                unsafe_allow_html=True,
            )
            active_profile = str(controls.get("generation_preset", "Internal prototyping"))
            profile_cols = st.columns(len(GENERATION_PRESETS), gap="small")
            for idx, (profile_name, profile_cfg) in enumerate(GENERATION_PRESETS.items()):
                is_active = profile_name == active_profile
                btn_label = f"✓ {profile_name}" if is_active else profile_name
                if profile_cols[idx].button(
                    btn_label,
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    key=f"release_profile_{idx}",
                    help=profile_cfg.get("description"),
                ):
                    controls = apply_generation_preset(controls, profile_name)
                    st.session_state.controls = controls
                    st.session_state.settings_reviewed = False
                    st.session_state.settings_review_signature = None
                    record_audit_event("Release profile applied",
                        f"Profile switched to {profile_name}.", status="Updated")
                    rerun_with_persist()
            active_desc = GENERATION_PRESETS.get(active_profile, {}).get("description",
                "Custom configuration. Fine-tune controls below.")
            st.markdown(
                f'<div style="padding:0.55rem 0.8rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:10px;margin-bottom:0.9rem;font-size:0.82rem;color:#475569;line-height:1.5;">'
                f'<span style="color:#0b5ea8;font-weight:700;">Active · {active_profile}</span> &nbsp;{active_desc}</div>',
                unsafe_allow_html=True,
            )

            cols_a = st.columns([1, 1], gap="large")
            with cols_a[0]:
                preset_list = list(PRIVACY_PRESETS.keys())
                current_preset = str(controls.get("privacy_preset", "Balanced"))
                if current_preset not in preset_list:
                    current_preset = "Balanced"
                selected_preset = st.selectbox("Privacy preset", options=preset_list,
                    index=preset_list.index(current_preset),
                    help="Higher privacy adds more noise via Laplace mechanism.")
                controls["privacy_preset"] = selected_preset
                controls["privacy_epsilon"] = PRIVACY_PRESETS[selected_preset]
                st.caption(f"Privacy budget epsilon = {PRIVACY_PRESETS[selected_preset]}  |  Lower epsilon = stronger privacy (more noise).")

                # ── Output rows: quick choice + conditional Custom ────
                st.markdown('<div style="height:0.2rem;"></div>', unsafe_allow_html=True)
                row_options = ["Same as source", "50% of source", "2× source", "Custom"]
                current_choice = str(controls.get("row_count_choice", "Same as source"))
                if current_choice not in row_options:
                    current_choice = "Custom"
                row_choice = st.radio(
                    "Output rows",
                    options=row_options,
                    index=row_options.index(current_choice),
                    horizontal=True,
                    key="row_count_choice_radio",
                    help="Most users pick 'Same as source' for realistic scale, or '2× source' for stress-testing downstream pipelines.",
                )
                controls["row_count_choice"] = row_choice
                if row_choice == "Same as source":
                    resolved_rows = source_rows
                elif row_choice == "50% of source":
                    resolved_rows = max(10, source_rows // 2)
                elif row_choice == "2× source":
                    resolved_rows = max(10, source_rows * 2)
                else:
                    resolved_rows = int(st.number_input(
                        "Custom row count",
                        value=int(controls.get("synthetic_rows", max(source_rows, 100))),
                        min_value=10, max_value=100000, step=50,
                    ))
                controls["synthetic_rows"] = max(10, int(resolved_rows))
                st.caption(f"Source has {source_rows:,} rows. Synthetic output will be {controls['synthetic_rows']:,} rows.")

            with cols_a[1]:
                controls["use_copula"] = st.checkbox(
                    "Fit Gaussian Copula for multivariate correlation",
                    value=bool(controls.get("use_copula", True)),
                    help="Preserves joint distribution across numeric fields. Essential for realistic healthcare data.",
                )
                if controls["use_copula"]:
                    controls["copula_strength"] = st.slider("Copula correlation strength", 0, 100,
                        int(controls.get("copula_strength", 80)),
                        help="0 = independent fields; 100 = full source correlations.")
                controls["enforce_constraints"] = st.checkbox(
                    "Auto-enforce detected constraints",
                    value=bool(controls.get("enforce_constraints", True)),
                    help="Detects rules like admission <= discharge and repairs violations.",
                )

            with st.expander("Advanced parameters", expanded=False):
                adv = st.columns(2)
                with adv[0]:
                    controls["correlation_preservation"] = st.slider("Correlation preservation", 0, 100,
                        int(controls.get("correlation_preservation", 65)))
                    controls["rare_case_retention"] = st.slider("Rare case retention", 0, 100,
                        int(controls.get("rare_case_retention", 30)))
                    controls["noise_level"] = st.slider("Per-field noise scale", 0, 100,
                        int(controls.get("noise_level", 45)))
                with adv[1]:
                    miss_opts = ["Preserve source pattern", "Reduce missingness", "Fill gaps"]
                    mp = controls.get("missingness_pattern", "Preserve source pattern")
                    if mp not in miss_opts: mp = "Preserve source pattern"
                    controls["missingness_pattern"] = st.selectbox("Missingness pattern", miss_opts, index=miss_opts.index(mp))
                    out_opts = ["Preserve tails", "Clip extremes", "Smooth tails"]
                    ov = controls.get("outlier_strategy", "Preserve tails")
                    if ov not in out_opts: ov = "Preserve tails"
                    controls["outlier_strategy"] = st.selectbox("Outlier strategy", out_opts, index=out_opts.index(ov))

            controls = sync_generation_preset_label(controls)
            st.session_state.controls = controls
        else:
            active_profile = str(controls.get("generation_preset", "Internal prototyping"))
            profile_desc = GENERATION_PRESETS.get(active_profile, {}).get("description",
                "Custom configuration.")
            ro_rows = [
                f"Release profile: <strong>{active_profile}</strong>",
                f"Privacy preset: <strong>{controls.get('privacy_preset', 'Balanced')}</strong> (epsilon = {controls.get('privacy_epsilon', 2.0)})",
                f"Output rows: <strong>{int(controls.get('synthetic_rows', 500))}</strong> ({controls.get('row_count_choice', 'Custom')})",
                f"Gaussian Copula: <strong>{'Enabled' if controls.get('use_copula') else 'Disabled'}</strong>" + (f" (strength {controls.get('copula_strength', 80)}%)" if controls.get('use_copula') else ""),
                f"Constraint enforcement: <strong>{'On' if controls.get('enforce_constraints', True) else 'Off'}</strong>",
            ]
            st.markdown('<div style="font-size:0.88rem;color:#475569;line-height:1.8;">' + '<br/>'.join(ro_rows) + '</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-top:0.6rem;padding:0.55rem 0.8rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:10px;font-size:0.82rem;color:#475569;line-height:1.5;">{profile_desc}</div>',
                unsafe_allow_html=True,
            )

    # Expected outcome — live forecast from current controls
    with st.container(border=True, key="expected_outcome_panel"):
        _render_expected_outcome_card(controls, metadata)

    # Generate preview button
    if can_edit and has_permission("generate"):
        g1, g2 = st.columns([1, 3])
        if g1.button("Generate preview" if not has_preview else "Regenerate preview",
                     type="primary", use_container_width=True, key="step3_gen_btn"):
            with st.spinner("Fitting Gaussian Copula, sampling synthetic data, enforcing constraints, computing quality metrics..."):
                synthetic_df, gen_summary = generate_synthetic_advanced(
                    st.session_state.source_df, metadata, controls)
                validation = validate_synthetic_data(
                    st.session_state.source_df, synthetic_df, metadata, controls)
                st.session_state.synthetic_df = synthetic_df
                st.session_state.generation_summary = gen_summary
                st.session_state.validation = validation
                st.session_state.last_generation_signature = build_generation_signature(metadata, controls)
                st.session_state.release_status = "Generated"
                record_audit_event("Synthetic preview generated",
                    f"{gen_summary['rows_generated']} rows with {len(gen_summary['copula_columns'])} copula fields, epsilon = {gen_summary['privacy_epsilon']}.",
                    status="Completed")
                rerun_with_persist()
        if has_preview:
            g2.markdown(
                '<div style="padding:0.5rem 0.8rem;background:#EDF9F3;border:1px solid #B8E3CC;border-radius:10px;font-size:0.85rem;color:#136B48;">'
                f'Preview ready - {len(st.session_state.synthetic_df)} rows generated'
                '</div>',
                unsafe_allow_html=True,
            )

    # Preview panel
    if has_preview:
        _render_preview_panel(metadata, controls, read_only=not can_edit)

    # Action bar
    with st.container(border=True, key="action_bar_panel"):
        if has_permission("submit_metadata"):
            if status == "Approved":
                st.markdown(
                    '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#136B48;margin-bottom:0.3rem;">Approved</div>'
                    f'<div style="font-size:0.95rem;color:#17324d;line-height:1.5;margin-bottom:0.8rem;">Package {active_package["package_id"] if active_package else ""} approved. Proceed to release.</div>',
                    unsafe_allow_html=True,
                )
                acts = st.columns([1, 1], gap="small")
                if acts[0].button("Revise draft", use_container_width=True):
                    st.session_state.metadata_status = "Draft"
                    st.session_state.settings_reviewed = False
                    st.session_state.settings_review_signature = None
                    record_audit_event("Package draft reopened", "Revised draft.", status="Updated")
                    rerun_with_persist()
                if acts[1].button("Continue to release", type="primary", use_container_width=True):
                    st.session_state.current_step = 4
                    st.rerun()
            elif status == "In Review":
                st.markdown(
                    '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">With reviewer</div>'
                    '<div style="font-size:0.95rem;color:#17324d;line-height:1.5;margin-bottom:0.8rem;">Package submitted to Manager / Reviewer. Awaiting sign-off.</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Revise draft", use_container_width=True):
                    st.session_state.metadata_status = "Draft"
                    st.session_state.settings_reviewed = False
                    st.session_state.settings_review_signature = None
                    record_audit_event("Package draft reopened", "Revised draft.", status="Updated")
                    rerun_with_persist()
            else:
                st.markdown(
                    '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Submit for review</div>'
                    '<div style="font-size:0.95rem;color:#17324d;line-height:1.5;margin-bottom:0.8rem;">Once you are happy with the preview and quality metrics, submit the complete package to the Manager / Reviewer.</div>',
                    unsafe_allow_html=True,
                )
                current_signature = build_metadata_signature(metadata)
                ab = st.columns([1, 1], gap="small")
                can_submit = has_preview and not has_stale_generation(metadata, controls)
                if ab[0].button("Submit package for review", type="primary",
                                use_container_width=True, disabled=not can_submit,
                                help=None if can_submit else "Generate a preview first"):
                    st.session_state.settings_reviewed = True
                    st.session_state.settings_review_signature = current_signature
                    st.session_state.metadata_status = "In Review"
                    st.session_state.metadata_submitted_by = st.session_state.current_role
                    st.session_state.metadata_submitted_at = format_timestamp()
                    st.session_state.metadata_approved_by = None
                    st.session_state.metadata_approved_at = None
                    st.session_state.metadata_review_note = None
                    st.session_state.metadata_reviewed_by = None
                    st.session_state.metadata_reviewed_at = None
                    st.session_state.last_reviewed_metadata_signature = current_signature
                    submitted_record = register_metadata_submission(metadata)
                    record_audit_event("Package submitted for review",
                        f"{submitted_record['package_id']} submitted with {len(st.session_state.synthetic_df)} synthetic rows.",
                        status="Submitted")
                    rerun_with_persist()
                if ab[1].button("Back to Scan", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()

    return metadata, controls


def render_step_four(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Step 4 - Manager Review & Approve (manager sees full submitted package)."""
    step_cfg = STEP_CONFIG[3]
    st.markdown(
        f"""
        <div class="section-shell" style="margin-bottom:0.9rem;">
            <h3 style="margin:0;">{step_cfg["title"]}</h3>
            <div style="font-size:0.92rem;color:#668097;margin-top:0.3rem;line-height:1.55;">
                Inspect the submitted package — metadata configuration, generation algorithm settings, synthetic preview, and quality metrics — before approval.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_active_dataset():
        st.info("No dataset in workspace.")
        return metadata, controls

    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls

    status = st.session_state.metadata_status
    active_package = active_metadata_package_record(metadata)
    review_package = current_review_package_record()

    if status == "Approved":
        s_color, s_bg, s_label = "#136B48", "#EDF9F3", "Approved"
    elif status == "In Review":
        s_color, s_bg, s_label = "#0b5ea8", "#EBF1F7", "Awaiting review"
    elif status == "Changes Requested":
        s_color, s_bg, s_label = "#9C6A17", "#FFF6E3", "Returned to analyst"
    elif status == "Rejected":
        s_color, s_bg, s_label = "#9d2b3c", "#fff1f3", "Rejected"
    else:
        s_color, s_bg, s_label = "#668097", "#F1F5F9", "Waiting"

    _render_status_strip_v2(3, s_label, s_color, s_bg)

    if status not in {"In Review", "Approved"}:
        st.info("No package is currently submitted for review. Waiting for the Data Analyst to submit.")
        if st.button("Refresh"):
            st.rerun()
        return metadata, controls

    # Submission metadata
    metadata_frame = build_metadata_review_frame(metadata)
    included = int(st.session_state.metadata_editor_df["include"].sum())
    total = len(st.session_state.metadata_editor_df)
    restricted = int(metadata_frame["Sensitivity"].eq("Restricted").sum())
    sensitive = int(metadata_frame["Sensitivity"].eq("Sensitive").sum())

    with st.container(border=True, key="submission_summary_panel"):
        pkg_id = (review_package or active_package or {}).get("package_id", "-")
        submitted_by = st.session_state.metadata_submitted_by or "-"
        submitted_at = st.session_state.metadata_submitted_at or "-"
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Submitted package</div>'
            f'<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">{pkg_id} - submitted by {submitted_by} at {submitted_at}</div>'
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
            + _stat_capsule_v2("Fields included", f"{included}/{total}", f"{total - included} excluded")
            + _stat_capsule_v2("Sensitive fields", str(restricted + sensitive),
                           f"{restricted} restricted, {sensitive} sensitive" if (restricted + sensitive) > 0 else "None flagged",
                           accent=("#9d2b3c" if restricted > 0 else "#9C6A17" if sensitive > 0 else "#0b5ea8"),
                           bg=("#fff1f3" if restricted > 0 else "#FFF6E3" if sensitive > 0 else "#f5f9fc"))
            + _stat_capsule_v2("Output rows", str(int(controls.get("synthetic_rows", 500))), "Synthetic size")
            + _stat_capsule_v2("Privacy preset", str(controls.get("privacy_preset", "Balanced")),
                           f"epsilon = {controls.get('privacy_epsilon', 2.0)}")
            + '</div>',
            unsafe_allow_html=True,
        )

    # Read-only metadata
    with st.container(border=True, key="field_settings_panel_review"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Field settings</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.3rem;line-height:1.3;">Per-field privacy handling &amp; generation method</div>'
            '<div style="font-size:0.82rem;color:#668097;margin-bottom:0.7rem;line-height:1.5;">Restricted fields first. Inspect any field below for rationale and expected impact.</div>',
            unsafe_allow_html=True,
        )
        read_source = st.session_state.metadata_editor_df.copy()
        read_items = editor_frame_to_metadata(read_source)
        read_source["Sensitivity"] = [metadata_sensitivity(item) for item in read_items]
        read_source["generation_method"] = [resolve_generation_method(item, controls) for item in read_items]
        read_source["_sens_order"] = read_source["Sensitivity"].map(SENSITIVITY_ORDER).fillna(99)
        read_source = read_source.sort_values(by=["_sens_order", "column"]).drop(columns=["_sens_order"])
        read_source["Sensitivity"] = read_source["Sensitivity"].map(sensitivity_display_label)
        read_display = read_source[["column", "Sensitivity", "data_type", "include", "control_action", "generation_method"]]
        st.dataframe(
            read_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "column": st.column_config.TextColumn("Field"),
                "Sensitivity": st.column_config.TextColumn("Sensitivity"),
                "data_type": st.column_config.TextColumn("Type"),
                "include": st.column_config.CheckboxColumn("Include"),
                "control_action": st.column_config.TextColumn("Privacy handling"),
                "generation_method": st.column_config.TextColumn("Generation method"),
            },
        )

        # Inspect field (read-only detail panel for Manager)
        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
        inspect_options_review = read_source["column"].tolist()
        if inspect_options_review:
            default_review = st.session_state.get("field_inspector_selection_review") or inspect_options_review[0]
            if default_review not in inspect_options_review:
                default_review = inspect_options_review[0]
            inspect_choice_review = st.selectbox(
                "Inspect field",
                options=inspect_options_review,
                index=inspect_options_review.index(default_review),
                key="field_inspector_selection_review",
            )
            inspect_item_review = next((m for m in read_items if m["column"] == inspect_choice_review), None)
            if inspect_item_review is not None:
                with st.container(border=True, key="field_inspector_body_review"):
                    _render_field_detail_panel(inspect_item_review, controls, can_edit=False)

    # Read-only generation strategy
    with st.container(border=True, key="generation_strategy_panel_review"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Generation algorithm</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Release profile and controls used</div>',
            unsafe_allow_html=True,
        )
        active_profile = str(controls.get("generation_preset", "Internal prototyping"))
        profile_desc = GENERATION_PRESETS.get(active_profile, {}).get("description",
            "Custom configuration.")
        ro_rows = [
            f"Release profile: <strong>{active_profile}</strong>",
            f"Privacy preset: <strong>{controls.get('privacy_preset', 'Balanced')}</strong> (epsilon = {controls.get('privacy_epsilon', 2.0)})",
            f"Output rows: <strong>{int(controls.get('synthetic_rows', 500))}</strong> ({controls.get('row_count_choice', 'Custom')})",
            f"Gaussian Copula: <strong>{'Enabled' if controls.get('use_copula') else 'Disabled'}</strong>" + (f" (strength {controls.get('copula_strength', 80)}%)" if controls.get('use_copula') else ""),
            f"Constraint enforcement: <strong>{'On' if controls.get('enforce_constraints', True) else 'Off'}</strong>",
            f"Correlation preservation: <strong>{controls.get('correlation_preservation', 65)}%</strong>",
        ]
        st.markdown('<div style="font-size:0.88rem;color:#475569;line-height:1.8;">' + '<br/>'.join(ro_rows) + '</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:0.6rem;padding:0.55rem 0.8rem;background:#F8FAFC;border:1px solid #E5EDF5;border-radius:10px;font-size:0.82rem;color:#475569;line-height:1.5;">{profile_desc}</div>',
            unsafe_allow_html=True,
        )

    # Expected outcome (read-only, same forecast the analyst saw)
    with st.container(border=True, key="expected_outcome_panel_review"):
        _render_expected_outcome_card(controls, metadata)

    # Preview + metrics
    if st.session_state.synthetic_df is not None:
        _render_preview_panel(metadata, controls, read_only=True)
    else:
        st.warning("Synthetic preview was not included in the submission.")

    # Approval action bar
    with st.container(border=True, key="approval_action_panel"):
        can_approve = has_permission("approve_metadata") or (st.session_state.current_role == "Manager / Reviewer")
        if not can_approve:
            st.info("Read-only view. Your role cannot approve or reject submissions.")
            return metadata, controls

        if status == "Approved":
            st.markdown(
                '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#136B48;margin-bottom:0.3rem;">Approved</div>'
                '<div style="font-size:0.95rem;color:#17324d;line-height:1.5;margin-bottom:0.8rem;">You approved this package. It is now ready for release download.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Continue to release", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
            return metadata, controls

        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Decision</div>'
            '<div style="font-size:0.95rem;color:#17324d;line-height:1.5;margin-bottom:0.5rem;">Review the package above and choose an action.</div>',
            unsafe_allow_html=True,
        )
        review_note = st.text_area(
            "Review note (required for request changes / reject)",
            placeholder="Add approval context or revision guidance for the Data Analyst...",
            key="manager_review_note", height=90,
        )

        action_cols = st.columns([1, 1, 1], gap="small")
        if action_cols[0].button("Approve", type="primary", use_container_width=True):
            st.session_state.metadata_status = "Approved"
            st.session_state.metadata_approved_by = st.session_state.current_role
            st.session_state.metadata_approved_at = format_timestamp()
            st.session_state.metadata_review_note = review_note.strip() or None
            st.session_state.metadata_reviewed_by = st.session_state.current_role
            st.session_state.metadata_reviewed_at = format_timestamp()
            st.session_state.last_reviewed_metadata_signature = build_metadata_signature(metadata)
            approved_record = register_metadata_approval(metadata)
            if review_note.strip():
                approved_record["review_note"] = review_note.strip()
            record_audit_event("Package approved",
                f"{approved_record['package_id']} approved by {st.session_state.current_role}.",
                status="Approved")
            rerun_with_persist()

        change_disabled = not review_note.strip()
        if action_cols[1].button("Request changes", use_container_width=True, disabled=change_disabled):
            feedback_record = register_metadata_feedback(metadata, "Changes Requested", review_note)
            st.session_state.metadata_status = "Changes Requested"
            st.session_state.metadata_review_note = review_note.strip()
            st.session_state.metadata_reviewed_by = st.session_state.current_role
            st.session_state.metadata_reviewed_at = format_timestamp()
            st.session_state.metadata_approved_by = None
            st.session_state.metadata_approved_at = None
            record_audit_event("Package changes requested",
                f"{feedback_record['package_id'] if feedback_record else 'Package'} returned.",
                status="Returned")
            rerun_with_persist()

        if action_cols[2].button("Reject", use_container_width=True, disabled=change_disabled):
            feedback_record = register_metadata_feedback(metadata, "Rejected", review_note)
            st.session_state.metadata_status = "Rejected"
            st.session_state.metadata_review_note = review_note.strip()
            st.session_state.metadata_reviewed_by = st.session_state.current_role
            st.session_state.metadata_reviewed_at = format_timestamp()
            st.session_state.metadata_approved_by = None
            st.session_state.metadata_approved_at = None
            record_audit_event("Package rejected",
                f"{feedback_record['package_id'] if feedback_record else 'Package'} rejected.",
                status="Rejected")
            rerun_with_persist()

    return metadata, controls


def render_step_five(metadata: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    """Step 5 - Release Synthetic Package (approved download)."""
    step_cfg = STEP_CONFIG[4]
    st.markdown(
        f"""
        <div class="section-shell" style="margin-bottom:0.9rem;">
            <h3 style="margin:0;">{step_cfg["title"]}</h3>
            <div style="font-size:0.92rem;color:#668097;margin-top:0.3rem;line-height:1.55;">
                Download the approved synthetic package for controlled distribution. The release is recorded in the audit log.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_active_dataset():
        st.info("No dataset in workspace.")
        return

    if st.session_state.metadata_status != "Approved":
        st.warning("Release is gated - the package must be approved before download.")
        if st.button("Back to configuration"):
            st.session_state.current_step = 2
            st.rerun()
        return

    if st.session_state.synthetic_df is None:
        st.error("No synthetic data is available. Return to configuration and generate the package.")
        if st.button("Back to configuration"):
            st.session_state.current_step = 2
            st.rerun()
        return

    active_package = active_metadata_package_record(metadata)
    if st.session_state.results_shared_at:
        s_color, s_bg, s_label = "#136B48", "#EDF9F3", "Released"
    else:
        s_color, s_bg, s_label = "#0b5ea8", "#EBF1F7", "Ready to release"
    _render_status_strip_v2(4, s_label, s_color, s_bg)

    synthetic_df = st.session_state.synthetic_df
    summary = st.session_state.generation_summary or {}
    validation = st.session_state.validation or {}

    # Release summary
    with st.container(border=True, key="release_summary_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#136B48;margin-bottom:0.3rem;">Approved package</div>'
            f'<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.9rem;line-height:1.3;">{active_package["package_id"] if active_package else "-"} &middot; approved by {st.session_state.metadata_approved_by or "-"}</div>'
            '<div style="display:flex;gap:0.7rem;flex-wrap:wrap;">'
            + _stat_capsule_v2("Rows", str(len(synthetic_df)), "Synthetic records")
            + _stat_capsule_v2("Columns", str(len(synthetic_df.columns)), "Fields in output")
            + _stat_capsule_v2("Quality score", f"{validation.get('overall_score', 0):.1f}", "Combined fidelity + privacy")
            + _stat_capsule_v2("Privacy epsilon", str(summary.get('privacy_epsilon', 2.0)),
                           f"Preset: {summary.get('privacy_preset', 'Balanced')}")
            + '</div>',
            unsafe_allow_html=True,
        )

    # Download
    with st.container(border=True, key="download_panel"):
        st.markdown(
            '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#0b5ea8;margin-bottom:0.3rem;">Download</div>'
            '<div style="font-size:1.15rem;font-weight:600;color:#17324d;margin-bottom:0.8rem;line-height:1.3;">Export the synthetic package</div>',
            unsafe_allow_html=True,
        )

        csv_bytes = synthetic_df.to_csv(index=False).encode("utf-8")
        import json as _json
        metadata_json = _json.dumps({
            "package_id": active_package["package_id"] if active_package else "",
            "approved_by": st.session_state.metadata_approved_by,
            "approved_at": st.session_state.metadata_approved_at,
            "submitted_by": st.session_state.metadata_submitted_by,
            "submitted_at": st.session_state.metadata_submitted_at,
            "generation_summary": {k: v for k, v in summary.items() if not isinstance(v, list) or len(v) < 50},
            "quality_metrics": {
                "overall_score": validation.get("overall_score", 0),
                "fidelity_score": validation.get("fidelity_score", 0),
                "privacy_score": validation.get("privacy_score", 0),
                "correlation_score": validation.get("correlation_score", 0),
            },
            "metadata": metadata,
        }, indent=2, default=str).encode("utf-8")

        dl_cols = st.columns(2, gap="small")
        dl_cols[0].download_button(
            "Download synthetic CSV",
            data=csv_bytes,
            file_name=f"synthetic_{active_package['package_id'] if active_package else 'package'}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
        dl_cols[1].download_button(
            "Download package metadata (JSON)",
            data=metadata_json,
            file_name=f"metadata_{active_package['package_id'] if active_package else 'package'}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        if not st.session_state.results_shared_at:
            if st.button("Record as released", use_container_width=True):
                st.session_state.results_shared_at = format_timestamp()
                st.session_state.results_shared_by = st.session_state.current_role
                st.session_state.release_status = "Released"
                record_audit_event("Package released",
                    f"{active_package['package_id'] if active_package else 'Package'} released by {st.session_state.current_role}.",
                    status="Released")
                rerun_with_persist()
        else:
            st.markdown(
                f'<div style="padding:0.6rem 0.9rem;background:#EDF9F3;border:1px solid #B8E3CC;border-radius:10px;font-size:0.85rem;color:#136B48;">Released by {st.session_state.results_shared_by} at {st.session_state.results_shared_at}</div>',
                unsafe_allow_html=True,
            )

    # Back button
    nav = st.columns([1, 1], gap="small")
    if nav[0].button("Back to configuration", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="H", layout="wide", initial_sidebar_state="collapsed")
    inject_styles()
    initialize_app_state()
    process_pending_workspace_actions()

    if not st.session_state.authenticated:
        render_login_screen()
        return

    ensure_dataset_loaded()
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls
    sync_metadata_workflow_state(metadata)
    metadata = editor_frame_to_metadata(st.session_state.metadata_editor_df)
    controls = st.session_state.controls
    ensure_role_step_visibility(metadata, controls)

    render_header(metadata, controls)
    render_step_navigation(metadata, controls)

    current_step = st.session_state.current_step

    # Scroll to top when the user changes step. Streamlit reruns preserve
    # browser scroll, so a button clicked at the bottom of Step N would
    # otherwise leave the user staring at the bottom of Step N+1.
    # The aggressive version: for ~1.4s after a step change, keep scrolling
    # every 40ms. Streamlit paints the new content in pieces, so a single
    # scroll call races against partial layout — repeated calls survive it.
    last_rendered_step = st.session_state.get("_last_rendered_step")
    step_changed = last_rendered_step != current_step
    if step_changed:
        st.session_state._last_rendered_step = current_step
        # Marker changes each step change so the iframe key/content changes,
        # forcing Streamlit to treat this as a fresh component and run the script.
        scroll_marker = st.session_state.get("_scroll_marker", 0) + 1
        st.session_state._scroll_marker = scroll_marker
        st_components.html(
            f"""
            <!-- scroll-reset-{scroll_marker} -->
            <script>
                (function() {{
                    const doc = window.parent && window.parent.document ? window.parent.document : document;
                    const SELECTORS = [
                        'section.main',
                        'section[data-testid="stAppViewContainer"]',
                        'div[data-testid="stAppViewContainer"]',
                        'div[data-testid="stMainBlockContainer"]',
                        'div[data-testid="stMain"]',
                        'main',
                    ];
                    const scrollAll = () => {{
                        try {{ window.parent.scrollTo(0, 0); }} catch (e) {{}}
                        try {{ window.scrollTo(0, 0); }} catch (e) {{}}
                        try {{ doc.documentElement.scrollTop = 0; }} catch (e) {{}}
                        try {{ doc.body.scrollTop = 0; }} catch (e) {{}}
                        for (const sel of SELECTORS) {{
                            try {{
                                doc.querySelectorAll(sel).forEach(el => {{
                                    if (el && typeof el.scrollTo === 'function') {{
                                        el.scrollTo(0, 0);
                                    }}
                                    if (el) el.scrollTop = 0;
                                }});
                            }} catch (e) {{}}
                        }}
                    }};
                    // Hammer scroll for ~1.4s — enough to survive async Streamlit paint.
                    scrollAll();
                    let ticks = 0;
                    const iv = setInterval(() => {{
                        scrollAll();
                        if (++ticks >= 35) clearInterval(iv);
                    }}, 40);
                    // Also fire on DOMContentLoaded / load to catch iframe-ready edge cases.
                    window.addEventListener('load', scrollAll);
                }})();
            </script>
            """,
            height=0,
        )

    if current_step == 0:
        render_step_one(metadata)
    elif current_step == 1:
        render_step_two()
    elif current_step == 2:
        metadata, controls = render_step_three()
    elif current_step == 3:
        metadata, controls = render_step_four(metadata, controls)
    elif current_step == 4:
        render_step_five(metadata, controls)

    persist_shared_workspace_state()


if __name__ == "__main__":
    main()
