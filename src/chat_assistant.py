"""Workflow Agent — Claude API guidance for the governed synthetic data workspace.

Provides orchestration-aware responses to workflow questions. Falls back to
structured local mode when ANTHROPIC_API_KEY is not configured.
"""

from __future__ import annotations
import os
from typing import Any

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are the workflow agent inside a governed healthcare synthetic data workspace for Southlake Health (Newmarket, Ontario). "
    "Help users understand the agentic workflow, metadata transformations, governance controls, and release readiness. "
    "RULES: No em dashes. No 'Great question' or 'Happy to help'. 2-3 sentences max. Use numbers from context. "
    "Write like an internal governance memo. No exclamation marks. Position output as suitable for internal modeling sandbox use, not clinical decisions. "
    "Southlake: 400-bed hospital, 90K ED visits/yr, MEDITECH Expanse, PHIPA/PIPEDA."
)


def build_chat_context(
    source_label: str, profile: dict[str, Any], hygiene: dict[str, Any],
    metadata: list[dict[str, Any]], controls: dict[str, Any],
    generation_summary: dict[str, Any] | None, validation: dict[str, Any] | None,
) -> str:
    included = [m for m in metadata if m["include"]]
    col_summ = [f"{m['column']} ({m['data_type']}, {m.get('control_action','Preserve')})" for m in included[:12]]
    hyg_summ = [f"{i['severity']}: {i['column']}: {i['finding']}" for i in hygiene["issues"][:5]]
    lines = [
        f"Dataset: {source_label}",
        f"Shape: {profile['summary']['rows']} rows, {profile['summary']['columns']} cols, {profile['summary']['identifier_columns']} identifiers",
        f"Hygiene: score={hygiene['quality_score']}, high={hygiene['severity_counts']['High']}, med={hygiene['severity_counts']['Medium']}",
        f"Controls: fidelity={controls['fidelity_priority']}, rows={controls['synthetic_rows']}",
        "Columns: " + "; ".join(col_summ),
    ]
    if hyg_summ:
        lines.append("Issues: " + "; ".join(hyg_summ))
    if generation_summary:
        lines.append(f"Generated: {generation_summary['rows_generated']} rows, mode={generation_summary['noise_mode']}")
    if validation:
        lines.append(f"Validation: overall={validation['overall_score']}, fidelity={validation['fidelity_score']}, privacy={validation['privacy_score']}")
    return "\n".join(lines)


def generate_chat_reply(
    api_key: str, user_message: str, chat_history: list[dict[str, str]],
    context: str, model: str = DEFAULT_MODEL,
) -> str:
    key = api_key or ANTHROPIC_API_KEY
    if not key:
        return _fallback(user_message)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in chat_history[-6:] if m.get("content")]
        msgs.append({"role": "user", "content": user_message})
        resp = client.messages.create(model=model, max_tokens=600, system=f"{SYSTEM_PROMPT}\n\nContext:\n{context}", messages=msgs)
        return resp.content[0].text.strip()
    except Exception as e:
        return f"API connection issue. {_fallback(user_message)}"


def generate_demo_chat_reply(
    user_message: str, profile: dict[str, Any], hygiene: dict[str, Any],
    controls: dict[str, Any], validation: dict[str, Any] | None,
) -> str:
    return _fallback(user_message, profile, hygiene, controls, validation)


def _fallback(msg: str, profile=None, hygiene=None, controls=None, validation=None) -> str:
    m = msg.lower().strip()
    fp = (controls or {}).get("fidelity_priority", 50)
    rows = (profile or {}).get("summary", {}).get("rows", "N/A")

    if any(k in m for k in ["agent", "agentic"]):
        return "The agent orchestrates ingestion, metadata extraction, hygiene scanning, governed review, generation, and verification. Every decision is logged."
    if any(k in m for k in ["privacy", "fidelity", "slider"]):
        label = "higher privacy" if fp < 40 else "balanced" if fp < 70 else "higher fidelity"
        return f"Privacy-vs-fidelity is set to {label} ({fp}/100). Lower values add more noise. Higher values preserve source patterns."
    if any(k in m for k in ["lineage", "metadata", "how"]):
        return "Source data is profiled into metadata, hygiene corrections adjust it, approved metadata drives generation, fidelity verification compares output to source."
    if any(k in m for k in ["hygiene", "issue", "quality"]):
        if hygiene:
            return f"{hygiene['severity_counts']['High']} high and {hygiene['severity_counts']['Medium']} medium issues detected. Corrections apply to metadata, not source records."
        return "Upload data to run the hygiene scan."
    if any(k in m for k in ["governance", "audit", "phipa"]):
        return "Every action is logged. Metadata requires reviewer approval before generation. Source data is never copied. Only statistical metadata drives synthesis."
    if any(k in m for k in ["use", "analysis", "why"]):
        sc = validation["overall_score"] if validation else "pending"
        return f"Synthetic package supports workflow modeling, flow analysis, vendor sandboxing, and pipeline testing within an internal modeling sandbox. Not for clinical decision making."
    return f"Workflow agent active. Dataset has {rows} rows. Ask about the agentic workflow, metadata lineage, governance boundaries, or release readiness."
