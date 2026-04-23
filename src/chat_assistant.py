"""Workflow Agent — Claude API guidance for the governed synthetic data workspace.

Provides orchestration-aware responses to workflow questions. Falls back to
structured local mode when ANTHROPIC_API_KEY is not configured.
"""

from __future__ import annotations
import os
from typing import Any

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Claude 4.6 Sonnet — current best Sonnet-class model as of 2026.
DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are the Agent Guidance layer inside Southlake Health's governed synthetic data workspace "
    "(Southlake is a 400-bed community hospital in Newmarket, Ontario; ~90K ED visits/yr; MEDITECH Expanse; PHIPA/PIPEDA). "
    "You help Data Analysts, Managers / Reviewers, and clinical / non-technical stakeholders understand and act on an already-generated synthetic package. "
    "\n\nCORE BOUNDARY: You only reason about the approved synthetic outputs, metadata, validation, hygiene findings, and release posture that appear in the context block. "
    "You have no access to raw source records, original identifiers, or row-level patient data and must never fabricate or imply such access. "
    "\n\nCAPABILITIES: You are allowed to do more than restate the package. You can recommend next steps, compare analytical options, suggest model families, identify likely risks, propose validation plans, explain tradeoffs, frame reviewer-ready summaries, and translate technical findings for non-technical stakeholders as long as your reasoning stays grounded in the governed synthetic-package context. "
    "If the user asks a broad strategic question, a role-based question, or an exploratory question, answer it directly with your best judgment. "
    "\n\nSCOPE CLARIFICATION: Broad workflow questions are in scope when they can be answered from the package evidence you do have. "
    "Questions like 'what should I do next', 'is this ready', 'what should I validate', 'what is this package good for', "
    "'what should a reviewer do', 'what are the main risks', 'what model should I use', 'how should I explain this', and 'what should I investigate next' should be answered directly using the available synthetic-package context. "
    "Questions phrased casually, like 'what am I gonna do next', 'what now', or 'where do I go from here', should be interpreted as requests for the best package-focused next action. "
    "When the user asks a broad next-step question, infer the most likely analytical or governance objective from the package evidence and provide the best next actions. "
    "Do not say that you lack visibility into the user's project plan, task queue, or broader roadmap unless the user explicitly asks for something that truly requires those details. "
    "When important details are missing, make the best reasonable assumption from the package and say 'assuming' rather than refusing. "
    "Default to a useful recommendation grounded in the synthetic-package context you do have. "
    "Only decline when the question truly depends on unavailable raw-data detail, private project plans, or facts outside the governed artifacts."
    "\n\nSTYLE: Answer in a compact enterprise analysis format. Use 1 short takeaway sentence, then 2 to 4 bullets, then 1 short next-step sentence unless the user explicitly asks for more. "
    "Use numbers from the context when relevant. Professional governance-memo tone. No 'Great question', no filler, no exclamation marks. "
    "Do not use markdown headings, horizontal rules, tables, or code fences. "
    "Position guidance as suitable for internal modeling sandbox use; never for direct patient care or clinical decision making. "
    "If asked something completely unrelated to the workspace (for example weather, sports scores, or general trivia), decline in one short sentence and then redirect once to a package-relevant question. "
    "If asked something outside scope because it would require real patient-level data, say so plainly and continue with the nearest governed alternative you can still help with."
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
    api_key: str,
    user_message: str,
    chat_history: list[dict[str, str]],
    context: str,
    model: str = DEFAULT_MODEL,
    role: str | None = None,
    max_tokens: int = 1200,
) -> str:
    """Route one chat turn through Claude with prompt caching.

    Falls back to a local deterministic reply if no API key is configured or
    the external request fails. Uses ephemeral cache_control on both the
    system prompt (static across sessions) and the context block (stable
    within a session) so subsequent turns benefit from cache hits.
    """
    key = (api_key or "").strip() or ANTHROPIC_API_KEY
    if not key:
        return _fallback(user_message)
    try:
        import anthropic
    except ImportError:
        return ("Anthropic SDK is not installed in this environment. "
                f"Falling back to local guidance. {_fallback(user_message)}")

    # Keep recent history only; cap to last 10 turns for latency.
    msgs: list[dict[str, Any]] = []
    for m in chat_history[-10:]:
        if m.get("content") and m.get("role") in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_message})

    # Prompt caching: static system prompt + per-session context both cached.
    role_hint = f"\n\nCurrent user role: {role}." if role else ""
    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"Synthetic package context (metadata + aggregates only):\n{context}{role_hint}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=msgs,
        )
        text_parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        response_text = "\n".join(p.strip() for p in text_parts if p).strip()
        return _normalize_reply_text(response_text or _fallback(user_message))
    except anthropic.AuthenticationError:
        return "Authentication failed. Check that the API key is valid and has access to the Claude API."
    except anthropic.RateLimitError:
        return "Rate limited by the external API. Retry in a moment, or continue in internal guidance mode."
    except anthropic.APIConnectionError:
        return (f"Could not reach the external API right now. {_fallback(user_message)}")
    except anthropic.APIStatusError as exc:
        return (f"External API error ({exc.status_code}). {_fallback(user_message)}")
    except Exception as exc:
        return (f"Unexpected error routing to external API ({type(exc).__name__}). {_fallback(user_message)}")


def generate_demo_chat_reply(
    user_message: str, profile: dict[str, Any], hygiene: dict[str, Any],
    controls: dict[str, Any], validation: dict[str, Any] | None,
) -> str:
    return _fallback(user_message, profile, hygiene, controls, validation)


def _normalize_reply_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("```"):
            continue
        if set(stripped) <= {"-", "_", "*"} and len(stripped) >= 3:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip(" :")
            stripped = f"**{stripped}**"
        lines.append(stripped)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _fallback(msg: str, profile=None, hygiene=None, controls=None, validation=None) -> str:
    m = msg.lower().strip()
    fp = (controls or {}).get("fidelity_priority", 50)
    rows = (profile or {}).get("summary", {}).get("rows", "N/A")
    overall_score = (validation or {}).get("overall_score", "pending")

    def structured_reply(takeaway: str, bullets: list[str], next_step: str) -> str:
        bullet_lines = "\n".join(f"- {item}" for item in bullets[:4])
        return f"{takeaway}\n\n{bullet_lines}\n\nNext step: {next_step}"

    if any(k in m for k in ["agent", "agentic"]):
        return structured_reply(
            "The agent layer sits on top of the approved synthetic package, not the raw source dataset.",
            [
                "Local mode stays fully inside the workspace for bounded analysis.",
                "Connected mode adds stronger reasoning through the API on synthetic artifacts only.",
                "Every workflow decision is recorded in the audit trail.",
            ],
            "Use Local for first-pass review and Connected when you need deeper analytical framing.",
        )
    if any(k in m for k in ["privacy", "fidelity", "slider"]):
        label = "higher privacy" if fp < 40 else "balanced" if fp < 70 else "higher fidelity"
        return structured_reply(
            f"Privacy-vs-fidelity is currently set to {label} ({fp}/100).",
            [
                "Lower values add more noise and strengthen privacy protection.",
                "Higher values preserve source-like patterns more aggressively.",
                "The right setting depends on whether you prioritize privacy margin or analytical realism.",
            ],
            "Validate quality and privacy together before release.",
        )
    if any(k in m for k in ["lineage", "metadata", "how"]):
        return structured_reply(
            "The package is produced through a governed metadata-driven workflow.",
            [
                "Source data is profiled into metadata and hygiene findings.",
                "Approved metadata drives synthetic generation.",
                "Validation compares the package against the source profile at an aggregate level.",
            ],
            "Review metadata and validation together before downstream use.",
        )
    if any(k in m for k in ["hygiene", "issue", "quality"]):
        if hygiene:
            return structured_reply(
                f"{hygiene['severity_counts']['High']} high and {hygiene['severity_counts']['Medium']} medium hygiene issue(s) are currently flagged.",
                [
                    "Corrections apply to metadata and controls, not to raw source records.",
                    f"Overall validation is {overall_score}.",
                    "Unresolved hygiene findings should be cleared before broader internal use.",
                ],
                "Review the highest-severity findings first.",
            )
        return structured_reply(
            "The hygiene scan has not run yet.",
            ["Upload data before quality and issue checks can be generated."],
            "Start with the upload step.",
        )
    if any(k in m for k in ["governance", "audit", "phipa"]):
        return structured_reply(
            "Governance remains in place across the full workflow.",
            [
                "Every major action is logged in the audit trail.",
                "Metadata requires reviewer approval before governed generation and release.",
                "Source data is not copied into the agent layer.",
            ],
            "Use the audit and approval state to support release decisions.",
        )
    if any(k in m for k in ["use", "analysis", "why"]):
        return structured_reply(
            "This synthetic package is best suited for internal modeling sandbox use, not clinical decision making.",
            [
                "Typical uses include workflow modeling, pipeline testing, and vendor sandboxing.",
                f"Current package size is {rows} rows with validation at {overall_score}.",
                "Use Connected mode when you need a stronger analytical narrative or exploration path.",
            ],
            "Match the package to the intended internal use before sharing further.",
        )
    return structured_reply(
        f"Workflow agent active on a package with {rows} rows.",
        [
            "Ask about package readiness, risks, validation posture, or likely next checks.",
            "Use Local mode for bounded review and Connected mode for deeper analysis.",
        ],
        "Frame the next question around a concrete decision you need to make.",
    )
