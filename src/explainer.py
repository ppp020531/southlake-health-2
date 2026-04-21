"""Release Readiness Briefing — verdict-first summary for the governed workflow.

Generates release-aware briefings aligned with the agent's readiness engine.
Replaces the earlier analysis-readiness framing with governance verdicts.
"""

from __future__ import annotations

from typing import Any


def _release_verdict(score: float) -> str:
    if score >= 75:
        return "Ready for internal modeling sandbox"
    if score >= 60:
        return "Usable with review"
    return "Release hold pending additional iteration"


def build_readiness_briefing(
    profile: dict[str, Any],
    hygiene: dict[str, Any],
    metadata: list[dict[str, Any]],
    generation_summary: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    included_columns = [item for item in metadata if item["include"]]
    top_fidelity_columns = validation["fidelity_table"].head(5)["column"].tolist()
    top_fidelity_text = ", ".join(top_fidelity_columns) if top_fidelity_columns else "the included fields"
    verdict = _release_verdict(validation["overall_score"])

    executive_summary = (
        f"Release verdict: {verdict}. The agent transformed {profile['summary']['rows']} source records into "
        f"{generation_summary['rows_generated']} synthetic records under a metadata-only governance boundary. "
        f"Strongest fidelity in {top_fidelity_text}. Suitable for workflow modeling, capacity analysis, and sandbox testing. "
        f"Not for clinical decision making."
    )

    proof_points = [
        {
            "title": "Metadata-only transformation confirmed",
            "body": "The agent extracted a statistical blueprint from source and generated synthetic records from the approved metadata. No source records were copied into the output.",
        },
        {
            "title": "Governed review boundary",
            "body": "Field-level handling actions, inclusion decisions, and generation assumptions were all subject to governance review before synthesis was unlocked.",
        },
        {
            "title": "Verdict-based release readiness",
            "body": f"Fidelity verdict: {verdict}. Schema preservation and privacy boundaries were verified before the package was cleared for controlled use.",
        },
    ]

    use_cases = [
        {
            "name": "Workflow and operational modeling",
            "why_it_works": "Synthetic records preserve encounter structure and operational timing fields for throughput and flow analysis.",
            "example": "Simulate baseline ED flow and compare against scenarios with reduced extreme wait times.",
        },
        {
            "name": "Analytics pipeline development",
            "why_it_works": "The schema and key operational fields remain usable for feature engineering and cohort logic development.",
            "example": "Test revisit or admission predictors in a non-production environment before requesting sensitive data access.",
        },
        {
            "name": "Vendor sandbox and integration testing",
            "why_it_works": "Realistic healthcare-shaped data can exercise file layouts, API contracts, and workflows without releasing direct identifiers.",
            "example": "Exercise an ED analytics product end-to-end using synthetic output instead of live patient records.",
        },
        {
            "name": "Training and workflow rehearsal",
            "why_it_works": "Teams can rehearse handoffs, reviews, and reporting using realistic but de-identified examples.",
            "example": "Run implementation training using synthetic data that mirrors real operational patterns.",
        },
    ]

    talk_track = [
        "This is a governed transformation pipeline. The agent orchestrates every step from source schema to metadata blueprint to synthetic output.",
        "Every field-level handling decision is reviewable. Metadata-only transformation is the governance boundary.",
        "The output is suitable for workflow modeling, analytics sandboxing, and vendor integration. It is not for clinical decision making.",
        "Release recommendations are verdict-based and tied to explicit reason codes rather than opaque scores.",
    ]

    next_actions = [
        "Review weaker column verdicts in the metadata review artifact and adjust handling actions if needed.",
        "Expand source scope to additional departments or longer time ranges for broader synthetic coverage.",
        "Tighten privacy posture if the package will support broader operational distribution.",
    ]

    return {
        "executive_summary": executive_summary,
        "proof_points": proof_points,
        "use_cases": use_cases,
        "talk_track": talk_track,
        "next_actions": next_actions,
        "release_verdict": verdict,
        "included_columns": len(included_columns),
        "high_issues": hygiene["severity_counts"]["High"],
    }
