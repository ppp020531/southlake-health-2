# Southlake Health — Agentic Synthetic Data Workspace

A governed, metadata-driven workflow for synthetic healthcare data creation, review, and release readiness.

## What It Is

This is a governed workflow platform for producing synthetic healthcare data under explicit control. The workflow agent orchestrates every step from source ingestion to controlled release. Metadata-only transformation is the boundary: source records are never copied into synthetic output.

The product is designed for internal modeling sandbox use, vendor integration testing, analytics pipeline development, and workflow rehearsal. It is not for clinical decision making.

## Governed Workflow

The agent guides each request through six controlled stages:

1. **Upload Data** — Source CSV is profiled into a statistical metadata blueprint
2. **Scan Data** — Agent classifies hygiene findings as blockers, warnings, or informational
3. **Review Data Settings** — Analyst reviews extracted metadata and confirms handling actions
4. **Submit for Review** — Metadata package is submitted for governance approval
5. **Generate Synthetic Data** — Approved metadata drives synthetic record generation
6. **Download & Share Results** — Release readiness is verdict-based, with stakeholder interpretations

Each stage is audit-logged. Every decision has a reason code. The agent visibly controls when the workflow advances or holds.

## What Makes It Agentic

The workflow agent is not a side chatbot. It is the operating logic of the workspace.

- **Readiness Engine** — Computes blocked / review-ready / release-ready status from actual conditions
- **Decision Log** — Every action is reason-coded (HYG-01, META-03, GOV-01, FID-03, REL-01)
- **Metadata Lineage** — Visual chain from source to extracted metadata to adjusted metadata to synthetic output to verified release
- **Hygiene Classification** — Findings are classified as blockers, warnings, or informational and actually affect readiness
- **Release Verdicts** — Verdict-first summaries replace opaque percentage scores
- **Stakeholder Interpretation** — Concise readings for Operations Manager, Clinical Analyst, and Compliance Officer

## Governance Boundaries

- Metadata-only transformation (GOV-01): source records are never copied into synthetic output
- All field-level handling actions are reviewable before generation
- Governance sign-off required before synthesis is unlocked
- Audit trail preserved across every workflow state change
- Output suitable for internal modeling sandbox use only (REL-03: not for clinical decision making)

## Repo Structure

```
├── app.py                          Main Streamlit app with full governed workflow
├── requirements.txt
├── sample_data.csv                 28 synthetic ED records for demo
├── sample_data_large.csv           Extended demo dataset
├── JUDGE_METHOD_EXPLAINER.md       Algorithmic methodology for judge review
└── src
    ├── profiler.py                 Source data profiling and semantic role inference
    ├── hygiene_advisor.py          Hygiene scan and severity classification
    ├── cleaner.py                  Targeted hygiene correction actions
    ├── metadata_builder.py         Metadata blueprint extraction
    ├── generator.py                Synthetic data generation engine
    ├── validator.py                Fidelity and privacy validation
    ├── explainer.py                Release readiness briefing
    ├── chat_assistant.py           Workflow agent with Claude API integration
    └── agent_orchestrator.py       Agent readiness engine, decision log, lineage, verdicts
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

A demo dataset (`sample_data.csv`) is available to upload on the first step. The workspace supports two roles:

- **Data Analyst** — Uploads data, reviews scan findings, finalizes settings, submits requests, downloads results
- **Manager / Reviewer** — Approves or rejects submitted requests and reviews final output

Demo credential for both roles: `test`

## Workflow Agent Configuration

The workflow agent supports Claude API for richer conversational guidance. To enable:

```bash
export ANTHROPIC_API_KEY="your-key"
```

Without an API key, the agent runs in structured local mode with deterministic responses about workflow state, governance, and release readiness.

## Intended Use

Suitable for:
- Internal operational modeling sandbox
- Workflow rehearsal and training
- Analytics pipeline development
- Vendor sandbox and integration testing

Not suitable for:
- Clinical decision making
- Direct patient care applications
- Release outside of approved governance controls
