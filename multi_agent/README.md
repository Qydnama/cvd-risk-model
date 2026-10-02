# Multi-Agent AI System for Interpretable Five-Year Cardiovascular Mortality Risk Assessment

This system is for research and educational use only. It wraps the dissertation's locked `minimal_v1` pipeline. It does not diagnose disease, recommend treatment, support clinical deployment, or claim Kazakhstan validation.

## Relationship to the dissertation

The existing research repository owns the seven-field input contract, fitted preprocessing, logistic estimator, sigmoid calibrator, model card, and release governance. The `multi_agent/` project does not train or alter any of them. See [AUDIT.md](AUDIT.md) for the canonical inference path and [ARCHITECTURE.md](ARCHITECTURE.md) for the six roles, Mermaid diagram, tool sources, state, and the reason for separating the agents.

## Setup

From `multi_agent/` on Windows PowerShell:

```powershell
uv sync --locked --group dev
uv run --locked python -m cvd_agents.cli health
uv run --locked pytest -q
uv run --locked python -m cvd_agents.scenario_runner
```

The subproject lock fixes Python 3.12-compatible `joblib==1.5.3`, `numpy==2.5.1`, `pandas==2.3.3`, and `scikit-learn==1.9.0`. Root `uv.lock` is untouched. The local root package is installed editable.

For optional Deep Agents experiments, copy `.env.example` to `.env` as a reference, then set a LangChain-compatible `AGENT_MODEL` and provider API key in the process environment. The CLI does not automatically load `.env`. Use `--agent-mode` with `assess` to run Deep Agents delegation after the deterministic assessment. It requires all five specialist task calls and returns the deterministic safety-reviewed answer. Transient model connection/timeout/server failures are retried once. Do not commit `.env`. The deterministic commands do not need a provider key. The Deep Agents graph builds and runs with a test chat model; a real provider assessment has not been run here.

## CLI

```powershell
uv run --locked python -m cvd_agents.cli assess --age 58 --sex male --bmi 28.1 --smoking-status former --hypertension --no-diabetes --general-health good
uv run --locked python -m cvd_agents.cli assess --agent-mode --age 58 --sex male --bmi 28.1 --smoking-status former --hypertension --no-diabetes --general-health good
uv run --locked python -m cvd_agents.cli evidence
uv run --locked python -m cvd_agents.cli chat
```

`chat` accepts one seven-field JSON object per line. Missing fields cause validation errors; the trained imputer is not used to supply absent personal values. The outcome is a **continuous probability of death from cardiovascular causes within five years**. The secondary cutoff is not a clinical risk category. Output from the CLI is for local research demonstration only.

## Workflow and tools

The orchestrator calls input validation, deterministic prediction, evidence, explanation, and safety review in order. Each boundary uses a Pydantic contract. The prediction tool first runs the repository's Stage 9 SHA/provenance verifier, then calls the calibrated method on the exact ordered DataFrame. The evidence tool reads manifest-bound local sources and development metrics. Safety review checks the final statement against the verified numeric probability and research-only wording. There is at most one explanation revision.

Logs are in `runs/<run_id>/events.jsonl`; sanitized state and measured load are in `state.json` and `load_report.json`. Health inputs, full user request, individual probability, and API keys are not stored there. `runs/` is ignored by Git. Scenario definitions are in `scenarios/scenarios.json`; the runner writes `runs/scenario_summary.json`.

## Limitations

- Development metrics are from NHIS 1997–2009. Temporal and external checks were underpowered and exploratory.
- The model was not validated for Kazakhstan and is not approved for clinical, API, or web deployment.
- The Deep Agents graph is constructed and tested offline. `--agent-mode` has not had a live provider test or a full real-LLM delegation audit.
- Deep Agents adds framework tools; the five-tools-per-agent requirement is met for explicitly assigned domain tools only.
- A 60-second run limit is checked between deterministic stages and enforced around the optional Deep Agents invocation; blocking Python tools lack hard per-call cancellation.
