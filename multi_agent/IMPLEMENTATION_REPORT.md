# Implementation report

## 1. Added system

An isolated research-only subproject wraps the unchanged `minimal_v1` model with six typed workflow components, Deep Agents graph configuration, deterministic inference/evidence tools, a CLI, logs, scenarios, and tests.

## 2. Files

`pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `AUDIT.md`, `ARCHITECTURE.md`, `README.md`, `scenarios/scenarios.json`, `src/cvd_agents/{config,contracts,tools,workflow,deep,cli,scenario_runner}.py`, and `tests/test_system.py`.

## 3. Agents and tools

The six components are `orchestrator_agent`, `input_validation_agent`, `risk_prediction_agent`, `evidence_agent`, `explanation_agent`, and `safety_review_agent`. Domain tools are `validate_patient_input`, `predict_cvd_mortality_risk`, `get_model_evidence`, and `check_research_only_constraints`; artifact verification and schema loading are deterministic internal helpers. The optional `create_deep_agent` graph declares the same five specialists beneath the orchestrator.

## 4. Canonical model inference

The repository Stage 9 verifier checks Git provenance, manifest/document SHA, all four joblib SHA values, runtime versions, and synthetic smoke inference. The wrapper then validates the exact seven ordered columns, loads the trusted full pipeline, and calls `model.predict_proba(frame, calibrated=True)`. The full pipeline already includes the fitted sigmoid calibrator. Its 1e-6 clip is applied to the estimator probability before the calibrator logit; the separate calibrator is not called again and no extra output clip is added.

## 5. Contracts and state

Pydantic models define patient, validation, prediction, evidence, explanation, safety, and workflow state. Per-run sanitized state snapshots omit patient values, free-text request, probability, and response text.

## 6. Errors and limits

Missing/invalid input stops before prediction. Stage 9 artifact/SHA failures stop before deserialization. File, deserialization, prediction, and safety failures produce terminal error states. The deterministic workflow has a 12-stage limit and checks a 60-second elapsed limit between stages. Safety permits one revision. Optional Deep Agents calls have a 60-second async timeout, LangGraph recursion limit 12, and one retry for transient connection/timeout/server errors. Blocking deterministic tool calls do not yet have hard 20-second cancellation.

## 7. Logging and workload

`runs/<run_id>/events.jsonl` records agent/tool name, status, error type, and duration without health fields or secrets. `state.json` and `load_report.json` are sanitized. Counting one event per Python agent call and explicit domain tool call, the standard complete run had 10 calls, six agents, and a maximum agent share of **20%**.

## 8. Executed verification

Run from `multi_agent/` unless a different directory is shown:

| Command | Actual result |
|---|---|
| `uv sync --locked --group dev` | Installed isolated Python 3.12 environment, Deep Agents 0.7.21, and locked scientific versions |
| `uv run --locked pytest -q` | **11 passed**, 0 failed, 14 NumPy/joblib deprecation warnings |
| `uv run --locked python -m cvd_agents.scenario_runner` | **12/12 passed** |
| `uv run --locked python -m cvd_agents.cli health` | `status: ok`, four locked artifacts verified |
| `uv run --locked python -m cvd_agents.cli assess --age 58 --sex male --bmi 28.1 --smoking-status former --hypertension --no-diabetes --general-health good` | Completed real synthetic assessment; probability **0.8374%** |
| Piped a synthetic JSON record and `quit` into `uv run --locked python -m cvd_agents.cli chat` | Returned the same research-only assessment and exited normally |
| From repository root: `multi_agent\.venv\Scripts\python.exe scripts\34_validate_stage9_research_release.py` | `validation_status: PASS`; four artifacts verified; primary export gate remains `FAIL` |
| `git diff --exit-code origin/main -- uv.lock pyproject.toml artifacts/minimal_v1 artifacts/minimal_v1_research_release` | Exit 0; root lock and locked research assets unchanged |
| Same assessment with `--agent-mode` | Failed safely with `OpenAIError: Missing credentials`; no provider key configured |

## 9. Remaining limitations

- A real LLM-driven five-subagent delegation was not executed because no provider API key was available. Offline tests build and invoke the Deep Agents graph and mock the delegation-result check.
- Deep Agents injects framework filesystem/task tools. The five-tool cap is met for assigned domain tools, but not audited or enforced over all built-in framework tools.
- The optional Deep Agents output is not used as the final medical-risk answer; the safety-reviewed deterministic response remains authoritative.
- Hard per-tool cancellation for blocking Python calls is absent; the total deterministic timeout is checked only between stages.
- The original repository's entire test suite was not run; Stage 9 validation and the subproject suite were run.

The model remains research-only, with underpowered exploratory external validation and no Kazakhstan validation or clinical deployment approval.
