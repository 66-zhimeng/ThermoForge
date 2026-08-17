# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ThermoForge is an autonomous physics–data hybrid modeling research system for data-center HVAC. It takes an object model (TFOM), historical data (TFDC), and a research goal, then runs the loop `目标 → 假设 → 实验 → 证据 → 结论 → 知识 → 新假设` and delivers a deployable Model Package (signature + constraints + metrics + data/research lineage), not a bare `model.pkl`.

Docs and code comments are written in Chinese — match that when editing.

## Commands

`requires-python = ">=3.12"`, but the local `.venv` runs 3.14 — check `.venv/Scripts/python --version` before blaming a version-sensitive failure. Dependencies are managed with `uv`. On Windows the console entry point is `.venv\Scripts\tf` (Bash: `.venv/Scripts/tf`).

```bash
uv sync                                   # create .venv and install deps
.venv/Scripts/python -m pytest            # full suite (527 tests, 2 slow ones deselected)
.venv/Scripts/python -m pytest tests/test_importer.py::test_name -x   # single test
.venv/Scripts/python -m pytest -m slow    # slow integration tests (43MB real workbook import)
.venv/Scripts/python scripts/export_schemas.py       # regenerate contracts/*/schema.json
.venv/Scripts/python examples/chiller_power/run_demo.py   # end-to-end vertical slice (~1–2 min)
.venv/Scripts/tf status                   # human-readable panel (--json for machines)
.venv/Scripts/tf agent --check            # validate agent config + endpoint reachability
.venv/Scripts/tf agent                    # conversational REPL (/tools, /exit)
.venv/Scripts/python tools/webui.py       # Streamlit console on http://127.0.0.1:8765
.venv/Scripts/python -m thermoforge_mcp   # MCP server over stdio (for Claude Code et al.)
```

`pyproject.toml` pins `--basetemp=.pytest_basetemp` (the system pytest temp dir has restricted ACLs on this machine) and `-m "not slow"`.

Non-CLI entry points: `启动.bat` → `tools/menu.py` (numeric menu over the envelope tools, single-object + linear-baseline path only), `启动网页版.bat` → the Streamlit console. These exist because the target user double-clicks — that constraint is load-bearing on dependency choices (see the `pyproject.toml` comment on why plotly static export goes through matplotlib instead of kaleido: kaleido 0.2.1 hangs forever on this machine, 1.x wants a downloaded Chrome).

There is no configured formatter/linter in the repo; `docs/implementation-notes.md` §12 recommends ruff + mypy.

## Layer map

| Package | Role |
|---|---|
| `thermoforge_core` | Contracts (Pydantic v2) + cross-cutting primitives: canonical JSON, fingerprints, sequential IDs, naming regexes, units, time, error registry |
| `thermoforge_data` | TFDC-XLSX importer/validator, immutable Data Vault (Parquet + DuckDB index), Dataset Views, legacy-workbook adapter, preprocessing rule library |
| `thermoforge_models` | Model implementations: linear baseline, chiller physics (Q·COP), residual hybrid (physics + XGBoost), feature scaler |
| `thermoforge_research` | Research kernel: Ledger, Experiment Runner (subprocess-isolated), splits/leakage, metrics, physics checks, modelability report, whitelist enforcement, orchestrator, and the 23-tool registry + envelope |
| `thermoforge_runtime` | Model Package build/verify, model registry + release gates, deployment binding, online inference |
| `thermoforge_cli` | `tf` CLI — one subcommand per tool, envelope JSON on stdout |
| `thermoforge_agent` | Built-in conversational agent (OpenAI-compatible chat completions + function calling) over the same tool registry |
| `thermoforge_webui` | Streamlit console (`tools/webui.py` launches it): AI copilot (asks → analyses → navigates), data browser, quality diagnosis→prescription, AI research loop with live progress, experiment result charts, model registry, report export (HTML/Markdown/PDF) |
| `thermoforge_mcp` | MCP server (`python -m thermoforge_mcp`, stdio) exposing the same tool registry to external agents (Claude Code etc.) — 24 tools = 23 registry − `tf_preprocess_approve` (`EXCLUDED_TOOLS`, approval requires `actor=human`) + `tf_status` / `tf_experiment_report` (`EXTRA_TOOLS`, MCP-only convenience wrappers) |

Runtime artifact roots (gitignored): `vault/` (data), `research/` (ledger, experiments, agent sessions), `models/` (registry). They are configurable via `tf --vault-root/--research-root/--models-root` and `ToolContext`.

`thermoforge_webui` is the largest package and splits four ways: `screens/` (one Streamlit page each), `services/` (page-free logic — this is what `tests/test_webui_*.py` targets), `charts/` (`series.py` prepares plot data once; `interactive.py`/`static.py` are the two renderers), `export/` (HTML/Markdown/PDF).

## Agent control plane (`pi/`)

`pi/` declares *how an agent calls the deterministic tools* and holds no research logic (`architecture.md` §4/§8 — the harness stays light and replaceable; research facts live in contracts and the Ledger).

- `pi/tools.json` — declarative tool name → Python entry manifest, kept in lockstep with `TOOL_REGISTRY` by `tests/test_tools.py`.
- `pi/prompts/{system,planner,copilot}.md` — the built-in agent's, research planner's, and Web copilot's prompts. Behavior changes usually belong here, not in Python.
- `pi/agent.toml` (gitignored; template `pi/agent.example.toml`) — OpenAI-compatible endpoint config. Precedence: CLI args > env (`TF_AGENT_API_KEY` / `TF_AGENT_BASE_URL` / `TF_AGENT_MODEL`) > TOML.
- Tool schemas are generated from `TOOL_REGISTRY`, so a model never sees a hand-written schema. Human-only tools are withheld and reached via the `tf_human_approval` round-trip.
- Session logs land in `research/agent_sessions/*.jsonl` — the first place to look when the agent misbehaves.
- One user turn is bounded at `config.max_tool_rounds = 16`; exhausting the bound must degrade into a normal resumable answer, not a UI error. Every assistant message carrying `tool_calls` must be followed by exactly one tool result per call ID — Chat Completions rejects orphaned groups, and preserving those message groups across history trimming is why `agent.py` normalizes history rather than filtering it (commit `ae52b97`).

## Architectural invariants

These are enforced by code and tests, not just documented. Breaking them breaks the suite.

**Agent never touches raw data.** All agent-facing capability goes through `thermoforge_research.tools.TOOL_REGISTRY` (23 tools), each returning the envelope from `envelope.py`: `ok/tool/id/status/inputs/summary/diagnostics/artifacts/truncated`. Hard limits: 32 KB response body (overflow spills to an artifact and sets `truncated`), `tf_dataset_sample` ≤ 200 rows, fixed profile quantile points. Known errors (`VaultError` / `ResearchError` / `ModelRegistryError` / contract validation) are converted to `ok=False` envelopes, never raised. Side-effecting tools must return a stable ID (`dataset@rev_NNNN`, `RG-`, `H-`, `EXP-`, `VIEW-`, `model@version`).

**CLI exit codes are not success signals.** `exit 0` means the command ran — including tool-level failure (`ok=false` in the envelope). `exit 2` means CLI-level error (bad args, malformed JSON). Callers must read `ok`.

**Immutability + atomic writes.** Vault revisions and completed experiments are never overwritten: content change ⇒ new revision (`TFV-701`, `TFX-904`). Every artifact write is temp file + `os.replace`. Sequential IDs come from `core.ids.IdAllocator` (counter file + exclusive lock), never reused or recycled. Single-writer model throughout; DuckDB is single-writer/multi-reader.

**Two fingerprints, one canonical JSON.** `source_sha256` (raw bytes, provenance) vs `content_sha256` (normalized logical content, dedup) — a re-saved Excel changes the former but not the latter. `canonical_json` in `core/canonical.py` is the single implementation of `docs/conventions.md` §5.1 (key sort, null-key removal, excluded non-semantic keys, set-semantic array sorting). `view_hash = sha256(dataset_revision_id + "\n" + canonical_json(view))` — the revision must be in the hash or caches collide across data versions.

**Candidate-input whitelist is a hard gate (DD-16).** `whitelist.py` enforces it twice: goal-level (entries must exist in the revision and must not be `source_kind=derived`) and experiment-level (view features ⊆ whitelist, view target == goal target). This exists because the surveyed data has `load = current_percent × 9672 / 100` — using it to predict power is circular reasoning that yields a fake MAPE of 4.35%. `modelability.py` is the semantic-layer gate (derivation chains, same-origin correlation, device diversity, physical plausibility, operating coverage); FAIL blocks entry to modeling.

**Metrics have exactly one implementation.** `research/metrics.py` (ratio-scale, not percent). Modeling code must not compute RMSE/MAPE/CVRMSE itself. MAPE drops `|y| < y_floor` and reports the valid fraction (`TFX-905` below 0.8); CVRMSE/NMBE are `None` when `mean(y) ≈ 0` rather than a plausible-looking number. NaN in metric input is a defect and raises.

**Reproducibility is machine-checked.** Experiments run in a subprocess with thread-count env vars (`OMP_/OPENBLAS_/MKL_NUM_THREADS`, `PYTHONHASHSEED`) set *before* numpy/sklearn import. Missing seed ⇒ `TFX-902`; `environment_lock` mismatch ⇒ `TFX-901`; same machine + same lock must reproduce metrics bit-exact.

**Splits are by time, not by row count.** Boundaries are floored to a resolution multiple, recorded as timestamps in the experiment artifact; purge at train tail and embargo (default 45 min) at validation head; overlap or insufficient gap ⇒ `TFX-903`.

**No pickle in model packages.** Artifacts serialize as JSON/YAML/XGBoost native format. Every package carries `checksums.json` (verified by `verify_package`, `TFM-1002`) and a `golden.parquet` prediction set replayed by the cold-load smoke test in a fresh interpreter (`TFM-1005`). Release gates and the version state machine (`candidate → validated → approved → production → deprecated → retired`, rollback being the only backward edge) live in `runtime/registry.py`.

**Naming rules matter on disk.** `object_id` must not contain `.` (else `variable_id` cannot be split), `variable_id` is case-sensitive — so it must never be used directly as a file or directory name on Windows/macOS. Split on the *first* dot.

**Preprocessing is proposal + approval, never generated code.** The agent may only propose parameters for pre-registered deterministic transforms (`data/preprocess.py`); `status=proposed` rulesets cannot produce a vault revision, and approval requires `actor=human` (`tf --actor human preprocess approve`, or the agent's `tf_human_approval` round-trip).

## Docs are a tested source of truth

`docs/` is not just prose — several tests parse it:

- `tests/test_errors.py` parses the error-code tables in `docs/conventions.md` §7 and compares them row-by-row with `core.errors.ERROR_REGISTRY`. Adding an error code means editing both, and `docs/conventions.md` is treated as frozen (see `docs/issues.md` I-53 — that is why preprocessing uses its own `TFPP-` registry instead).
- `tests/test_contracts.py` re-renders the five Pydantic contracts and asserts `contracts/*/schema.json` is fresh — rerun `scripts/export_schemas.py` after changing a contract model.
- `tests/test_tools.py` asserts `pi/tools.json` matches `TOOL_REGISTRY` — adding a tool means updating both, plus a `tf` subcommand in `thermoforge_cli/main.py`.
- `tests/fixtures/` holds one minimal xlsx per import error code; `tests/fixtures/generate.py` regenerates them.

Module docstrings cite the doc section they implement (`data-contract.md §7`, `implementation-notes.md §4`, `DD-12`, `TFDC-502`, …). When changing behavior, follow the citation to the spec first; if the code must diverge, record it in `docs/issues.md` rather than silently editing a frozen doc.

Doc layers: `docs/scope.md` / `design-decisions.md` / `risks.md` / `open-questions.md` / `issues.md` (proposal level) → `architecture.md` / `data-contract.md` / `research-loop.md` / `model-package.md` / `roadmap.md` (design) → `conventions.md` / `implementation-notes.md` / `glossary.md` (implementation). `docs/getting-started.md` is the user-facing tutorial.

## Windows notes

Development happens on Windows; these bite here and not on Linux (`implementation-notes.md` §10.3):

- All text I/O must be explicit `encoding="utf-8", newline="\n"` — default `\r\n` and non-UTF-8 encodings break cross-platform hash stability. `cli/main.py` reconfigures stdout/stderr to UTF-8 for the same reason.
- Locking is platform-split (`msvcrt.locking` vs `fcntl.flock`) in `core/ids.py` and `research/ledger.py`.
- Close DuckDB connections and file handles promptly, or `os.replace` fails on locked files.
- Keep vault roots on shallow paths (260-char limit).
- HTTP clients pointed at loopback must set `trust_env=False`. `httpx` takes proxies from `urllib.request.getproxies()`, which reads the registry system proxy but ignores the registry's `ProxyOverride` bypass list — so `127.0.0.1` requests get handed to a local proxy that refuses to forward to loopback. `agent/client.py` special-cases loopback/localhost `base_url` for this reason; it previously made `tests/test_agent.py::test_agent_check_ok` fail with a bogus 502.

## Current state

Phase 0–4 are implemented and validated end-to-end by `examples/chiller_power/`; Phase 5 (containers, queues, multi-site, multi-agent) is not started. `plan/active-plan.md` tracks the current work item — read it before starting, it states the assumptions in force.

Two P0 data questions were answered on 2026-08-09 (`docs/open-questions.md` Q10, `docs/issues.md` I-48) and the answers shape modeling: the workbook's driver series are field-measured but ~2.96M cells are Excel-derived, hence the mandatory `source_kind` measured/derived split; and the COP 9–11 range is physically valid for this high-temperature centrifugal site, which un-blocked the physics route. The I-01/I-02 rows in the `docs/issues.md` priority table still carry the pre-answer wording — trust `open-questions.md` / the I-48 detail section over them.

The full suite is green on this machine.
