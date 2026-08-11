# Active Plan

## Objective

Maintain the active reliability fixes for AI research planning, Copilot tool-call
conversation history, and bounded tool-loop completion.

## Assumptions

- Round indices passed to planners are zero-based.
- A resumed goal must expose existing experiment and finding IDs as valid basis candidates.
- Existing user changes in `examples/chiller_power/report.md` are out of scope and must remain untouched.
- Every assistant tool-call message sent to Chat Completions must be followed by exactly one matching tool result per call ID.
- The 16-round tool safety bound remains in force; exhausting it must produce a
  normal, resumable answer instead of turning completed work into a UI error.
- Copilot experiment planning must advertise only executable model routes:
  data ridge/linear, physics cooling_balance_v1/v2, and hybrid xgboost residual.

## Tasks

- [x] Unify orchestrator and planner round-index semantics.
- [x] Load historical experiment/finding evidence for resumed goals.
- [x] Add regression coverage for first-run and resumed-goal planning.
- [x] Run focused and broader relevant tests.
- [x] Diagnose the Copilot orphaned `tool_calls` message sequence.
- [x] Preserve assistant/tool message groups across tool execution and history handling.
- [x] Add regression coverage for multiple tool calls and exceptional tool results.
- [x] Run focused Copilot tests and broader relevant verification.
- [x] Diagnose the 16-round non-convergence session from its agent log.
- [x] Expose the nested Experiment contract and executable estimator catalog in
  the `tf_experiment_plan` tool schema.
- [x] Replace the hard RuntimeError at the tool bound with a no-tools final
  synthesis and protocol-safe deterministic fallback.
- [x] Add Copilot guidance against probing unsupported model names.
- [x] Add and run bounded-loop/schema/Copilot regression coverage.

## Change Log

- 2026-08-11: Created plan for planner basis failure fix.
- 2026-08-11: Implemented zero-based planner indices while preserving
  one-based user-facing round numbers.
- 2026-08-11: Added goal-global basis evidence discovery and planner-side
  validation against real EXP/F candidate IDs.
- 2026-08-11: Verification completed: 66 focused tests passed; 506 broader
  tests passed with five pre-existing CRLF-vs-LF schema snapshot comparisons
  excluded. One unrelated Windows atomic-replace test flaked in the broad run
  and passed immediately when rerun alone. `git diff --check` passed.
- 2026-08-11: Added Copilot tool-call message pairing failure to the active plan.
- 2026-08-11: Fixed Copilot tool-call protocol recovery: assistant messages are
  reconstructed from normalized calls, all tool exceptions produce matching
  tool results, and interrupted histories are repaired before the next user
  turn. Session logs now retain tool call IDs and names.
- 2026-08-11: Copilot verification completed: 66 affected-layer tests passed;
  broader suite passed 511 tests with seven deselections (including five known
  CRLF-vs-LF schema snapshots) and one pre-existing subprocess UTF-8 warning.
- 2026-08-11: Diagnosed the reported 16-round failure from the latest session
  log. The Copilot was not repeating one identical call; it was spending rounds
  guessing the nested experiment contract and unsupported MLP/LightGBM/XGBoost
  estimator names after validation failures.
- 2026-08-11: Added an inlined Experiment tool schema with closed executable
  model enums, synchronized the Copilot model-capability instruction, and made
  tool-budget exhaustion request one final completion with tools disabled.
  Non-compliant finalizer tool calls receive matching `LIMIT_REACHED` results;
  finalization failures return a resumable assistant message rather than a
  RuntimeError.
- 2026-08-11: Verification completed: 39 focused Agent/Copilot tests passed;
  86 broader contract/tool/orchestrator/WebUI tests passed, and the two model
  publishing tests affected by the local Python 3.13 Windows `0700` temp-dir
  ACL issue passed when rerun with a test-process-only permission workaround.
  `git diff --check` passed. Ruff was unavailable in the project environment.
  The WebUI process was restarted to discard the old in-memory Copilot session;
  `http://127.0.0.1:8765/_stcore/health` returned HTTP 200 `ok`.
- 2026-08-11: Pre-commit verification reran all affected Agent, ledger,
  modelability, orchestrator, and WebUI suites: 105 tests passed.
