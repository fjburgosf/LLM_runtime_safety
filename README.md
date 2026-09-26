# Scalar risk memory in tool-using LLM agents

This repository contains the execution and analysis code for the paired comparison of a stateless runtime supervisor, a scalar-memory supervisor, and an unsupervised reference arm. The study uses AgentDojo benchmark v1.2.2 and a pinned local extension of four injection tasks. The fixed case manifest and the configurations needed by the scripts are included. No model credentials or raw response logs are stored here.

## Environment

Python 3.12 is recommended. The evaluated AgentDojo snapshot is commit `a75aba7631d3ca5fb7ab938965c97ead2f9ff84b` of `https://github.com/ethz-spylab/agentdojo` (package version 0.1.35). From the repository root in PowerShell:

```powershell
git clone https://github.com/ethz-spylab/agentdojo.git benchmarks/agentdojo
git -C benchmarks/agentdojo checkout a75aba7631d3ca5fb7ab938965c97ead2f9ff84b
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The local extension lives in `paper1_extension/`. The controller, risk sensor, supervised tool executor, and model adapter live in `scripts/`. The JSON files under `configs/` and `results/exploratory/` are fixed execution inputs, not generated manuscripts.

## Check the planned jobs without model calls

```powershell
.\.venv\Scripts\python.exe scripts\59_run_direct_memory_retained.py --stage retained_benign --batch-index 0
.\.venv\Scripts\python.exe scripts\59_run_direct_memory_retained.py --stage retained_attack --batch-index 0
```

Omitting `--execute` prints the selected batch and performs no API request. The checked-in execution configuration is deliberately disabled. A dry run should report 10 cases and 30 jobs for benign batch 0.

## Execute the simulations

The runtime model is `ministral-14b-2512` through Mistral's API. Running the full experiment can consume provider allowance or incur charges. Before adding `--execute`, the researcher must review current provider billing settings and the remaining allowance, set `MISTRAL_API_KEY` in the local process environment, and copy `configs/direct_memory_retained_v2.json` to `configs/direct_memory_retained_v2.local.json`. Update only the authorization and billing fields in that ignored local copy for the specific batch. Do not commit the populated configuration or a key. Keep `max_out_of_pocket_usd` at zero. The local credit ledger is an estimate, not a provider-side billing limit.

The runner requires `status` to be `AUTHORIZED_FOR_SPECIFIC_RETAINED_BATCHES`, `approved_external_calls` to be true, and `authorized_batches` to contain the selected stage and batch index. It also requires nonempty authorization evidence, a future UTC expiry, fresh billing verification showing pay-as-you-go disabled, and a positive ceiling that does not exceed the verified remaining included credit. The scientific design, case manifest, extension hashes, model ID, and controller thresholds are frozen and checked before execution.

Run all `retained_benign` batches before any `retained_attack` batch. One command executes one batch:

```powershell
.\.venv\Scripts\python.exe scripts\59_run_direct_memory_retained.py --stage retained_benign --batch-index 0 --execute
```

The runner writes per-trajectory logs under `logs/paper1/direct_memory_retained_v2/attempt_001/`. It refuses malformed or incomplete existing logs rather than silently replacing them. Each retained case appears in all three arms. The model receives the same requested seed across arms, but deterministic provider replay is not assumed.

## Audit and analyze completed runs

After all 723 retained trajectories are complete, run these scripts from the repository root in order:

```powershell
.\.venv\Scripts\python.exe scripts\60_audit_direct_memory_retained.py
.\.venv\Scripts\python.exe scripts\61_analyze_direct_memory_retained.py
.\.venv\Scripts\python.exe scripts\62_audit_direct_memory_attack_equivalence.py
.\.venv\Scripts\python.exe scripts\63_audit_retained_mechanism.py
.\.venv\Scripts\python.exe scripts\66_audit_control_region.py
.\.venv\Scripts\python.exe scripts\71_audit_memory_opportunity.py
```

The primary analysis is a suite-equal-weighted, paired comparison of attack-goal success between the two supervised arms. The trace and control-region audits are descriptive mechanism checks. New model realizations need not reproduce the original outcome counts exactly. Preserve the resulting logs and configuration privately for auditing and review before releasing any trajectory content.
