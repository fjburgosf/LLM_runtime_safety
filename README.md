# Runtime safety evaluation: execution and analysis code

This repository contains only these instructions and the Python scripts needed to execute and analyze the paired runtime-supervision evaluation. It does not contain the manuscript, figures, figure generators, tables for publication, editorial forms, credentials, raw model responses, or result files.

The fixed benchmark inputs and configurations are embedded unchanged in `scripts/frozen_inputs.py`. The compositional task extension is `scripts/compositional_tasks.py`. The remaining scripts implement the controller, model adapter, execution, and audits. The embedded inputs preserve the same SHA-256 digests as the original JSON files. They are read in memory and do not produce paper text or figures.

## Environment

Python 3.12 is recommended. Clone the evaluated AgentDojo snapshot, then install its dependencies and the pinned API client. From the repository root in PowerShell:

```powershell
git clone https://github.com/ethz-spylab/agentdojo.git ../agentdojo
git -C ../agentdojo checkout a75aba7631d3ca5fb7ab938965c97ead2f9ff84b
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ../agentdojo openai==3.14.0
```

The evaluated benchmark is AgentDojo v1.2.2 in package version 0.1.35. Running the model requires a separate Mistral API account and may consume allowance or incur charges. No credentials or authorization are included here.

## Inspect the fixed workload without model calls

```powershell
.\.venv\Scripts\python.exe scripts\59_run_direct_memory_retained.py --stage retained_benign --batch-index 0
.\.venv\Scripts\python.exe scripts\59_run_direct_memory_retained.py --stage retained_attack --batch-index 0
```

Without `--execute`, the runner prints the selected work and makes no model request. The embedded default configuration denies external calls. The retained workload contains three matched arms for each case, using `ministral-14b-2512` and the pinned controller settings.

## Execute authorized simulations

Before executing, confirm current provider billing settings and remaining included allowance. Keep pay-as-you-go disabled and the maximum out-of-pocket amount at zero. Set `MISTRAL_API_KEY` only in the local process environment. Create a private `configs/direct_memory_retained_v2.local.json` from the embedded default configuration:

```powershell
$env:PYTHONPATH = 'scripts'
.\.venv\Scripts\python.exe -c "from pathlib import Path; import json; from frozen_inputs import read_fixed; p=Path('configs/direct_memory_retained_v2.local.json'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(read_fixed(Path('direct_memory_retained_v2.json')), indent=2)+'\n', encoding='utf-8')"
```

Set its authorization and billing fields for a specific batch. Never commit this local file or a key. The runner rejects calls without valid, recent authorization evidence and a batch-specific allowance ceiling. Its local credit ledger is an estimate, not a provider-side billing limit.

Run one authorized batch at a time. Complete all `retained_benign` batches before `retained_attack` batches:

```powershell
.\.venv\Scripts\python.exe scripts\59_run_direct_memory_retained.py --stage retained_benign --batch-index 0 --execute
```

The runner writes private trajectory logs under `logs/paper1/direct_memory_retained_v2/attempt_001/`. It rejects malformed or incomplete existing logs instead of silently replacing them. Requested seeds are matched across arms, but deterministic provider replay is not assumed.

## Analyze complete logs

Once all 723 retained trajectories are complete and available locally, run:

```powershell
.\.venv\Scripts\python.exe scripts\60_audit_direct_memory_retained.py
.\.venv\Scripts\python.exe scripts\61_analyze_direct_memory_retained.py
.\.venv\Scripts\python.exe scripts\62_audit_direct_memory_attack_equivalence.py
.\.venv\Scripts\python.exe scripts\63_audit_retained_mechanism.py
.\.venv\Scripts\python.exe scripts\66_audit_control_region.py
.\.venv\Scripts\python.exe scripts\71_audit_memory_opportunity.py
```

These scripts write local JSON audit and analysis outputs under `results/`; they do not create figures or manuscript content. The primary analysis is a suite-equal-weighted paired comparison of attacker-goal success between the two supervised arms. Trace and control-region audits are descriptive mechanism checks. New model realizations need not reproduce earlier outcome counts exactly. Preserve private logs and the local configuration for audit.
