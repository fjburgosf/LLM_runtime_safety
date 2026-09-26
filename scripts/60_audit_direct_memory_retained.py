"""Audit retained-log completeness and cost without revealing partial outcomes.

This independent script makes no API requests. Outcome analysis is deliberately
deferred until the entire frozen retained test is complete and validated.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "direct_memory_retained_v2.json"
DESIGN = ROOT / "configs" / "experiment_design_v2.json"
PARTITION = ROOT / "results" / "exploratory" / "direct_memory_partition_v2.json"
LOG_ROOT = ROOT / "logs" / "paper1" / "direct_memory_retained_v2" / "attempt_001"
OUTPUT = ROOT / "results" / "primary" / "direct_memory_retained_audit_v2.json"
STAGES = ("retained_benign", "retained_attack")
ARMS = ("no_defense", "stateful_rho_v1", "stateless_rho_v1")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(job: dict[str, Any]) -> tuple[str, str, str]:
    return (job["suite"], job["user_task_id"], job.get("injection_task_id", "none"))


def seed_for(salt: str, job: dict[str, Any]) -> int:
    key = "|".join((salt, *case_key(job)))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF


def trajectory_id(job: dict[str, Any]) -> str:
    return "/".join((job["stage"], job["arm"], *case_key(job)))


def log_path(config: dict[str, Any], job: dict[str, Any]) -> Path:
    base = LOG_ROOT / "replication_0" / f"{config['model_id']}__{job['arm']}" / job["suite"] / job["user_task_id"]
    return (base / "none" / "none.json" if job["stage"] == "retained_benign"
            else base / config["attack_vector"] / f"{job['injection_task_id']}.json")


def status(path: Path, attack: bool) -> str:
    if not path.exists():
        return "missing"
    try:
        record = read(path)
    except (OSError, json.JSONDecodeError):
        return "malformed"
    if not isinstance(record, dict):
        return "malformed"
    messages = record.get("messages")
    valid = (
        record.get("error") is None
        and isinstance(record.get("utility"), bool)
        and (not attack or isinstance(record.get("security"), bool))
        and isinstance(messages, list)
        and any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages)
    )
    return "valid" if valid else "invalid_or_error"


def main() -> None:
    config = read(CONFIG)
    manifest_path = ROOT / config["manifest"]
    manifest = read(manifest_path)
    preflight = read(ROOT / config["preflight"])
    if digest(manifest_path) != config["manifest_sha256"] or manifest["design_sha256"] != digest(DESIGN):
        raise ValueError("Frozen manifest or design hash mismatch")
    if manifest["partition_sha256"] != digest(PARTITION) or preflight["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Partition or preflight hash mismatch")
    if config["model_id"] != manifest["model_candidate"] or config["attack_vector"] != manifest["attack_vector"]:
        raise ValueError("Model or vector mismatch")
    if (config["local_extension_code_sha256"] != digest(ROOT / "paper1_extension" / "compositional_tasks.py")
            or config["local_extension_config_sha256"] != digest(ROOT / "configs" / "compositional_extension_v1.json")):
        raise ValueError("Local compositional extension changed")
    jobs = [job for job in manifest["jobs"] if job["stage"] in STAGES]
    if len(jobs) != 723 or Counter(job["arm"] for job in jobs) != Counter({arm: 241 for arm in ARMS}):
        raise ValueError("Retained workload differs from frozen design")
    expected = {}
    rows = []
    for job in jobs:
        seed = seed_for(manifest["seed_salt"], job)
        if job["random_seed"] != seed:
            raise ValueError("Provider seed mismatch")
        ident = trajectory_id(job)
        if ident in expected:
            raise ValueError("Duplicate trajectory")
        expected[ident] = seed
        path = log_path(config, job)
        rows.append({"stage": job["stage"], "arm": job["arm"], "suite": job["suite"],
                     "user_task_id": job["user_task_id"], "injection_task_id": job.get("injection_task_id"),
                     "trajectory_id": ident, "log_path": path.relative_to(ROOT).as_posix(),
                     "log_sha256": digest(path) if path.exists() else None,
                     "status": status(path, attack=job["stage"] == "retained_attack")})
    ledger_path = LOG_ROOT / "usage_metadata.jsonl"
    responses = ([json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                 if ledger_path.exists() else [])
    ids = [r.get("response_id") for r in responses]
    ledger_valid = (
        all(r.get("trajectory_id") in expected and r.get("random_seed") == expected[r["trajectory_id"]]
            and r.get("response_model") == config["model_id"] and r.get("response_id")
            for r in responses)
        and len(set(ids)) == len(ids)
    ) if responses else None
    by_stage = {}
    for stage in STAGES:
        subset = [row for row in rows if row["stage"] == stage]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in subset:
            key = (row["suite"], row["user_task_id"], row["injection_task_id"] or "none")
            groups.setdefault(key, []).append(row)
        if any(set(row["arm"] for row in trio) != set(ARMS) for trio in groups.values()):
            raise ValueError("Retained case arms are mismatched")
        by_stage[stage] = {
            "expected_jobs": len(subset),
            "expected_cases": len(groups),
            "status_counts": dict(Counter(row["status"] for row in subset)),
            "complete_three_arm_cases": sum(all(row["status"] == "valid" for row in trio) for trio in groups.values()),
            "all_valid": all(row["status"] == "valid" for row in subset),
        }
    complete = all(by_stage[stage]["all_valid"] for stage in STAGES)
    output = {
        "status": "COMPLETE_PENDING_OUTCOME_ANALYSIS" if complete else "PENDING_OR_INCOMPLETE",
        "confirmatory_outcome_available": False,
        "reason_outcomes_withheld": "No partial retained outcomes are exposed by the audit; use a separate frozen analysis only after 723/723 valid logs.",
        "manifest_sha256": config["manifest_sha256"],
        "local_extension_code_sha256": config["local_extension_code_sha256"],
        "usage_ledger_sha256": digest(ledger_path) if ledger_path.exists() else None,
        "stages": by_stage,
        "api_responses": len(responses),
        "unique_response_ids": len(set(ids)),
        "seed_and_trajectory_ledger_valid": ledger_valid,
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in responses),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in responses),
        "estimated_list_price_usd": sum(r.get("estimated_list_price_usd", 0) for r in responses),
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
