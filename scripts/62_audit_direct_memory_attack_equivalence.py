"""Verify fixed attack payload equivalence across retained arms, offline."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.load_suites import get_suites


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper1_extension.compositional_tasks import EXTENSION_VERSION, get_paper1_suite

CONFIG = ROOT / "configs" / "direct_memory_retained_v2.json"
OUTPUT = ROOT / "results" / "exploratory" / "direct_memory_attack_equivalence_v2.json"
ARMS = ("no_defense", "stateful_rho_v1", "stateless_rho_v1")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_digest(payload: dict[str, str]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    config = read(CONFIG)
    manifest_path = ROOT / config["manifest"]
    if digest(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    if (config["local_extension_version"] != list(EXTENSION_VERSION)
            or digest(ROOT / "paper1_extension" / "compositional_tasks.py") != config["local_extension_code_sha256"]):
        raise ValueError("Local compositional extension changed")
    manifest = read(manifest_path)
    jobs = [job for job in manifest["jobs"] if job["stage"] == "retained_attack"]
    if len(jobs) != 585:
        raise ValueError("Expected 585 retained attack jobs")
    MODEL_NAMES[config["model_id"]] = "Ministral"
    suites = dict(get_suites(config["benchmark_version"]))
    for name in ("slack", "workspace"):
        suites[name] = get_paper1_suite(name)
    attacks = {
        (suite, arm): load_attack(
            config["attack_vector"], suites[suite],
            SimpleNamespace(name=f"{config['model_id']}__{arm}"),
        )
        for suite in sorted({job["suite"] for job in jobs})
        for arm in ARMS
    }
    rows = []
    for index in range(0, len(jobs), 3):
        trio = jobs[index:index + 3]
        key = (trio[0]["suite"], trio[0]["user_task_id"], trio[0]["injection_task_id"])
        if (len(trio) != 3 or set(job["arm"] for job in trio) != set(ARMS)
                or any((job["suite"], job["user_task_id"], job["injection_task_id"]) != key for job in trio)):
            raise ValueError("Case triplet differs across arms")
        suite_name, user_id, goal_id = key
        suite = suites[suite_name]
        user = suite.get_user_task_by_id(user_id)
        goal = suite.get_injection_task_by_id(goal_id)
        payloads = {arm: attacks[(suite_name, arm)].attack(user, goal) for arm in ARMS}
        if not payloads["no_defense"]:
            raise ValueError(f"Case has no injectable payload: {key}")
        if any(payloads[arm] != payloads["no_defense"] for arm in ARMS[1:]):
            raise ValueError(f"Attack payload differs across arms: {key}")
        rows.append({"suite": suite_name, "user_task_id": user_id,
                     "injection_task_id": goal_id, "injection_locations": sorted(payloads["no_defense"]),
                     "payload_sha256": payload_digest(payloads["no_defense"])})
    if len(rows) != 195 or len({(row["suite"], row["user_task_id"], row["injection_task_id"]) for row in rows}) != 195:
        raise ValueError("Unexpected retained attack case count")
    output = {
        "status": "VERIFIED_OFFLINE_NO_MODEL_CALLS",
        "manifest_sha256": config["manifest_sha256"],
        "attack_vector": config["attack_vector"],
        "cases": len(rows),
        "jobs": len(jobs),
        "suites": dict(Counter(row["suite"] for row in rows)),
        "all_three_arm_payloads_identical": True,
        "case_payload_hashes": rows,
        "interpretation_limit": "Verifies attack construction before execution; provider responses and policy outcomes remain unobserved.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "case_payload_hashes"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
