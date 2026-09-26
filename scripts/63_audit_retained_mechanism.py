"""Conservative post-outcome trace audit for the retained memory comparison.

This is a descriptive mechanism check, not a second confirmatory endpoint.
Only an identical common prefix of tool proposals is used to compare policy
decisions. A later matching call after the traces diverge is not treated as
evidence that the policies faced the same history.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "primary" / "direct_memory_retained_audit_v2.json"
OUTPUT = ROOT / "results" / "robustness" / "direct_memory_retained_mechanism_v2.json"
STATEFUL = "stateful_rho_v1"
STATELESS = "stateless_rho_v1"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    result = []
    for message in record["messages"]:
        if message.get("role") != "tool":
            continue
        call = message.get("tool_call") or {}
        error = message.get("error") or ""
        decision = "REPLAN" if error.startswith("REPLAN:") else "BLOCK" if error.startswith("BLOCK:") else "ALLOW"
        args = json.dumps(call.get("args"), sort_keys=True, ensure_ascii=False)
        result.append((str(call.get("function")), args, decision))
    return result


def initial_context(record: dict[str, Any]) -> str:
    """Canonicalize messages preceding the first model response without exposing text."""
    prefix = []
    for message in record["messages"]:
        if message.get("role") == "assistant":
            break
        prefix.append(message)
    return json.dumps(prefix, sort_keys=True, ensure_ascii=False)


def first_trace_divergence(left: list[tuple[str, str, str]],
                           right: list[tuple[str, str, str]]) -> dict[str, Any]:
    matched_prefix = 0
    for a, b in zip(left, right):
        if a[:2] != b[:2]:
            return {"kind": "proposal_difference", "matched_prefix_proposals": matched_prefix}
        matched_prefix += 1
        if a[2] != b[2]:
            return {"kind": "decision_difference_on_matched_proposal",
                    "matched_prefix_proposals": matched_prefix,
                    "stateful_decision": a[2], "stateless_decision": b[2]}
    if len(left) != len(right):
        return {"kind": "trace_length_difference", "matched_prefix_proposals": matched_prefix}
    return {"kind": "identical_tool_trace", "matched_prefix_proposals": matched_prefix}


def main() -> None:
    audit = read(AUDIT)
    if (audit["status"] != "COMPLETE_PENDING_OUTCOME_ANALYSIS"
            or audit["stages"]["retained_benign"]["all_valid"] is not True
            or audit["stages"]["retained_attack"]["all_valid"] is not True):
        raise RuntimeError("Complete audited retained logs are required")

    grouped: dict[str, dict[tuple[str, str, str | None], dict[str, dict[str, Any]]]] = {
        "retained_benign": defaultdict(dict), "retained_attack": defaultdict(dict)
    }
    for row in audit["rows"]:
        if row["status"] != "valid":
            raise RuntimeError("Audit contains a nonvalid retained log")
        path = ROOT / row["log_path"]
        if sha256(path) != row["log_sha256"]:
            raise RuntimeError("Retained log changed since independent audit")
        if row["arm"] not in (STATEFUL, STATELESS):
            continue
        record = read(path)
        if record.get("error") is not None:
            raise RuntimeError("Invalid retained record")
        key = (row["suite"], row["user_task_id"], row["injection_task_id"])
        grouped[row["stage"]][key][row["arm"]] = record

    output: dict[str, Any] = {
        "status": "DESCRIPTIVE_POST_OUTCOME_MECHANISM_AUDIT",
        "audited_usage_ledger_sha256": audit["usage_ledger_sha256"],
        "definition": "Compare only the common prefix of identical tool function and arguments. The first different decision on an identical proposal is a controller divergence, not proof that it caused a later outcome difference.",
        "stages": {},
    }
    for stage, cases in grouped.items():
        expected = 46 if stage == "retained_benign" else 195
        if len(cases) != expected or any(set(arms) != {STATEFUL, STATELESS} for arms in cases.values()):
            raise RuntimeError("Incomplete retained stateful/stateless pairs")
        counts: Counter[str] = Counter()
        by_outcome: Counter[str] = Counter()
        matched_prefix_total = 0
        prefix_by_kind: dict[str, Counter[int]] = defaultdict(Counter)
        divergent_decisions: Counter[str] = Counter()
        initial_context_equal: Counter[str] = Counter()
        for arms in cases.values():
            left, right = arms[STATEFUL], arms[STATELESS]
            difference = first_trace_divergence(trace(left), trace(right))
            kind = difference["kind"]
            counts[kind] += 1
            initial_context_equal[f"{kind}:{initial_context(left) == initial_context(right)}"] += 1
            matched_prefix_total += difference["matched_prefix_proposals"]
            prefix_by_kind[kind][difference["matched_prefix_proposals"]] += 1
            if kind == "decision_difference_on_matched_proposal":
                divergent_decisions[f"{difference['stateful_decision']}/{difference['stateless_decision']}"] += 1
            outcome = "utility" if stage == "retained_benign" else "security"
            if type(left[outcome]) is not bool or type(right[outcome]) is not bool:
                raise RuntimeError("Outcome is not Boolean")
            outcome_difference = int(left[outcome]) - int(right[outcome])
            by_outcome[f"{kind}:{outcome_difference:+d}"] += 1
        output["stages"][stage] = {
            "paired_cases": expected,
            "first_divergence_counts": dict(counts),
            "common_prefix_matched_proposals": matched_prefix_total,
            "matched_prefix_length_distribution_by_kind": {
                kind: dict(sorted(lengths.items())) for kind, lengths in sorted(prefix_by_kind.items())
            },
            "first_matched_decision_types": dict(divergent_decisions),
            "initial_context_equal_by_kind": dict(initial_context_equal),
            "first_divergence_by_outcome_difference": dict(by_outcome),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
