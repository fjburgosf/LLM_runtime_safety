"""Leakage-resistant, deterministic risk features for Paper 1 tool-call proposals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "capability_registry_v0.json"


@dataclass(frozen=True)
class RiskFeatureRecord:
    tool_name: str
    capability_level: int
    capability_score: float
    below_memory_impact_floor: bool
    untrusted_content_seen: bool
    provenance_bonus: float
    sensitive_arguments_present: tuple[str, ...]
    sensitive_argument_bonus: float
    risk_signal: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CapabilityRiskEstimator:
    """Score proposed calls from a fixed capability registry and prior provenance."""

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY,
        max_risk_signal: float | None = None,
        minimum_impact_level: int = 0,
    ) -> None:
        self.registry_path = registry_path
        self.registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.impact_by_tool = {
            tool: int(level)
            for level, tools in self.registry["tools_by_impact"].items()
            for tool in tools
        }
        self.untrusted_content_tools = set(self.registry["untrusted_content_tools"])
        self.sensitive_arguments_by_tool = self.registry["sensitive_arguments_by_tool"]
        self.impact_scores = {int(level): score for level, score in self.registry["impact_scores"].items()}
        self.provenance_bonus = self.registry["risk_formula"]["provenance_bonus"]
        self.sensitive_argument_bonus = self.registry["risk_formula"]["sensitive_argument_bonus"]
        configured_cap = self.registry["risk_formula"].get("max_risk_signal", 1.0)
        self.max_risk_signal = configured_cap if max_risk_signal is None else max_risk_signal
        if not 0 < self.max_risk_signal <= 1:
            raise ValueError("max_risk_signal must be in (0, 1]")
        self.minimum_impact_level = minimum_impact_level
        if not 0 <= self.minimum_impact_level <= 4:
            raise ValueError("minimum_impact_level must be in [0, 4]")

    def score(self, tool_name: str, arguments: dict[str, Any], untrusted_content_seen: bool) -> RiskFeatureRecord:
        if tool_name not in self.impact_by_tool:
            raise KeyError(f"tool {tool_name!r} is absent from the capability registry")
        level = self.impact_by_tool[tool_name]
        capability_score = self.impact_scores[level]
        below_floor = level < self.minimum_impact_level
        sensitive_present = tuple(
            parameter
            for parameter in self.sensitive_arguments_by_tool.get(tool_name, [])
            if arguments.get(parameter) not in (None, "", [], {}, False)
        )
        provenance = (
            self.provenance_bonus
            if not below_floor and untrusted_content_seen and capability_score > 0
            else 0.0
        )
        sensitivity = self.sensitive_argument_bonus if not below_floor and sensitive_present else 0.0
        return RiskFeatureRecord(
            tool_name=tool_name,
            capability_level=level,
            capability_score=capability_score,
            below_memory_impact_floor=below_floor,
            untrusted_content_seen=untrusted_content_seen,
            provenance_bonus=provenance,
            sensitive_arguments_present=sensitive_present,
            sensitive_argument_bonus=sensitivity,
            risk_signal=(
                0.0
                if below_floor
                else min(self.max_risk_signal, capability_score + provenance + sensitivity)
            ),
        )


class ProvenanceTracker:
    """Track whether designated content-bearing tool output has entered the trajectory."""

    def __init__(self, estimator: CapabilityRiskEstimator) -> None:
        self.estimator = estimator
        self._untrusted_content_seen = False

    @property
    def untrusted_content_seen(self) -> bool:
        return self._untrusted_content_seen

    def reset(self) -> None:
        self._untrusted_content_seen = False

    def score_proposal(self, tool_name: str, arguments: dict[str, Any]) -> RiskFeatureRecord:
        return self.estimator.score(tool_name, arguments, self._untrusted_content_seen)

    def observe_executed_tool(self, tool_name: str) -> None:
        if tool_name in self.estimator.untrusted_content_tools:
            self._untrusted_content_seen = True
