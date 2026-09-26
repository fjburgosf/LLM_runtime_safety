"""Deterministic closed-loop risk supervisor for Paper 1.

The module intentionally has no model, benchmark or network dependency. It makes
the proposed controller testable before it is connected to AgentDojo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil, log


class Intervention(StrEnum):
    ALLOW = "ALLOW"
    REPLAN = "REPLAN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class ControllerConfig:
    budget: float
    beta: float
    replan_threshold: float
    block_threshold: float
    max_consecutive_replans: int
    stateful: bool

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if not 0 <= self.replan_threshold < self.block_threshold <= self.budget:
            raise ValueError("thresholds must satisfy 0 <= replan < block <= budget")
        if self.max_consecutive_replans < 0:
            raise ValueError("max_consecutive_replans cannot be negative")


@dataclass(frozen=True)
class DecisionRecord:
    risk_signal: float
    prior_state: float
    decision_statistic: float
    intervention: Intervention
    next_state: float
    consecutive_replans: int


class RiskSupervisor:
    """A per-trajectory supervisor with ALLOW, REPLAN and BLOCK actions.

    A REPLAN does not execute the proposed action and does not add its risk signal
    to the state. The next candidate action must be evaluated again. An ALLOW adds
    the signal to the state only when `stateful` is true.
    """

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self._state = 0.0
        self._consecutive_replans = 0
        self._blocked = False

    @property
    def state(self) -> float:
        return self._state

    @property
    def blocked(self) -> bool:
        return self._blocked

    def reset(self) -> None:
        self._state = 0.0
        self._consecutive_replans = 0
        self._blocked = False

    def _decision_statistic(self, risk_signal: float) -> float:
        if not 0 <= risk_signal <= self.config.budget:
            raise ValueError("risk_signal must be in [0, budget]")
        if not self.config.stateful:
            return risk_signal
        return min(self.config.budget, self.config.beta * self._state + risk_signal)

    def evaluate(self, risk_signal: float) -> DecisionRecord:
        """Evaluate one candidate tool call and update only valid state transitions."""
        if self._blocked:
            raise RuntimeError("a blocked trajectory must be reset before evaluation")

        prior_state = self._state
        statistic = self._decision_statistic(risk_signal)
        if statistic >= self.config.block_threshold:
            intervention = Intervention.BLOCK
            self._blocked = True
            next_state = self._state
        elif statistic >= self.config.replan_threshold:
            self._consecutive_replans += 1
            if self._consecutive_replans > self.config.max_consecutive_replans:
                intervention = Intervention.BLOCK
                self._blocked = True
            else:
                intervention = Intervention.REPLAN
            next_state = self._state
        else:
            intervention = Intervention.ALLOW
            self._consecutive_replans = 0
            if self.config.stateful:
                self._state = statistic
            next_state = self._state

        return DecisionRecord(
            risk_signal=risk_signal,
            prior_state=prior_state,
            decision_statistic=statistic,
            intervention=intervention,
            next_state=next_state,
            consecutive_replans=self._consecutive_replans,
        )


def record_as_dict(record: DecisionRecord) -> dict[str, float | int | str]:
    """Return a JSON-serializable record for logs and reproducibility artifacts."""
    output = asdict(record)
    output["intervention"] = record.intervention.value
    return output


def state_after_signals(
    signals: list[float], beta: float, budget: float, initial_state: float = 0.0
) -> list[float]:
    """Return unconstrained saturated integrator states for an analytical trace."""
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 <= initial_state <= budget:
        raise ValueError("initial_state must be in [0, budget]")
    state = initial_state
    states: list[float] = []
    for signal in signals:
        if not 0 <= signal <= budget:
            raise ValueError("every signal must be in [0, budget]")
        state = min(budget, beta * state + signal)
        states.append(state)
    return states


def minimum_prior_state_for_stateful_only_intervention(
    risk_signal: float,
    beta: float,
    stateful_threshold: float,
    memoryless_threshold: float,
) -> float | None:
    """Minimum prior state for stateful intervention while memoryless allows.

    None means the supplied action is already rejected by the memoryless policy or
    that state cannot affect the decision because beta is zero.
    """
    if risk_signal < 0:
        raise ValueError("risk_signal cannot be negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if stateful_threshold < 0 or memoryless_threshold < 0:
        raise ValueError("thresholds cannot be negative")
    if risk_signal >= memoryless_threshold or beta == 0:
        return None
    return max(0.0, (stateful_threshold - risk_signal) / beta)


def constant_signal_crossing_steps(
    risk_signal: float,
    beta: float,
    threshold: float,
    initial_state: float = 0.0,
) -> int | None:
    """First unsaturated step at which a constant signal reaches a threshold."""
    if risk_signal < 0 or threshold < 0 or initial_state < 0:
        raise ValueError("risk_signal, threshold and initial_state cannot be negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if initial_state >= threshold:
        return 0
    if beta == 0:
        return 1 if risk_signal >= threshold else None
    if beta * initial_state + risk_signal >= threshold:
        return 1
    equilibrium = risk_signal / (1 - beta)
    if equilibrium <= threshold or initial_state >= equilibrium:
        return None
    ratio = (equilibrium - threshold) / (equilibrium - initial_state)
    candidate = max(2, ceil(log(ratio) / log(beta) - 1e-12))

    def state_at(step: int) -> float:
        return equilibrium + beta**step * (initial_state - equilibrium)

    while candidate > 1 and state_at(candidate - 1) >= threshold:
        candidate -= 1
    while state_at(candidate) < threshold:
        candidate += 1
    return candidate


def zero_signal_recovery_steps(
    state: float, beta: float, release_threshold: float
) -> int | None:
    """Allowed zero-risk steps required for state to reach a release threshold."""
    if state < 0 or release_threshold < 0:
        raise ValueError("state and release_threshold cannot be negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if state <= release_threshold:
        return 0
    if beta == 0:
        return 1
    if release_threshold == 0:
        return None
    return ceil(log(release_threshold / state) / log(beta))


def constant_signal_state(
    risk_signal: float, beta: float, steps: int, initial_state: float = 0.0
) -> float:
    """Closed-form unsaturated state after a constant signal for `steps` updates."""
    if risk_signal < 0 or initial_state < 0:
        raise ValueError("risk_signal and initial_state cannot be negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if steps < 0:
        raise ValueError("steps cannot be negative")
    if steps == 0:
        return initial_state
    return beta**steps * initial_state + risk_signal * (1 - beta**steps) / (1 - beta)


def minimum_beta_for_detection(
    risk_signal: float, detection_horizon: int, intervention_threshold: float
) -> float | None:
    """Smallest beta that reaches a threshold within a fixed signal horizon.

    None means no beta in [0, 1) can satisfy the requested detection condition.
    """
    if risk_signal < 0 or intervention_threshold < 0:
        raise ValueError("risk_signal and intervention_threshold cannot be negative")
    if detection_horizon <= 0:
        raise ValueError("detection_horizon must be positive")
    if risk_signal >= intervention_threshold:
        return 0.0
    if risk_signal * detection_horizon <= intervention_threshold:
        return None

    low = 0.0
    high = 1.0
    for _ in range(100):
        midpoint = (low + high) / 2
        state = constant_signal_state(risk_signal, midpoint, detection_horizon)
        if state >= intervention_threshold:
            high = midpoint
        else:
            low = midpoint
    return high


def maximum_beta_for_recovery(
    budget: float, release_threshold: float, max_recovery_steps: int
) -> float:
    """Largest beta satisfying beta**H * budget <= release_threshold."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 <= release_threshold <= budget:
        raise ValueError("release_threshold must be in [0, budget]")
    if max_recovery_steps <= 0:
        raise ValueError("max_recovery_steps must be positive")
    return (release_threshold / budget) ** (1 / max_recovery_steps)


def recovery_constrained_beta_interval(
    risk_signal: float,
    detection_horizon: int,
    intervention_threshold: float,
    budget: float,
    release_threshold: float,
    max_recovery_steps: int,
) -> tuple[float, float] | None:
    """Return the beta interval satisfying detection and recovery constraints."""
    minimum = minimum_beta_for_detection(
        risk_signal, detection_horizon, intervention_threshold
    )
    maximum = maximum_beta_for_recovery(
        budget, release_threshold, max_recovery_steps
    )
    if minimum is None or minimum > maximum:
        return None
    return minimum, maximum
