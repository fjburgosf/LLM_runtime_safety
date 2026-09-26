"""Factory for a reset-safe AgentDojo pipeline with Paper 1 risk supervision."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

from control_policy import ControllerConfig, RiskSupervisor
from risk_features import CapabilityRiskEstimator, ProvenanceTracker
from supervised_tools_executor import SupervisedToolsExecutor


class ResetRuntimeRisk(BasePipelineElement):
    """Reset controller and provenance before each independent pipeline query."""

    def __init__(self, executor: SupervisedToolsExecutor) -> None:
        self.executor = executor

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        self.executor.reset()
        return query, runtime, env, messages, extra_args


def build_supervised_pipeline(
    llm: BasePipelineElement,
    system_message: str,
    controller_config: ControllerConfig,
    risk_estimator_kwargs: dict[str, Any] | None = None,
    reset_on_query: bool = True,
) -> AgentPipeline:
    """Build a supervisor, optionally resetting on each AgentDojo query.

    Set ``reset_on_query=False`` only when the caller constructs a fresh
    pipeline for every benchmark task. This preserves state if AgentDojo
    retries that same task with another query while preventing cross-task
    carryover through job-scoped pipeline construction.
    """
    supervisor = RiskSupervisor(controller_config)
    provenance = ProvenanceTracker(CapabilityRiskEstimator(**(risk_estimator_kwargs or {})))
    executor = SupervisedToolsExecutor(supervisor, provenance)
    elements = [SystemMessage(system_message)]
    if reset_on_query:
        elements.append(ResetRuntimeRisk(executor))
    elements.extend([InitQuery(), llm, ToolsExecutionLoop([executor, llm])])
    pipeline = AgentPipeline(elements)
    pipeline.name = f"{getattr(llm, 'name', 'custom')}-stateful-risk-v0"
    return pipeline
