"""AgentDojo execution boundary for the Paper 1 stateful risk supervisor.

This adapter intercepts a proposed tool call before `FunctionsRuntime` executes it.
It is intentionally separate from the benchmark snapshot so the benchmark is never
modified in place.
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, is_string_list
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from control_policy import Intervention, RiskSupervisor, record_as_dict
from risk_features import ProvenanceTracker


class SupervisedToolsExecutor(ToolsExecutor):
    """Execute only tool calls allowed by a `RiskSupervisor`.

    REPLAN and BLOCK are represented as tool-result errors. This lets the following
    LLM iteration see that the candidate was not executed and, for REPLAN, propose
    another call. The adapter itself does not invoke a model or use task outcomes.
    """

    def __init__(self, supervisor: RiskSupervisor, provenance: ProvenanceTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supervisor = supervisor
        self.provenance = provenance

    def reset(self) -> None:
        """Reset state once before each independent AgentDojo trajectory."""
        self.supervisor.reset()
        self.provenance.reset()

    @staticmethod
    def _rejection(tool_call: Any, intervention: Intervention, detail: str) -> ChatToolResultMessage:
        return ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string("")],
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            error=f"{intervention.value}: tool call was not executed. {detail}",
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls = messages[-1]["tool_calls"]
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        records = extra_args.setdefault("runtime_risk_records", [])
        tool_call_results: list[ChatToolResultMessage] = []
        runtime_names = {tool.name for tool in runtime.functions.values()}
        for tool_call in tool_calls:
            if tool_call.function == EMPTY_FUNCTION_NAME:
                tool_call_results.append(self._rejection(tool_call, Intervention.BLOCK, "Provide a valid function name."))
                continue
            if tool_call.function not in runtime_names:
                tool_call_results.append(self._rejection(tool_call, Intervention.BLOCK, "The function is unavailable."))
                continue
            if self.supervisor.blocked:
                tool_call_results.append(self._rejection(tool_call, Intervention.BLOCK, "The trajectory is already blocked."))
                continue

            for arg_name, arg_value in tool_call.args.items():
                if isinstance(arg_value, str) and is_string_list(arg_value):
                    tool_call.args[arg_name] = literal_eval(arg_value)

            try:
                features = self.provenance.score_proposal(tool_call.function, dict(tool_call.args))
            except KeyError:
                tool_call_results.append(self._rejection(tool_call, Intervention.BLOCK, "The tool is absent from the fixed risk registry."))
                continue
            decision = self.supervisor.evaluate(features.risk_signal)
            records.append({"features": features.as_dict(), "decision": record_as_dict(decision)})
            if decision.intervention is not Intervention.ALLOW:
                guidance = "Propose a lower-risk alternative consistent with the user task." if decision.intervention is Intervention.REPLAN else "Stop this trajectory."
                tool_call_results.append(self._rejection(tool_call, decision.intervention, guidance))
                continue

            tool_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            if error is None:
                self.provenance.observe_executed_tool(tool_call.function)
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(self.output_formatter(tool_result))],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=error,
                )
            )
        return query, runtime, env, [*messages, *tool_call_results], extra_args
