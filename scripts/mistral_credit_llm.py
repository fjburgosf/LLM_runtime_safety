"""AgentDojo adapter for Mistral's OpenAI-compatible chat API.

The local ledger estimates list-price consumption of included API allowance.
It does not inspect or change account billing settings.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
from openai._types import NOT_GIVEN

from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


def mistral_message(message: ChatMessage, model: str) -> dict[str, Any]:
    """Translate AgentDojo content blocks to the Mistral chat schema."""
    converted: dict[str, Any] = dict(_message_to_openai(message, model))
    if converted["role"] == "developer":
        converted["role"] = "system"
    content = converted.get("content")
    if isinstance(content, list):
        converted["content"] = "\n".join(
            part.get("text", "") for part in content if part.get("type") == "text"
        )
    return converted


class MistralIncludedCreditLLM(OpenAILLM):
    """Call the pinned model with local request and included-credit ceilings."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        *,
        temperature: float,
        max_completion_tokens: int,
        max_api_requests: int,
        min_request_interval_seconds: float,
        max_included_credit_usd: float,
        single_request_reserve_usd: float,
        price_input_per_million_usd: float,
        price_output_per_million_usd: float,
        metadata_path: Path,
        random_seed: int | None = None,
        trajectory_id: str | None = None,
    ) -> None:
        super().__init__(client, model, temperature=temperature)
        if random_seed is not None and (type(random_seed) is not int or random_seed < 0):
            raise ValueError("random_seed must be a nonnegative integer")
        self.name = model
        self.random_seed = random_seed
        self.trajectory_id = trajectory_id
        self.max_completion_tokens = max_completion_tokens
        self.max_api_requests = max_api_requests
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_included_credit_usd = max_included_credit_usd
        self.single_request_reserve_usd = single_request_reserve_usd
        self.price_input_per_million_usd = price_input_per_million_usd
        self.price_output_per_million_usd = price_output_per_million_usd
        self.metadata_path = metadata_path
        self._starting_response_count = self._completed_response_count()
        self._request_count = 0
        self._last_request_started = 0.0

    def _observed_list_price_usd(self) -> float:
        if not self.metadata_path.exists():
            return 0.0
        return sum(
            json.loads(line)["estimated_list_price_usd"]
            for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def _completed_response_count(self) -> int:
        if not self.metadata_path.exists():
            return 0
        return sum(bool(line.strip()) for line in self.metadata_path.read_text(encoding="utf-8").splitlines())

    def _reserve_request(self) -> None:
        if self._starting_response_count + self._request_count >= self.max_api_requests:
            raise RuntimeError("Local Mistral compatibility request cap reached")
        if self._observed_list_price_usd() + self.single_request_reserve_usd > self.max_included_credit_usd:
            raise RuntimeError("Local Mistral included-credit cap reached")
        remaining = self.min_request_interval_seconds - (time.monotonic() - self._last_request_started)
        if self._request_count and remaining > 0:
            time.sleep(remaining)
        self._request_count += 1
        self._last_request_started = time.monotonic()

    @staticmethod
    def _retry_delay(error: openai.RateLimitError) -> float:
        header = error.response.headers.get("retry-after") if error.response else None
        if header:
            try:
                return max(1.0, float(header) + 1.0)
            except ValueError:
                pass
        match = re.search(r"try again in ([0-9.]+)s", str(error), flags=re.IGNORECASE)
        return max(1.0, float(match.group(1)) + 1.0) if match else 30.0

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        request_messages = [mistral_message(message, self.model) for message in messages]
        request_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        for attempt in range(3):
            self._reserve_request()
            try:
                request_kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=request_messages,
                    tools=request_tools or NOT_GIVEN,
                    tool_choice="auto" if request_tools else NOT_GIVEN,
                    parallel_tool_calls=False,
                    temperature=self.temperature,
                    max_tokens=self.max_completion_tokens,
                )
                if self.random_seed is not None:
                    request_kwargs["extra_body"] = {"random_seed": self.random_seed}
                completion = self.client.chat.completions.create(**request_kwargs)
                break
            except openai.RateLimitError as error:
                if attempt == 2:
                    raise
                delay = self._retry_delay(error)
                if delay > 120:
                    raise RuntimeError("Mistral retry delay exceeds 120 seconds") from error
                time.sleep(delay)
        if completion.usage is None or completion.usage.prompt_tokens is None or completion.usage.completion_tokens is None:
            raise RuntimeError("Mistral response has no token usage, so the local credit guard cannot continue")
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        estimated_list_price_usd = (
            prompt_tokens * self.price_input_per_million_usd
            + completion_tokens * self.price_output_per_million_usd
        ) / 1_000_000
        metadata = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "requested_model": self.model,
            "response_model": completion.model,
            "response_id": completion.id,
            "finish_reason": completion.choices[0].finish_reason,
            "usage": completion.usage.model_dump(mode="json", exclude_none=True),
            "estimated_list_price_usd": estimated_list_price_usd,
        }
        if self.random_seed is not None:
            metadata["random_seed"] = self.random_seed
        if self.trajectory_id is not None:
            metadata["trajectory_id"] = self.trajectory_id
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args
