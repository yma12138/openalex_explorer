"""LLM abstraction layer using litellm + JSON mode.

Provides a unified interface for all LLM calls in the pipeline.
Supports any provider litellm supports (DeepSeek, OpenAI, Claude, Gemini, etc.)
via a JSON config file.

Usage:
    from src.llm import LLM
    from pydantic import BaseModel, Field

    config = {"model": "deepseek/deepseek-v4-flash", "api_key": "sk-..."}
    llm = LLM(config)

    # Plain text generation
    result = llm.invoke("Write a summary...")

    # Structured output (JSON mode)
    class Judgment(BaseModel):
        decision: str = Field(description="include or exclude")
    result = llm.structured("Judge this paper...", output_type=Judgment)
"""

import json
import logging
from typing import Any, Generator, Optional, TypeVar

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Cache for text responses (only when mocking)
_TESPONSE_CACHE: dict[str, str] = {}


class LLM:
    """Unified LLM interface backed by litellm.

    Initialized from a JSON config dict, then passed through the pipeline.
    All structured output uses JSON mode (response_format) instead of
    function calling.
    """

    def __init__(self, config: dict):
        self._model: str = config.get("model", "deepseek/deepseek-chat")
        self._api_key: str = config.get("api_key", "")
        self._temperature: float = config.get("temperature", 0.3)
        self._max_tokens: int = config.get("max_tokens", 4096)

        # Store extra litellm params (base_url, api_base, etc.)
        self._extra: dict = {
            k: v
            for k, v in config.items()
            if k not in ("model", "api_key", "temperature", "max_tokens")
        }

        # Token usage counters
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_calls: int = 0

    # ---- Basic invoke ----

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Send a prompt to the LLM and return the text response."""
        messages = self._build_messages(prompt, system_prompt)
        response = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            **self._extra,
            **kwargs,
        )
        self._record_usage(response)
        return response.choices[0].message.content

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of invoke."""
        messages = self._build_messages(prompt, system_prompt)
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            **self._extra,
            **kwargs,
        )
        self._record_usage(response)
        return response.choices[0].message.content

    # ---- Structured output (JSON mode) ----

    def structured(
        self,
        prompt: str,
        output_type: type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Prompt the LLM and parse the response into a structured Pydantic model.

        Uses response_format=\"json_object\" + JSON schema in the prompt,
        compatible with DeepSeek, OpenAI, and most modern providers.

        Args:
            prompt: The user/human message.
            output_type: A Pydantic BaseModel subclass defining the schema.
            system_prompt: Optional system instruction.

        Returns:
            An instance of output_type.
        """
        messages = self._build_messages(prompt, system_prompt, output_type)
        response = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            response_format={"type": "json_object"},
            **self._extra,
        )
        self._record_usage(response)
        text = response.choices[0].message.content
        return self._parse_json(text, output_type, messages)

    async def astructured(
        self,
        prompt: str,
        output_type: type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Async version of structured."""
        messages = self._build_messages(prompt, system_prompt, output_type)
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            response_format={"type": "json_object"},
            **self._extra,
        )
        self._record_usage(response)
        text = response.choices[0].message.content
        return self._parse_json(text, output_type, messages)

    # ---- Stream ----

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream the response token by token."""
        messages = self._build_messages(prompt, system_prompt)
        response = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            stream=True,
            **self._extra,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ---- Internal ----

    # ---- Token usage tracking ----

    def get_usage(self) -> dict:
        """Return accumulated token usage as a dict."""
        return {
            "calls": self._total_calls,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    def _record_usage(self, response) -> None:
        """Extract and accumulate token usage from a litellm response."""
        try:
            usage = response.usage
            self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
            self._total_completion_tokens += getattr(usage, "completion_tokens", 0)
            self._total_calls += 1
        except (AttributeError, Exception):
            pass

    # ---- Internal ----

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_type: Optional[type[T]] = None,
    ) -> list[dict]:
        """Build message list for LLM invocation.

        Args:
            prompt: The user/human message.
            system_prompt: Optional system instruction.
            output_type: If set, appends JSON schema to system prompt.

        Returns:
            List of message dicts for litellm.
        """
        messages: list[dict] = []
        system_parts: list[str] = []

        if system_prompt:
            system_parts.append(system_prompt)

        # Append JSON schema instruction for structured output
        if output_type is not None:
            schema = output_type.model_json_schema()
            # Drop title/description noise, keep properties + required
            schema_compact = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
            system_parts.append(
                "你必须输出JSON格式。输出必须符合以下JSON Schema:\n"
                + json.dumps(schema_compact, ensure_ascii=False, indent=2)
            )

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_json(
        self,
        text: str,
        output_type: type[T],
        messages: list[dict],
    ) -> T:
        """Parse JSON text into output_type with retry on failure.

        Args:
            text: Raw response text from LLM.
            output_type: Expected Pydantic model.
            messages: The original message list (extended on retry).

        Returns:
            Parsed output_type instance.

        Raises:
            ValueError: If parsing fails after retry.
        """
        clean = text.strip()
        # Strip markdown code fences if present
        if clean.startswith("```"):
            # Find the first { after ```
            start = clean.find("{")
            if start >= 0:
                clean = clean[start:]
        if clean.endswith("```"):
            clean = clean[: clean.rfind("```")].rstrip()

        try:
            data = json.loads(clean)
            return output_type.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "JSON parse failed (will retry): %s. Text: %s",
                e,
                text[:200],
            )

        # Retry once with explicit instruction
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Please output ONLY valid JSON matching the schema provided."
                ),
            }
        )
        response = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
            response_format={"type": "json_object"},
            **self._extra,
        )
        self._record_usage(response)
        text2 = response.choices[0].message.content
        text2 = text2.strip()
        if text2.startswith("```"):
            start = text2.find("{")
            if start >= 0:
                text2 = text2[start:]
        if text2.endswith("```"):
            text2 = text2[: text2.rfind("```")].rstrip()

        try:
            data = json.loads(text2)
            return output_type.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"LLM did not return valid JSON. Attempt 1: {text[:200]} | "
                f"Attempt 2: {text2[:200]}"
            ) from e
