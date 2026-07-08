"""Named Agent — 统一管理 system prompt、tools、memory。

每个 Agent 从 JSON 配置加载，有自己的：
- 名称和描述
- system prompt（txt 文件路径）
- tool 定义（LLM function calling）
- 记忆策略
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from src.llm import LLM
from src.memory import AgentMemory, MemoryStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

AGENTS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = AGENTS_DIR / "configs"
PROMPTS_DIR = AGENTS_DIR / "prompts"


class ToolDef:
    """A tool definition: Pydantic model + description."""

    def __init__(self, model: type[BaseModel], description: str = ""):
        self.model = model
        self.description = description or model.__doc__ or ""


@dataclass
class AgentConfig:
    """Agent configuration loaded from JSON."""

    name: str
    description: str = ""
    system_prompt: str = ""  # loaded from file
    system_prompt_file: str = ""
    memory_strategy: MemoryStrategy = "conversation"
    max_turns: int = 20


class Agent:
    """A named LLM agent with its own prompt, tools, and memory.

    Usage:
        agent = Agent.load("asker")
        result = agent.run("用户输入")
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        tools: Optional[list[ToolDef]] = None,
        memory_strategy: MemoryStrategy = "conversation",
        max_turns: int = 20,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.memory = AgentMemory(strategy=memory_strategy, max_turns=max_turns)
        self._llm: Optional[LLM] = None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            # Legacy Agent — provide minimal config (this module is deprecated)
            self._llm = LLM(config={"api_key": "", "model": "deepseek/deepseek-chat"})
        return self._llm

    @classmethod
    def load(cls, name: str) -> "Agent":
        """Load an agent from config JSON.

        Looks for configs/{name}.json and prompts/{name}.txt.
        """
        config_path = CONFIGS_DIR / f"{name}.json"
        prompt_path = PROMPTS_DIR / f"{name}.txt"

        if not config_path.exists():
            logger.warning(f"Agent config not found: {config_path}")
            return cls(name=name)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        cfg = AgentConfig(
            name=cfg_data.get("name", name),
            description=cfg_data.get("description", ""),
            system_prompt_file=cfg_data.get("system_prompt_file", ""),
            memory_strategy=cfg_data.get("memory", {}).get("type", "conversation"),
            max_turns=cfg_data.get("memory", {}).get("max_turns", 20),
        )

        # Load system prompt
        sp_path = prompt_path
        if cfg.system_prompt_file:
            sp_path = PROMPTS_DIR / cfg.system_prompt_file
        system_prompt = ""
        if sp_path.exists():
            with open(sp_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()

        return cls(
            name=cfg.name,
            system_prompt=system_prompt,
            memory_strategy=cfg.memory_strategy,
            max_turns=cfg.max_turns,
        )

    def run(
        self,
        prompt: str,
        output_type: Optional[type[T]] = None,
    ) -> Any:
        """Run the agent with a prompt.

        Args:
            prompt: Current user input / task description.
            output_type: Optional Pydantic model for structured output.

        Returns:
            LLM response text, or Pydantic instance if output_type given.
        """
        # Build full context with memory
        memory_context = self.memory.to_context()
        full_system = self._build_system(memory_context)

        if output_type:
            result = self.llm.structured(
                prompt,
                output_type=output_type,
                system_prompt=full_system,
            )
            self.memory.add_message("user", prompt)
            self.memory.add_message(
                "assistant", self._clean_surrogates(str(result)[:200])
            )
            return result

        response = self.llm.invoke(prompt, system_prompt=full_system)
        response = self._clean_surrogates(response)
        self.memory.add_message("user", prompt)
        self.memory.add_message("assistant", response)
        return response

    @staticmethod
    def _clean_surrogates(text: str) -> str:
        """Remove surrogate characters that break UTF-8 encoding.

        Some LLM APIs return lone surrogates (e.g. \\udcef),
        which Python can't encode as UTF-8.
        """
        result = []
        for ch in text:
            if 0xD800 <= ord(ch) <= 0xDFFF:
                result.append("?")
            else:
                result.append(ch)
        return "".join(result)

    def _build_system(self, memory_context: str) -> str:
        parts = [self.system_prompt]
        if memory_context:
            parts.append("\n\n对话历史：\n" + memory_context)
        return "\n".join(parts)

    def set_memory(self, key: str, value: Any):
        self.memory.set(key, value)

    def get_memory(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)

    def clear_memory(self):
        self.memory.clear()
