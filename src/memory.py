"""Memory management for LLM agents.

每个 Agent 有独立的记忆空间，支持不同策略：
- conversation: 完整对话历史
- summary: 只保留摘要（节省 token）
- keyvalue: 键值对存储（搜索状态等）
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

MemoryStrategy = Literal["conversation", "summary", "keyvalue"]


@dataclass
class AgentMemory:
    """A single agent's memory."""

    strategy: MemoryStrategy = "conversation"
    max_turns: int = 20

    # conversation strategy
    messages: list[dict[str, str]] = field(default_factory=list)

    # keyvalue strategy
    store: dict[str, Any] = field(default_factory=dict)

    # summary strategy
    summary: str = ""

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        if self.strategy != "conversation":
            return
        self.messages.append({"role": role, "content": content})
        # Trim oldest if over limit
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2) :]

    def set(self, key: str, value: Any):
        """Set a key-value pair (for keyvalue strategy)."""
        self.store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def set_summary(self, text: str):
        self.summary = text

    def to_context(self, max_chars: int = 4000) -> str:
        """Format memory as context string for LLM.

        Args:
            max_chars: Max characters to include. 0 = no limit.
        """
        if self.strategy == "conversation":
            parts = []
            for m in self.messages:
                prefix = "用户" if m["role"] == "user" else "助手"
                parts.append(f"{prefix}: {m['content'][:200]}")
            text = "\n".join(parts)
            if max_chars and len(text) > max_chars:
                text = "（历史记录过长，截断）\n" + text[-max_chars:]
            return text

        elif self.strategy == "summary":
            return self.summary

        elif self.strategy == "keyvalue":
            return json.dumps(self.store, ensure_ascii=False, indent=2)

        return ""

    def clear(self):
        self.messages.clear()
        self.store.clear()
        self.summary = ""
