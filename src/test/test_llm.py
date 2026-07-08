"""Tests for LLM abstraction (litellm + JSON mode)."""

import json
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LLM


class Judgment(BaseModel):
    """A test structured output schema."""

    paper_id: str = Field(description="Paper ID")
    decision: Literal["include", "exclude"] = Field(description="Decision")
    reason: str = Field(description="Reason")


class TestLLMInit:
    """Test LLM initialization from config dict."""

    def test_init_defaults(self):
        """Should use defaults for missing config keys."""
        llm = LLM({"api_key": "placeholder"})
        assert llm._model == "deepseek/deepseek-chat"
        assert llm._api_key == "placeholder"
        assert llm._temperature == 0.3
        assert llm._max_tokens == 4096

    def test_init_full_config(self):
        """Should read all fields from config dict."""
        config = {
            "model": "openai/gpt-4o",
            "api_key": "sk-test",
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        llm = LLM(config)
        assert llm._model == "openai/gpt-4o"
        assert llm._api_key == "sk-test"
        assert llm._temperature == 0.7
        assert llm._max_tokens == 2048

    def test_init_extra_params(self):
        """Should store extra params (base_url, etc.) for litellm."""
        config = {
            "model": "deepseek/deepseek-v4-flash",
            "api_key": "test",
            "api_base": "https://custom.example.com/v1",
        }
        llm = LLM(config)
        assert llm._extra["api_base"] == "https://custom.example.com/v1"


class TestLLMPromptBuilding:
    """Test message construction."""

    def test_build_messages_no_system(self):
        """No system prompt should produce only user message."""
        llm = LLM({"api_key": "test"})
        msgs = llm._build_messages("Hello", None)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Hello"}

    def test_build_messages_with_system(self):
        """System prompt should appear before user."""
        llm = LLM({"api_key": "test"})
        msgs = llm._build_messages("Hello", "You are a bot.")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are a bot."}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_build_messages_with_schema(self):
        """With output_type, should append JSON schema to system prompt."""
        llm = LLM({"api_key": "test"})
        msgs = llm._build_messages("Classify", "You judge papers.", Judgment)
        assert len(msgs) == 2
        sys_content: str = msgs[0]["content"]
        assert "You judge papers." in sys_content
        assert "JSON Schema" in sys_content
        assert "paper_id" in sys_content

    def test_build_messages_schema_only(self):
        """With output_type but no system_prompt, schema is the only system msg."""
        llm = LLM({"api_key": "test"})
        msgs = llm._build_messages("Classify", None, Judgment)
        assert len(msgs) == 2
        sys_content: str = msgs[0]["content"]
        assert "JSON Schema" in sys_content
        assert msgs[1] == {"role": "user", "content": "Classify"}


class TestJSONParsing:
    """Test _parse_json with various response formats."""

    def test_parse_plain_json(self):
        """Should parse clean JSON."""
        llm = LLM({"api_key": "test"})
        text = '{"paper_id": "123", "decision": "include", "reason": "Good paper"}'
        result = llm._parse_json(text, Judgment, [])
        assert isinstance(result, Judgment)
        assert result.paper_id == "123"
        assert result.decision == "include"

    def test_parse_json_with_code_fence(self):
        """Should strip markdown code fences."""
        llm = LLM({"api_key": "test"})
        text = '```json\n{"paper_id": "456", "decision": "exclude", "reason": "Bad"}\n```'
        result = llm._parse_json(text, Judgment, [])
        assert result.paper_id == "456"
        assert result.decision == "exclude"

    @patch("litellm.completion")
    def test_parse_retry_on_failure(self, mock_completion):
        """Should retry once on malformed JSON."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"paper_id": "789", "decision": "include", "reason": "OK"}'
            ))]
        )
        llm = LLM({"api_key": "test"})
        text = "not json at all"
        msgs = [{"role": "user", "content": "hello"}]
        result = llm._parse_json(text, Judgment, msgs)
        assert result.paper_id == "789"
        # Should have added a retry message
        assert len(msgs) == 2
        assert "not valid JSON" in msgs[1]["content"]

    @patch("litellm.completion")
    def test_parse_retry_still_fails(self, mock_completion):
        """Should raise on second failure."""
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="still not json"))]
        )
        llm = LLM({"api_key": "test"})
        import pytest
        with pytest.raises(ValueError, match="did not return valid JSON"):
            llm._parse_json("bad", Judgment, [])


class TestStructuredOutputSchema:
    """Test that structured output schemas are correctly defined."""

    def test_judgment_schema(self):
        """Verify that the Judgment Pydantic model generates correct JSON schema."""
        schema = Judgment.model_json_schema()
        assert schema["required"] == ["paper_id", "decision", "reason"]
        assert schema["properties"]["decision"]["enum"] == ["include", "exclude"]

    def test_build_messages_includes_schema(self):
        """Build messages with output_type should include the schema."""
        llm = LLM({"api_key": "test"})
        msgs = llm._build_messages("test", None, Judgment)
        sys_content: str = msgs[0]["content"]
        # Should include the schema
        assert "paper_id" in sys_content
        assert "decision" in sys_content
        assert "required" in sys_content
