"""Tests for TopicSpec validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.topic import (
    TopicSpec,
    format_ask_user_prompt,
    format_retry_prompt,
    validate_topic_spec,
)


class TestValidateTopicSpec:
    """Test JSON schema validation."""

    # --- Valid cases ---

    def test_valid_required_only(self):
        """Only research_question is required."""
        j = '{"research_question": "A very specific research question here"}'
        result = validate_topic_spec(j)
        assert result.passed
        assert result.action == "pass"
        spec = result.topic_spec
        assert spec.research_question == "A very specific research question here"
        assert spec.start_year == 0
        assert spec.end_year == 0
        assert spec.min_papers == 15

    def test_valid_with_years(self):
        j = (
            '{"research_question": "A specific question",'
            ' "start_year": 2020, "end_year": 2025, "min_papers": 10}'
        )
        result = validate_topic_spec(j)
        assert result.passed
        spec = result.topic_spec
        assert spec.start_year == 2020
        assert spec.end_year == 2025
        assert spec.min_papers == 10

    def test_valid_with_optional_fields(self):
        j = (
            '{"research_question": "A very specific research question here",'
            ' "start_year": 2022, "end_year": 2025, "min_papers": 10,'
            ' "method_preference": "transformer-based",'
            ' "research_field": "NLP"}'
        )
        result = validate_topic_spec(j)
        assert result.passed
        assert result.topic_spec.method_preference == "transformer-based"
        assert result.topic_spec.research_field == "NLP"

    def test_valid_start_year_only(self):
        j = (
            '{"research_question": "A specific question",'
            ' "start_year": 2020, "min_papers": 5}'
        )
        result = validate_topic_spec(j)
        assert result.passed
        assert result.topic_spec.start_year == 2020
        assert result.topic_spec.end_year == 0

    # --- Invalid JSON ---

    def test_invalid_json(self):
        result = validate_topic_spec("not json")
        assert not result.passed
        assert result.action == "retry"
        assert any("JSON" in e for e in result.errors)

    def test_empty_string(self):
        result = validate_topic_spec("")
        assert not result.passed

    def test_not_a_dict(self):
        result = validate_topic_spec("[1, 2, 3]")
        assert not result.passed

    # --- Missing required fields ---

    def test_missing_research_question(self):
        result = validate_topic_spec('{"start_year": 2020}')
        assert not result.passed
        assert any("缺少" in e for e in result.errors)

    # --- research_question ---

    def test_question_too_short(self):
        j = '{"research_question": "short"}'
        result = validate_topic_spec(j)
        assert not result.passed
        assert any("太短" in e for e in result.errors)

    def test_question_not_string(self):
        j = '{"research_question": 123}'
        result = validate_topic_spec(j)
        assert not result.passed

    # --- start_year / end_year ---

    def test_start_year_out_of_range(self):
        j = '{"research_question": "A specific question", "start_year": 1800}'
        result = validate_topic_spec(j)
        assert not result.passed

    def test_end_year_before_start(self):
        j = (
            '{"research_question": "A specific question",'
            ' "start_year": 2025, "end_year": 2020}'
        )
        result = validate_topic_spec(j)
        assert not result.passed
        assert any("小于" in e for e in result.errors)

    def test_year_not_int(self):
        j = (
            '{"research_question": "A specific question",'
            ' "start_year": "twenty twenty"}'
        )
        result = validate_topic_spec(j)
        assert not result.passed

    # --- min_papers ---

    def test_min_papers_zero(self):
        j = '{"research_question": "A specific question", "min_papers": 0}'
        result = validate_topic_spec(j)
        assert not result.passed
        assert any("小于 1" in e for e in result.errors)

    def test_min_papers_negative(self):
        j = '{"research_question": "A specific question", "min_papers": -5}'
        result = validate_topic_spec(j)
        assert not result.passed

    def test_min_papers_too_large(self):
        j = '{"research_question": "A specific question", "min_papers": 999}'
        result = validate_topic_spec(j)
        assert not result.passed

    def test_min_papers_not_int(self):
        j = '{"research_question": "A specific question", "min_papers": "fifteen"}'
        result = validate_topic_spec(j)
        assert not result.passed

    # --- Optional fields ---

    def test_valid_without_optional_fields(self):
        j = '{"research_question": "A very specific research question here"}'
        result = validate_topic_spec(j)
        assert result.passed
        assert result.topic_spec.method_preference == ""
        assert result.topic_spec.research_field == ""

    def test_optional_field_wrong_type(self):
        j = '{"research_question": "A specific question", "method_preference": 123}'
        result = validate_topic_spec(j)
        assert not result.passed

    # --- Retry logic ---

    def test_retry_count_tracking(self):
        js = '{"research_question": "short"}'
        assert validate_topic_spec(js, retry_count=0).action == "retry"
        assert validate_topic_spec(js, retry_count=1).action == "retry"
        assert validate_topic_spec(js, retry_count=2).action == "retry"
        assert validate_topic_spec(js, retry_count=3).action == "ask_user"

    def test_default_max_retries_is_three(self):
        js = '{"research_question": "short"}'
        result = validate_topic_spec(js, retry_count=3)
        assert result.action == "ask_user"

    # --- Format helpers ---

    def test_format_retry_prompt_includes_fields(self):
        prompt = format_retry_prompt(["research_question 太短"])
        assert "research_question" in prompt
        assert "start_year" in prompt
        assert "min_papers" in prompt

    def test_format_ask_user_prompt_includes_questions(self):
        prompt = format_ask_user_prompt(["research_question 太短"])
        assert "核心研究问题" in prompt
        assert "起始年份" in prompt

    # --- TopicSpec methods ---

    def test_topic_spec_to_json(self):
        spec = TopicSpec(
            research_question="测试问题",
            start_year=2022,
            end_year=2025,
            min_papers=15,
        )
        j = spec.to_json()
        assert "测试问题" in j
        assert "2022" in j
        assert "2025" in j
        assert "15" in j

    def test_topic_spec_from_json(self):
        j = (
            '{"research_question": "test",'
            ' "start_year": 2020, "end_year": 2025, "min_papers": 5}'
        )
        spec = TopicSpec.from_json(j)
        assert spec.research_question == "test"
        assert spec.start_year == 2020
        assert spec.end_year == 2025
        assert spec.min_papers == 5

    def test_topic_spec_to_json_includes_optionals(self):
        spec = TopicSpec(
            research_question="test",
            start_year=2022,
            min_papers=10,
            method_preference="diffusion models",
            research_field="computer vision",
        )
        j = spec.to_json()
        assert "diffusion models" in j
        assert "computer vision" in j

    def test_topic_spec_to_json_omits_empty_optionals(self):
        spec = TopicSpec(research_question="test")
        j = spec.to_json()
        assert "start_year" not in j
        assert "method_preference" not in j
