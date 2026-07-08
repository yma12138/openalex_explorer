"""Topic specification — clarify user intent into structured JSON.

Flow:
  User input -> LLM generates TopicSpec JSON -> validate_topic_spec()
  -> pass -> proceed to search
  -> retry -> re-generate with error feedback
  -> ask_user -> direct user input needed
"""

import json
from dataclasses import dataclass, field
from typing import Optional

TOPIC_SPEC_FIELDS = {
    "research_question": {
        "type": "string",
        "rule": "should be detailed and specific",
        "example": "大型语言模型的推理能力研究综述",
    },
    "start_year": {
        "type": "integer",
        "rule": "optional, e.g. 2020",
        "example": 2020,
    },
    "end_year": {
        "type": "integer",
        "rule": "optional, e.g. 2025",
        "example": 2025,
    },
    "min_papers": {
        "type": "integer",
        "rule": "≥ 1, ≤ 200",
        "example": 15,
    },
    "method_preference": {
        "type": "string",
        "rule": "optional, preferred research method",
        "example": "transformer-based, diffusion models",
    },
    "research_field": {
        "type": "string",
        "rule": "optional, specific field or domain",
        "example": "computer vision, natural language processing",
    },
}

CLARIFYING_QUESTIONS = [
    "1. 核心研究问题是什么？请用一句话明确描述。",
    "2. 起始年份是什么？(可选，如 2020，不填则不限制)",
    "3. 终止年份是什么？(可选，如 2025，不填则以当前年份为准)",
    "4. 最少需要多少篇论文？",
    "5. 有偏好的研究方法吗？(可选，如 transformer-based，没有可跳过)",
    "6. 限定在哪个研究领域吗？(可选，如 computer vision，没有可跳过)",
]


@dataclass
class TopicSpec:
    """Structured and validated topic specification."""

    research_question: str
    start_year: int = 0  # 0 means no limit
    end_year: int = 0  # 0 means current year
    min_papers: int = 15
    method_preference: str = ""
    research_field: str = ""

    def to_json(self) -> str:
        data = {
            "research_question": self.research_question,
            "min_papers": self.min_papers,
        }
        if self.start_year:
            data["start_year"] = self.start_year
        if self.end_year:
            data["end_year"] = self.end_year
        if self.method_preference:
            data["method_preference"] = self.method_preference
        if self.research_field:
            data["research_field"] = self.research_field
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TopicSpec":
        data = json.loads(text)
        return cls(
            research_question=data["research_question"],
            start_year=data.get("start_year", 0) or 0,
            end_year=data.get("end_year", 0) or 0,
            min_papers=data.get("min_papers", 15),
            method_preference=data.get("method_preference", ""),
            research_field=data.get("research_field", ""),
        )


@dataclass
class ValidationResult:
    """Result of a TopicSpec validation."""

    passed: bool
    action: str  # "pass" | "retry" | "ask_user"
    errors: list[str] = field(default_factory=list)
    topic_spec: Optional[TopicSpec] = None


def validate_topic_spec(
    json_text: str,
    retry_count: int = 0,
    max_retries: int = 3,
) -> ValidationResult:
    """Validate a TopicSpec JSON string against the schema.

    Args:
        json_text: Raw JSON string to validate.
        retry_count: Number of retries already attempted.
        max_retries: Max retries before switching to ask_user.

    Returns:
        ValidationResult with pass/retry/ask_user decision.
    """
    # 1. Check it's valid JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        errors = [f"JSON 格式错误: {e}"]
        return _retry_or_ask(errors, retry_count, max_retries)

    if not isinstance(data, dict):
        errors = ["JSON 根必须是对象 (object)"]
        return _retry_or_ask(errors, retry_count, max_retries)

    errors = []

    # 2. Check required fields
    required = ["research_question"]
    missing = [f for f in required if f not in data]
    if missing:
        errors.append(f"缺少必填字段: {', '.join(missing)}")
        return _retry_or_ask(errors, retry_count, max_retries)

    # 3. Validate research_question
    rq = data["research_question"]
    if not isinstance(rq, str):
        errors.append("research_question 必须是字符串")
    elif len(rq.strip()) < 10:
        errors.append(
            f"research_question 太短 ({len(rq.strip())} 字符), "
            f"至少需要 10 个字符，请更具体地描述研究问题"
        )

    # 4. Validate start_year (optional)
    sy = data.get("start_year", 0) or 0
    if sy:
        if not isinstance(sy, int) or isinstance(sy, bool):
            errors.append("start_year 必须是整数")
        elif sy < 1900 or sy > 2100:
            errors.append(f"start_year 范围不合理 (当前值: {sy})")

    # 5. Validate end_year (optional)
    ey = data.get("end_year", 0) or 0
    if ey:
        if not isinstance(ey, int) or isinstance(ey, bool):
            errors.append("end_year 必须是整数")
        elif ey < 1900 or ey > 2100:
            errors.append(f"end_year 范围不合理 (当前值: {ey})")
        elif sy and ey < sy:
            errors.append(f"end_year ({ey}) 不能小于 start_year ({sy})")

    # 6. Validate min_papers (optional, default 15)
    mp = data.get("min_papers", 15)
    if not isinstance(mp, int) or isinstance(mp, bool):
        errors.append("min_papers 必须是整数")
    elif mp < 1:
        errors.append(f"min_papers 不能小于 1 (当前值: {mp})")
    elif mp > 200:
        errors.append(f"min_papers 不能大于 200 (当前值: {mp})")

    # 7. Optional fields — validate type if present
    for field_name in ("method_preference", "research_field"):
        val = data.get(field_name)
        if val is not None and val != "" and not isinstance(val, str):
            errors.append(f"{field_name} 必须是字符串")

    if errors:
        return _retry_or_ask(errors, retry_count, max_retries)

    return ValidationResult(
        passed=True,
        action="pass",
        topic_spec=TopicSpec(
            research_question=rq.strip(),
            start_year=sy,
            end_year=ey,
            min_papers=mp,
            method_preference=data.get("method_preference", ""),
            research_field=data.get("research_field", ""),
        ),
    )


def _retry_or_ask(
    errors: list[str],
    retry_count: int,
    max_retries: int,
) -> ValidationResult:
    """Decide whether to retry or ask the user based on retry count."""
    if retry_count >= max_retries:
        return ValidationResult(
            passed=False,
            action="ask_user",
            errors=errors,
        )
    return ValidationResult(
        passed=False,
        action="retry",
        errors=errors,
    )


def format_retry_prompt(
    errors: list[str],
    spec_fields: dict = TOPIC_SPEC_FIELDS,
) -> str:
    """Format validation errors into a prompt for LLM to fix the JSON."""

    fields_desc = "\n".join(
        f"  - {k}: {v['type']}, {v['rule']} (示例: {v['example']})"
        for k, v in spec_fields.items()
    )

    error_text = "\n".join(f"  ❌ {e}" for e in errors)

    return f"""TopicSpec JSON 校验未通过，请根据以下错误修正：

{error_text}

输出格式（JSON，research_question 为必填，
start_year / end_year / min_papers 为可选（有默认值），
method_preference / research_field 为可选）：
{fields_desc}

请只返回修正后的 JSON，不要包含其他内容。"""


def format_ask_user_prompt(errors: list[str]) -> str:
    """Format errors into a message asking user to clarify."""
    error_text = "\n".join(f"  ❌ {e}" for e in errors)
    questions = "\n".join(CLARIFYING_QUESTIONS)

    return f"""系统无法生成有效的 TopicSpec，原因是：

{error_text}

请直接回答以下问题：

{questions}
"""
