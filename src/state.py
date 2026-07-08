"""Global state definition for the LangGraph pipeline.

Flows through 4 stages:
  明确问题 → 文献搜寻 → 文献过滤 → 综述攥写
"""

from dataclasses import dataclass, field
from typing import Optional

from src.models import PaperMeta
from src.topic import TopicSpec


@dataclass
class PipelineState:
    """Shared state across the entire LangGraph pipeline."""

    # --- Stage 1: 明确问题 ---
    topic_spec: Optional[TopicSpec] = None
    topic_spec_json: str = ""  # raw JSON string from LLM
    validation_retries: int = 0
    validation_action: str = ""  # "pass" / "retry" / "ask_user"
    validation_errors: list[str] = field(default_factory=list)

    # --- Database ---
    db_path: str = "papers.db"  # SQLite database path

    # --- Stage 2: 文献搜寻 ---
    search_query: str = ""
    search_results: list[PaperMeta] = field(default_factory=list)
    current_keywords: list[str] = field(default_factory=list)
    search_round: int = 0
    keyword_refine_result: Optional[str] = None

    # --- Stage 3: 文献过滤 ---
    filtered_papers: list[PaperMeta] = field(default_factory=list)

    # --- Stage 4: 综述攥写 ---
    review_draft: str = ""

    # --- Errors ---
    errors: list[str] = field(default_factory=list)
