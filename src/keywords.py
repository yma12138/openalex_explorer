"""Keyword brainstorming and refinement for paper search.

同步/异步双模式:
  - brainstorm_keywords()         ← 同步
  - brainstorm_keywords_async()   ← 异步
"""

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.llm import LLM
from src.prompts import load_prompt
from src.sources.openalex import OpenAlexSource
from src.topic import TopicSpec

logger = logging.getLogger(__name__)


def _with_instruction(system: str, lang_instruction: str = "") -> str:
    """如果提供了 lang_instruction，将其追加到 system prompt 前面。"""
    return lang_instruction + "\n\n" + system if lang_instruction else system


# ═══════════════════════════════════════════════════
# Tool schemas (Pydantic = function calling definition)
# ═══════════════════════════════════════════════════


class KeywordSet(BaseModel):
    """Initial keyword brainstorming from a research question.

    Generate diverse keyword groups that cover different aspects
    of the research question. Each keyword should be a complete
    search query (3-6 words).
    """

    keywords: list[str] = Field(
        description=(
            "搜索关键词列表，每个关键词 3-6 个英文词。"
            "根据研究问题的复杂度和覆盖方向，自行决定生成多少组。"
            "最少 5 组。"
        ),
        min_length=5,
    )


class KeywordRefinement(BaseModel):
    """Refine and expand keywords based on search results.

    Review the papers found so far and decide:
    - What new keywords to try next?
    - Are there uncovered aspects?
    - Should we continue searching?
    """

    new_keywords: list[str] = Field(
        default_factory=list,
        description="Newly discovered keywords for next search round",
    )
    uncovered_aspects: list[str] = Field(
        default_factory=list,
        description="Aspects not covered by current results",
    )
    total_found: int = Field(
        description="Total number of papers found so far",
        ge=0,
    )
    max_results_per_keyword: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "本轮新关键词每个应搜索多少篇论文。"
            "如果之前的关键词返回结果太少，可以增大此值（如 50 或 100）。"
        ),
    )
    should_continue: bool = Field(
        description="Whether to continue searching (True=more directions available)",
    )
    stop_reason: str = Field(
        default="",
        description="Reason to stop searching",
    )


# ═══════════════════════════════════════════════════
# Brainstorm → Search → Refine loop
# ═══════════════════════════════════════════════════


def brainstorm_keywords(llm: LLM, spec: TopicSpec) -> list[str]:
    """同步：生成初始搜索关键词。"""
    return asyncio.run(brainstorm_keywords_async(llm, spec))


async def brainstorm_keywords_async(
    llm: LLM, spec: TopicSpec, lang: str = "zh-hans",
    lang_instruction: str = "",
) -> list[str]:
    """异步：生成初始搜索关键词。"""
    prompt = (
        f"研究问题: {spec.research_question}\n"
        f"研究领域: {spec.research_field or '(不限)'}\n"
        f"偏好方法: {spec.method_preference or '(不限)'}\n\n"
        "请针对这个综述需求生成搜索关键词。"
        "每组关键词 3-6 个英文词，覆盖不同子方向。"
        "根据研究问题的覆盖范围自行决定需要多少组，至少 5 组。"
    )
    system = _with_instruction(load_prompt("brainstorm", lang), lang_instruction)
    result: KeywordSet = await llm.astructured(
        prompt, output_type=KeywordSet, system_prompt=system
    )
    logger.info(f"初始关键词 ({len(result.keywords)} 个): {result.keywords}")
    return result.keywords


def refine_keywords(
    llm: LLM,
    spec: TopicSpec,
    current_keywords: list[str],
    papers_summary: str,
    total_found: int,
    lang: str = "zh-hans",
    lang_instruction: str = "",
) -> KeywordRefinement:
    """Review current search results and decide next steps."""
    prompt = (
        f"研究问题: {spec.research_question}\n"
        f"已尝试的关键词: {', '.join(current_keywords)}\n"
        f"已找到 {total_found} 篇论文\n\n"
        f"部分论文摘要：\n{papers_summary}\n\n"
        "请分析当前搜索结果，决定：\n"
        "1. 是否需要新的关键词来覆盖未涉及的方面？\n"
        "2. 当前搜索方向是否已经足够？\n"
        "3. 如果停止，原因是什么？"
    )
    system = _with_instruction(load_prompt("keyword_refine", lang), lang_instruction)

    result: KeywordRefinement = llm.structured(
        prompt, output_type=KeywordRefinement, system_prompt=system
    )
    return result


def keyword_search(
    keywords: list[str],
    max_per_keyword: int = 20,
    filters: Optional[dict] = None,
) -> dict[str, str]:
    """Search OpenAlex for each keyword.

    Returns dict mapping paper_id → title for dedup and review.
    """
    source = OpenAlexSource()
    all_papers: dict[str, str] = {}  # paper_id → title

    for kw in keywords:
        result = source.search(
            query=kw,
            max_results=max_per_keyword,
            sort="relevance_score:desc",
            filters=filters,
        )
        for p in result.papers:
            if p.id not in all_papers:
                all_papers[p.id] = p.title

    logger.info(f"关键词搜索后共 {len(all_papers)} 篇去重论文")
    return {pid: all_papers[pid] for pid in list(all_papers.keys())[:200]}


def summarize_papers_for_llm(papers: dict[str, str], max_count: int = 15) -> str:
    """Format paper titles for LLM review."""
    lines = []
    for i, (pid, title) in enumerate(papers.items()):
        if i >= max_count:
            break
        lines.append(f"  [{i + 1}] {pid}: {title[:80]}")
    return "\n".join(lines) if lines else "  (无)"
