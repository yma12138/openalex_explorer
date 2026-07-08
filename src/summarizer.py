"""论文Summary extraction — 从论文中提取结构化信息。

对通过质量审核的论文，逐 papers提取:
  - Research question: 论文试图解决什么问题
  - Method: 技术路线
  - 主要Findings: 核心结论 1-3 条
  - 局限性: 作者提到的不足
  - 与主题的关系: 对综述的价值
"""

import asyncio
import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.llm import LLM
from src.models import PaperSummary
from src.prompts import load_prompt
from src.store import PaperRecord, PaperStore
from src.topic import TopicSpec

logger = logging.getLogger(__name__)


def _with_instruction(system: str, lang_instruction: str = "") -> str:
    """如果提供了 lang_instruction，将其追加到 system prompt 前面。"""
    return lang_instruction + "\n\n" + system if lang_instruction else system


def _user_prompt(name: str, lang: str, **kwargs) -> str:
    """加载用户 prompt 模板并填充 {placeholder}。"""
    return load_prompt(name, lang).format(**kwargs)


class SummaryExtract(BaseModel):
    """Extract structured info from a paper."""

    research_question: str = Field(description="What problem does this paper solve")
    method: str = Field(description="Technical approach used")
    main_findings: list[str] = Field(
        default_factory=list,
        description="Core conclusions (1-3 items)",
        min_length=1,
        max_length=3,
    )
    limitations: str = Field(description="Limitations mentioned by the authors")
    relevance_to_topic: str = Field(
        description="Value of this paper to the review topic"
    )


def summarize_paper(
    llm: LLM,
    paper: PaperRecord,
    spec: TopicSpec,
    lang: str = "zh-hans",
) -> Optional[PaperSummary]:
    """同步：提取单篇论文的结构化信息。"""
    return asyncio.run(summarize_paper_async(llm, paper, spec, lang))


async def summarize_paper_async(
    llm: LLM,
    paper: PaperRecord,
    spec: TopicSpec,
    lang: str = "zh-hans",
    lang_instruction: str = "",
) -> Optional[PaperSummary]:
    """异步：提取单 papers论文的结构化信息。

    Args:
        llm: LLM 实例。
        paper: PaperRecord，至少包含 title + abstract。
        spec: TopicSpec，包含Research question用于上下文。

    Returns:
        PaperSummary 或 None（Extraction failed时）。
    """
    # 构建输入：优先用全文 md，否则用摘要
    full_text = ""
    if paper.md_path:
        try:
            from pathlib import Path

            full_text = Path(paper.md_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("读取全文失败 %s: %s", paper.id, e)

    content = full_text[:8000] if full_text else paper.abstract[:3000]

    if not content.strip():
        logger.warning("论文 %s 无内容可提取", paper.id)
        return None

    prompt = _user_prompt(
        "summary_user",
        lang,
        question=spec.research_question,
        title=paper.title,
        content=content,
    )
    system = _with_instruction(load_prompt("summary", lang), lang_instruction)

    try:
        result: SummaryExtract = llm.structured(
            prompt,
            output_type=SummaryExtract,
            system_prompt=system,
        )
        return PaperSummary(
            research_question=result.research_question,
            method=result.method,
            main_findings=result.main_findings,
            limitations=result.limitations,
            relevance_to_topic=result.relevance_to_topic,
        )
    except Exception as e:
        logger.error("提取摘要失败 %s: %s", paper.id, e)
        return None


async def batch_summarize_async(
    llm: LLM,
    store: PaperStore,
    spec: TopicSpec,
    limit: Optional[int] = None,
    lang: str = "zh-hans",
    lang_instruction: str = "",
) -> int:
    """批量提取所有 quality_passed 论文的结构化信息。

    Args:
        llm: LLM 实例。
        store: PaperStore。
        spec: TopicSpec。
        limit: 最多处理多少 papers。

    Returns:
        成功提取的论文数。
    """
    papers = store.list_papers(
        status="quality_passed",
        limit=limit or 100,
        sort_by="quality",
    )
    if not papers:
        print("  [Summary Extraction] No papers to process (status=quality_passed)")
        return 0

    print(f"\n{'=' * 60}")
    print(f"  Summary extraction — total {len(papers)} papers")
    print(f"{'=' * 60}")

    success = 0
    for i, p in enumerate(papers, 1):
        print(f"\n  [{i}/{len(papers)}] {p.title[:60]}")

        summary = await summarize_paper_async(llm, p, spec, lang, lang_instruction)
        if summary is None:
            print("      ❌ Extraction failed")
            continue

        summary_json = json.dumps(
            {
                "research_question": summary.research_question,
                "method": summary.method,
                "main_findings": summary.main_findings,
                "limitations": summary.limitations,
                "relevance_to_topic": summary.relevance_to_topic,
            },
            ensure_ascii=False,
        )

        store.update_status(p.id, "summarized", summary_json=summary_json)
        success += 1

        print(f"      ✅ Research question: {summary.research_question[:60]}...")
        print(f"        Method: {summary.method[:60]}...")
        print(f"        Findings: {'; '.join(summary.main_findings[:2])[:80]}...")

    print("\n  📊 Successfully processed")
    return success
