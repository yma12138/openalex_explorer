"""Relevance and quality filters for paper screening.

同步/异步双模式:
  - relevance_filter() / batch_relevance_filter()          ← 同步兼容
  - relevance_filter_async() / batch_relevance_filter_async()  ← 异步
  - quality_filter() / batch_quality_filter()              ← 同步兼容
  - quality_filter_async() / batch_quality_filter_async()       ← 异步 + gather
"""

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.llm import LLM
from src.prompts import load_prompt
from src.store import PaperRecord, PaperStore
from src.topic import TopicSpec

logger = logging.getLogger(__name__)


def _with_instruction(system: str, lang_instruction: str = "") -> str:
    """如果提供了 lang_instruction，将其追加到 system prompt 前面。"""
    return lang_instruction + "\n\n" + system if lang_instruction else system


# ═══════════════════════════════════════════════════
# Relevance filter — title + abstract screening
# ═══════════════════════════════════════════════════


class RelevanceJudgment(BaseModel):
    """Judge a paper's relevance to the research question.

    Score 0-100: 0-30不相关 31-59弱相关 60-79相关 80-100高度相关
    """

    score: float = Field(ge=0, le=100, description="Relevance score (0-100)")
    reason: str = Field(
        min_length=10,
        description="Reason for the score (required, min 10 chars)",
    )


def relevance_filter(
    llm: LLM,
    paper: PaperRecord,
    spec: TopicSpec,
    screened_titles: list[str],
) -> tuple[float, str]:
    """同步：单 papers论文相关度判断。返回 (score, reason)。"""
    return asyncio.run(relevance_filter_async(llm, paper, spec, screened_titles))


async def relevance_filter_async(
    llm: LLM,
    paper: PaperRecord,
    spec: TopicSpec,
    screened_titles: list[str],
    lang: str = "zh-hans",
    lang_instruction: str = "",
) -> tuple[float, str]:
    """异步：单 papers论文相关度判断。返回 (score, reason)。"""
    already_passed = ""
    if screened_titles:
        already_passed = "\n\n已pass论文主题方向（避免重复）:\n" + "\n".join(
            f"  - {t[:80]}" for t in screened_titles[-5:]
        )

    prompt = (
        f"研究问题: {spec.research_question}\n"
        f"论文标题: {paper.title}\n"
        f"论文摘要: {paper.abstract[:1500]}\n{already_passed}"
    )
    system = _with_instruction(load_prompt("relevance", lang), lang_instruction)

    result: RelevanceJudgment = await llm.astructured(
        prompt,
        output_type=RelevanceJudgment,
        system_prompt=system,
    )
    reason = result.reason.strip() if result.reason else "LLM 未提供理由"
    return result.score, reason


def batch_relevance_filter(
    llm: LLM,
    store: PaperStore,
    spec: TopicSpec,
    limit: Optional[int] = None,
) -> dict[str, float]:
    """同步：批量Relevance filter。"""
    return asyncio.run(batch_relevance_filter_async(llm, store, spec, limit))


async def batch_relevance_filter_async(
    llm: LLM,
    store: PaperStore,
    spec: TopicSpec,
    limit: Optional[int] = None,
) -> dict[str, float]:
    """异步：批量Relevance filter（顺序执行，依赖已pass论文列表）。"""
    papers = store.list_papers(status="searched", limit=limit or 500)
    if not papers:
        print("  [Relevance filter] No papers pending review (status=searched)")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  Relevance filter — total {len(papers)}  pending")
    print(f"{'=' * 60}")

    passed = store.list_papers(status="relevanced", limit=10)
    screened_titles = [p.title for p in passed]

    passed_count = 0
    results = {}

    for i, p in enumerate(papers, 1):
        score, reason = await relevance_filter_async(llm, p, spec, screened_titles)

        if score >= 60:
            store.update_status(
                p.id,
                "relevanced",
                relevance_score=score,
                relevance_reason=reason,
            )
            screened_titles.append(p.title)
            passed_count += 1
            results[p.id] = score
            status = "✅ pass"
        else:
            store.update_status(
                p.id,
                "rejected",
                relevance_score=score,
                relevance_reason=reason,
            )
            status = "❌ reject"

        print(f"  [{i:2d}/{len(papers)}] {status}  score={score:.1f}  {p.title}")

    print(
        f"\n  📊 Result: {passed_count}/{len(papers)}  passed "
        f"({(passed_count / len(papers) * 100):.0f}%)"
    )
    return results


# ═══════════════════════════════════════════════════
# Quality filter — citation / venue / author scoring
# ═══════════════════════════════════════════════════


class QualityJudgment(BaseModel):
    """Judge a paper's quality based on citation, venue, and author metrics.

    Score 0-100: 0-30低质量 31-59一般 60-79良好 80-100优秀
    """

    score: float = Field(ge=0, le=100, description="Quality score (0-100)")
    reason: str = Field(
        min_length=10,
        description="Reason for the score (required, min 10 chars)",
    )


def quality_filter(
    llm: LLM,
    paper: PaperRecord,
    profile: dict,
    spec: TopicSpec,
) -> tuple[float, str]:
    """同步：单 papers论文质量判断。"""
    return asyncio.run(quality_filter_async(llm, paper, profile, spec))


async def quality_filter_async(
    llm: LLM,
    paper: PaperRecord,
    profile: dict,
    spec: TopicSpec,
    lang: str = "zh-hans",
    lang_instruction: str = "",
) -> tuple[float, str]:
    """异步：单 papers论文质量判断。"""
    profile_lines = [
        f"论文: {paper.title[:80]}",
        f"年份: {paper.year}",
        f"引用数: {profile.get('cited_by_count', 'N/A')}",
    ]

    venue = profile.get("venue_stats")
    if venue:
        profile_lines.append(
            f"期刊: {venue.get('name', '')} ({venue.get('type', '')})"
            f" — 2年引用均值: {venue.get('2yr_mean_citedness', 'N/A')}"
            f", H指数: {venue.get('h_index', 'N/A')}"
        )

    authors = profile.get("authors", [])
    if authors:
        author_lines = ["作者统计:"]
        for a in authors[:5]:
            if "error" not in a:
                author_lines.append(
                    f"  - {a.get('name', '')}: {a.get('works_count', 0)} papers, "
                    f"{a.get('cited_by_count', 0)} 引用, "
                    f" papers均 {a.get('avg_citations_per_paper', 0)}"
                )
        profile_lines.extend(author_lines)

    months = profile.get("months_since_publication")
    if months is not None:
        profile_lines.append(f"距今 {months} 个月")

    prompt = "\n".join(profile_lines)
    system = _with_instruction(load_prompt("quality", lang), lang_instruction)

    result: QualityJudgment = await llm.astructured(
        prompt,
        output_type=QualityJudgment,
        system_prompt=system,
    )
    reason = result.reason.strip() if result.reason else "LLM 未提供理由"
    return result.score, reason


def batch_quality_filter(
    llm: LLM,
    store: PaperStore,
    spec: TopicSpec,
    limit: Optional[int] = None,
) -> dict[str, float]:
    """同步：批量Quality filter。"""
    return asyncio.run(batch_quality_filter_async(llm, store, spec, limit))


async def batch_quality_filter_async(
    llm: LLM,
    store: PaperStore,
    spec: TopicSpec,
    limit: Optional[int] = None,
) -> dict[str, float]:
    """异步：批量Quality filter — profile 构建 + LLM 打分全部并行。

    流程:
      1. 从 SQLite 读取所有 relevanced 论文
      2. 并行构建所有论文的 profile（每 papers内部 work/venue/authors 并行）
      3. 并行调用 LLM 打分
      4. 串行写回 SQLite（SQLite 单写者模型）
    """
    from src.models import PaperMeta
    from src.paper_profile import build_paper_profile_async

    papers = store.list_papers(status="relevanced", limit=limit or 100)
    if not papers:
        print("  [Quality filter] No papers pending review (status=relevanced)")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  Quality filter — total {len(papers)}  pending")
    print(f"{'=' * 60}")

    # ── Step 1: 并行构建所有 profile ──────────────────────────
    print("  Building paper profiles in parallel...")
    metas = [
        PaperMeta(
            id=p.id,
            title=p.title,
            authors=p.authors,
            abstract=p.abstract,
            year=p.year,
            source=p.source,
        )
        for p in papers
    ]
    profiles = await asyncio.gather(*[build_paper_profile_async(m) for m in metas])
    print(f"  ✓ Profile building done ({len(profiles)}  papers)")

    # ── Step 2: 并行 LLM 打分 ─────────────────────────────────
    print("  Parallel LLM quality scoring...")
    score_tasks = [
        quality_filter_async(llm, p, profiles[i], spec) for i, p in enumerate(papers)
    ]
    score_results = await asyncio.gather(*score_tasks)
    print(f"  ✓ Scoring done ({len(score_results)}  papers)")

    # ── Step 3: 串行写回 SQLite ───────────────────────────────
    passed_count = 0
    results = {}
    for i, p in enumerate(papers):
        score, reason = score_results[i]
        profile = profiles[i]

        cits = profile.get("cited_by_count", 0)
        venue = profile.get("venue_stats", {})
        venue_name = venue.get("name", "") if venue else ""

        if score >= 60:
            store.update_status(
                p.id,
                "quality_passed",
                quality_score=score,
                quality_reason=reason,
            )
            passed_count += 1
            results[p.id] = score
            status = "✅ pass"
        else:
            store.update_status(
                p.id,
                "rejected",
                quality_score=score,
                quality_reason=reason,
            )
            status = "❌ reject"

        print(
            f"  [{i + 1:2d}/{len(papers)}] {status}  score={score:.1f}"
            f"  cit={cits}  {venue_name[:25]:25s}  {p.title}"
        )

    print(
        f"\n  📊 Result: {passed_count}/{len(papers)}  passed "
        f"({(passed_count / len(papers) * 100):.0f}%)"
    )
    return results
