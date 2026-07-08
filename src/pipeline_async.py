"""工厂流水线 — 搜索 → Relevance → 质量，三个阶段像流水线一样持续流转。

设计原则:
  1. 每个阶段是独立的 worker 池，通过 asyncio.Queue 解耦
  2. 一经搜索到 → 立即进入Relevance；一经通过 → 立即进入质量
  3. 搜索阶段支持关键词反哺循环（搜 → 检查数量 → 反哺 → 再搜）
  4. 所有关键参数集中管理（PipelineConfig）

架构:
  RefineLoop
    ├── brainstorm_keywords(spec)
    ├── PipelineCore.run(keywords)     ← 启动流水线
    │   ├── Search Workers  ──→  [rel_queue]
    │   ├── Relevance Workers ──→  [qual_queue]
    │   └── Quality Workers  ──→   SQLite
    ├── check_enough(store, spec)
    └── refine → 再跑 PipelineCore
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.filters import quality_filter_async, relevance_filter_async
from src.keywords import brainstorm_keywords_async, refine_keywords
from src.llm import LLM
from src.models import PaperMeta
from src.sources.openalex import OpenAlexSource
from src.store import PaperStore
from src.topic import TopicSpec

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════


@dataclass
class PipelineConfig:
    """流水线配置参数。"""

    # 搜索
    initial_depth: int = 20  # 每关键词初始Search depth（后续可由 LLM 动态调整）
    max_total_papers: int = 1000  # 数据库总论文数上限（达到即停所有搜索）
    max_quality_papers: int = 200  # quality passed最多多少 papers（达到即停）
    max_refine_rounds: int = 3  # 关键词反哺最多几轮
    refine_sample_size: int = 15  # 反哺时给 LLM 看多少 papers样本文

    # 并发
    relevance_workers: int = 20  # Relevance审核并发数
    quality_workers: int = 20  # 质量审核并发数

    # 显示
    verbose: bool = True  # 打印详细日志
    compact: bool = False  # 精简模式：用单行状态条代替逐条打印
    language: str = "zh-hans"  # "zh-hans", "zh-hant", "en"
    lang_instruction: str = ""  # 可选：外部传入的语言指令，会追加到 system_prompt


# ══════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════


@dataclass
class PipelineStats:
    """流水线统计。"""

    total_searched: int = 0
    relevanced: int = 0
    quality_passed: int = 0
    rejected: int = 0
    refine_rounds: int = 0
    errors: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 流水线编排器
# ══════════════════════════════════════════════════════════


class PipelineOrchestrator:
    """工厂流水线编排器。

    用法:
        pipe = PipelineOrchestrator(llm, store, spec, config)
        stats = await pipe.run_full()  # 含关键词反哺循环
        # 或手动精细控制:
        #   await pipe.start_stages()
        #   await pipe.feed_keywords(keywords)
        #   await pipe.wait_all()
    """

    def __init__(
        self,
        llm: LLM,
        spec: TopicSpec,
        config: Optional[PipelineConfig] = None,
    ):
        self.llm = llm
        self.spec = spec
        self.cfg = config or PipelineConfig()
        self.stats = PipelineStats()

        # 先建一个临时 session，澄清后移到正式目录
        self.session_dir = Path("output") / "_tmp"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.store = PaperStore(str(self.session_dir / "papers.db"))

        # 阶段间队列
        self._rel_queue: asyncio.Queue = asyncio.Queue()
        self._qual_queue: asyncio.Queue = asyncio.Queue()

        # worker 任务句柄（用于启动 / 停止）
        self._search_tasks: list[asyncio.Task] = []
        self._rel_tasks: list[asyncio.Task] = []
        self._qual_tasks: list[asyncio.Task] = []

        # 已通过论文标题（用于Relevance去重上下文）
        self._screened_titles: list[str] = []

        # 计数器（用于显示）
        self._rel_count = 0
        self._qual_count = 0

        # 记录关键词使用历史（避免反哺重复）
        self._keywords_used: list[str] = []

    # ══════════════════════════════════════════════════════
    # 入口: 全自动运行（含关键词反哺）
    # ══════════════════════════════════════════════════════

    async def run_full(self) -> PipelineStats:
        """全自动运行：头脑风暴 → 预热 → 反哺 → 主流水线。

        步骤:
           1. brainstorm → 初始关键词
           2. 预热: 每关键词搜 5 papers，不进审核
           3. 早期反哺: LLM 审查预热结果，调整/扩充关键词
           4. 主流水线: 用优化后关键词全量搜索 + 审核
           5. 检查 quality_passed 数量
           6. 不够 → 反哺 → 补搜 → 回到 5
        """
        # 初始化 session 目录（以研究问题命名）
        from datetime import datetime

        from src.session import sanitize_filename

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = sanitize_filename(self.spec.research_question)
        dir_name = f"{base}_{ts}"
        new_path = self.session_dir.parent / dir_name
        if new_path != self.session_dir:
            import shutil

            old_dir = self.session_dir
            old_db = old_dir / "papers.db"
            self.store.close()
            new_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(old_db), str(new_path / "papers.db"))
            shutil.rmtree(str(old_dir), ignore_errors=True)
            self.session_dir = new_path
            self.store = PaperStore(str(self.session_dir / "papers.db"))
            print(f"  📁 Session: {self.session_dir.name}/")

        # ══════════════════════════════════════════════════════
        # Phase 1: 头脑风暴 + 预热 + 早期反哺
        # ══════════════════════════════════════════════════════

        print("\n🔍 Brainstorming keywords...")
        keywords = await brainstorm_keywords_async(
            self.llm, self.spec, lang=self.cfg.language,
            lang_instruction=self.cfg.lang_instruction,
        )
        print(f"  Generated {len(keywords)}  initial keyword groups")
        self._keywords_used = list(keywords)

        # 预热: 每关键词只搜 5 papers，不进相关/质量审核
        print("\n📦 [Warm-up] Searching 5/keyword to survey the landscape")
        await self._run_with_keywords(keywords, max_per_keyword=5)
        print(f"  Warm-up complete,  {self.store.count()} papers")

        # 早期反哺
        print("\n🔄 [Early Refinement] Early refinement after warm-up...")
        warm_kw, warm_depth = await self._refine_keywords()
        if warm_kw:
            before = len(self._keywords_used)
            self._keywords_used.extend(
                [kw for kw in warm_kw if kw not in self._keywords_used]
            )
            added = len(self._keywords_used) - before
            print(f"  Added  {added}  keyword groups: {warm_kw}")
        else:
            print("  No adjustment needed")

        # ══════════════════════════════════════════════════════
        # Phase 2: 主流水线（全量搜索 + 审核）
        # ══════════════════════════════════════════════════════

        main_kw = list(set(self._keywords_used))
        print(
            f"\n📦 [Main Pipeline] {len(main_kw)}  keyword groups"
            f" (Search depth {self.cfg.initial_depth})..."
        )
        await self._run_with_keywords(main_kw)
        print("  ⏳ Waiting for review to complete...")
        await self._drain_pipeline()
        passed = self.store.count(status="quality_passed")
        print(
            f"  📊 Main pipeline complete: {self.store.count()} papers → quality passed {passed} papers"
        )

        # 检查上限
        if self.store.count() >= self.cfg.max_total_papers:
            total = self.store.count()
            print(
                f"\n⏹ Total paper limit reached ({total} ≥ {self.cfg.max_total_papers})"
            )
            return self.stats
        if passed >= self.cfg.max_quality_papers:
            print(
                f"\n⏹ Quality-passed limit reached ({passed} ≥ {self.cfg.max_quality_papers})"
            )
            return self.stats

        if passed >= self.spec.min_papers:
            print(f"\n✅ Quality-passed target met ({passed} ≥ {self.spec.min_papers})")
            return self.stats

        # ══════════════════════════════════════════════════════
        # Phase 3: 反哺循环（不足时补搜）
        # ══════════════════════════════════════════════════════

        for round_num in range(1, self.cfg.max_refine_rounds + 1):
            if self.store.count() >= self.cfg.max_total_papers:
                print(
                    f"  ⏹ Total paper limit reached"
                    f" ({self.store.count()} ≥ {self.cfg.max_total_papers})"
                )
                break
            qp = self.store.count(status="quality_passed")
            if qp >= self.cfg.max_quality_papers:
                print(
                    f"  ⏹ Quality-passed limit reached ({qp} ≥ {self.cfg.max_quality_papers})"
                )
                break
            print(f"\n📦 [Refinement#{round_num}] Reviewing search directions...")
            new_kw, kw_depth = await self._refine_keywords()

            if new_kw:
                depth = kw_depth or self.cfg.initial_depth
                print(f"  New keywords: {new_kw}  (Search depth: {depth})")
                self._keywords_used.extend(new_kw)
                self.stats.refine_rounds = round_num
                await self._run_with_keywords(new_kw, max_per_keyword=depth)
            else:
                # LLM 没有新方向，尝试用现有关键词扩大Search depth
                depth = self.cfg.initial_depth * 2
                print(f"  No new directions，expanding search depth to {depth}")
                self.stats.refine_rounds = round_num
                await self._run_with_keywords(
                    self._keywords_used,
                    max_per_keyword=depth,
                )

            print("  ⏳ Waiting for review to complete...")
            await self._drain_pipeline()

            passed = self.store.count(status="quality_passed")
            print(
                f"  📊 After refinement: {self.store.count()} papers → quality passed {passed} papers"
                f" (target {self.spec.min_papers})"
            )
            if passed >= self.spec.min_papers:
                print(
                    f"\n✅ Refinement successful！({passed} ≥ {self.spec.min_papers})"
                )
                break

        passed_final = self.store.count(status="quality_passed")
        if passed_final < self.spec.min_papers:
            print(
                f"\n⚠ Final:  quality passed {passed_final} papers < target {self.spec.min_papers}"
            )
        return self.stats

    async def _run_summarization(self):
        """对 quality_passed 论文运行结构化信息提取。"""
        from src.summarizer import batch_summarize_async

        print("\n📄 [Summary Extraction] Starting...")
        count = await batch_summarize_async(
            self.llm, self.store, self.spec, lang=self.cfg.language,
            lang_instruction=self.cfg.lang_instruction,
        )
        print(f"  ✅ Done: {count} papers")

    # ══════════════════════════════════════════════════════
    # 核心: 运行一 keyword groups通过流水线
    # ══════════════════════════════════════════════════════

    async def _run_with_keywords(
        self,
        keywords: list[str],
        max_per_keyword: int | None = None,
    ):
        """启动流水线，搜索一 keyword groups，完成后关闭搜索 worker。

        Args:
            keywords: 待搜索的关键词列表。
            max_per_keyword: 每个关键词最多搜多少 papers。None=用配置文件默认值。
        """
        self._ensure_stages_running()

        depth = max_per_keyword or self.cfg.initial_depth
        new_tasks = []
        for kw in keywords:
            t = asyncio.create_task(self._search_worker(kw, max_results=depth))
            self._search_tasks.append(t)
            new_tasks.append(t)

        await asyncio.gather(*new_tasks)

    def _ensure_stages_running(self):
        """确保流水线各阶段 worker 已启动（幂等）。"""
        if not self._rel_tasks:
            for i in range(self.cfg.relevance_workers):
                t = asyncio.create_task(self._rel_worker(f"R{i}"))
                self._rel_tasks.append(t)

        if not self._qual_tasks:
            for i in range(self.cfg.quality_workers):
                t = asyncio.create_task(self._qual_worker(f"Q{i}"))
                self._qual_tasks.append(t)

    async def _drain_pipeline(self):
        """等待所有队列消费完毕，停止所有 worker。"""
        # 停止搜索 worker（不需要了）
        for t in self._search_tasks:
            t.cancel()
        self._search_tasks.clear()

        # 等待Relevance队列消费完毕，通知 worker 停止
        await self._rel_queue.join()
        for _ in self._rel_tasks:
            await self._rel_queue.put(None)
        for t in self._rel_tasks:
            await t
        self._rel_tasks.clear()

        # 等待质量队列消费完毕，通知 worker 停止
        await self._qual_queue.join()
        for _ in self._qual_tasks:
            await self._qual_queue.put(None)
        for t in self._qual_tasks:
            await t
        self._qual_tasks.clear()

    # ══════════════════════════════════════════════════════
    # 关键词反哺
    # ══════════════════════════════════════════════════════

    def _check_enough(self) -> bool:
        """检查是否已搜索到足够论文。"""
        return self.store.count() >= self.spec.min_papers

    async def _refine_keywords(self) -> tuple[list[str], int]:
        """让 LLM 审查已有结果，返回新的关键词方向和Search depth。

        Returns:
            (new_keywords, max_per_keyword):
                new_keywords: 去重后的新关键词列表（空=No new directions）。
                max_per_keyword: LLM 建议的Search depth（默认 20）。
        """
        from src.context import build_keyword_refine_context

        context = build_keyword_refine_context(
            store=self.store,
            spec=self.spec,
            current_keywords=self._keywords_used,
            n_top_papers=5,
            n_sample_papers=self.cfg.refine_sample_size,
        )

        refinement = await asyncio.to_thread(
            refine_keywords,
            llm=self.llm,
            spec=self.spec,
            current_keywords=self._keywords_used,
            papers_summary=context,
            total_found=self.store.count(),
            lang=self.cfg.language,
            lang_instruction=self.cfg.lang_instruction,
        )

        if not refinement.should_continue:
            return [], 0

        # 去重：只返回没用过的关键词
        new_kw = [kw for kw in refinement.new_keywords if kw not in self._keywords_used]
        depth = refinement.max_results_per_keyword
        return new_kw, depth

    # ══════════════════════════════════════════════════════
    # 搜索 Worker
    # ══════════════════════════════════════════════════════

    async def _search_worker(self, keyword: str, max_results: int | None = None):
        """搜索单个关键词，逐 papers存入 SQLite 并放入Relevance队列。

        Args:
            keyword: 搜索关键词。
            max_results: 最大结果数。None=用配置文件默认值。
        """
        source = OpenAlexSource()
        filters = {}
        if self.spec.start_year:
            filters["from_publication_date"] = f"{self.spec.start_year}-01-01"
        if self.spec.end_year:
            filters["to_publication_date"] = f"{self.spec.end_year}-12-31"

        depth = max_results or self.cfg.initial_depth
        # 总量上限检查
        # 总量上限: 数据库总论文数不超过 max_total_papers
        if self.store.count() >= self.cfg.max_total_papers:
            return

        result = await source.search_async(
            query=keyword,
            max_results=depth,
            sort="relevance_score:desc",
            filters=filters if filters else None,
        )

        if not result.papers:
            print(f"  🔍 [{keyword}] → 0 papers")
            return

        for paper in result.papers:
            if self.store.count() >= self.cfg.max_total_papers:
                break
                continue
            # 去重：如果论文已存在（被其他关键词搜到过），跳过
            if self.store.get_paper(paper.id):
                continue
            self.stats.total_searched += 1
            self.store.insert_paper(paper, search_query=keyword)
            record = self.store.get_paper(paper.id)
            if record:
                await self._rel_queue.put(record)

        sample = result.papers[0].title
        print(f"  🔍 [{keyword}] → {len(result.papers)}  papers (e.g.: {sample}...)")

    # ══════════════════════════════════════════════════════
    # Relevance Worker
    # ══════════════════════════════════════════════════════

    async def _rel_worker(self, name: str):
        """消费Relevance队列 → LLM 评分 → 通过的放入质量队列。"""
        while True:
            paper = await self._rel_queue.get()
            if paper is None:
                self._rel_queue.task_done()
                return

            self._rel_count += 1
            try:
                score, reason = await relevance_filter_async(
                    self.llm,
                    paper,
                    self.spec,
                    self._screened_titles,
                    lang=self.cfg.language,
                    lang_instruction=self.cfg.lang_instruction,
                )
                if score >= 60:
                    self.store.update_status(
                        paper.id,
                        "relevanced",
                        relevance_score=score,
                        relevance_reason=reason,
                    )
                    self._screened_titles.append(paper.title)
                    self.stats.relevanced += 1
                    record = self.store.get_paper(paper.id)
                    if record:
                        await self._qual_queue.put(record)
                    tag = "✅"
                else:
                    self.store.update_status(
                        paper.id,
                        "rejected",
                        relevance_score=score,
                        relevance_reason=reason,
                    )
                    self.stats.rejected += 1
                    tag = "⏭"

                print(
                    f"  [{name}] Relevance  #{self._rel_count:>2d}"
                    f"  {tag} score={score:.1f}  {paper.title}"
                )
            except Exception as e:
                logger.error("Relevance error %s: %s", paper.id, e)
                self.stats.errors.append(f"relevance {paper.id}: {e}")
            finally:
                self._rel_queue.task_done()

    # ══════════════════════════════════════════════════════
    # 质量 Worker
    # ══════════════════════════════════════════════════════

    async def _qual_worker(self, name: str):
        """消费质量队列 → 构建画像 + LLM 打分 → 存入 SQLite。"""
        from src.paper_profile import build_paper_profile_async

        while True:
            paper = await self._qual_queue.get()
            if paper is None:
                self._qual_queue.task_done()
                return

            self._qual_count += 1
            try:
                meta = PaperMeta(
                    id=paper.id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    year=paper.year,
                    source=paper.source,
                )
                profile = await build_paper_profile_async(meta)
                cits = profile.get("cited_by_count", 0)
                venue = profile.get("venue_stats", {})
                vname = venue.get("name", "")[:20] if venue else "-"

                score, reason = await quality_filter_async(
                    self.llm,
                    paper,
                    profile,
                    self.spec,
                    lang=self.cfg.language,
                    lang_instruction=self.cfg.lang_instruction,
                )

                if score >= 60:
                    self.store.update_status(
                        paper.id,
                        "quality_passed",
                        quality_score=score,
                        quality_reason=reason,
                    )
                    self.stats.quality_passed += 1
                    tag = "✅"
                else:
                    self.store.update_status(
                        paper.id,
                        "rejected",
                        quality_score=score,
                        quality_reason=reason,
                    )
                    self.stats.rejected += 1
                    tag = "⏭"

                print(
                    f"  [{name}] Quality    #{self._qual_count:>2d}"
                    f"  {tag} score={score:.1f}"
                    f"  cit={cits:<4d}  {vname:20s}  {paper.title}"
                )
            except Exception as e:
                logger.error("Quality error %s: %s", paper.id, e)
                self.stats.errors.append(f"quality {paper.id}: {e}")
            finally:
                self._qual_queue.task_done()
