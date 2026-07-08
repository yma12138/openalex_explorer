"""综述撰写编排器 — 分类 → 归入 → 梳理 → 撰写 → 检查 → 补搜。"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.llm import LLM
from src.prompts import load_prompt
from src.review.models import (
    PaperInfo,
    ReviewDraft,
    Section,
    ThemeAnalysis,
    ThemeInfo,
)
from src.store import PaperStore
from src.topic import TopicSpec


def _with_instruction(system: str, lang_instruction: str = "") -> str:
    """如果提供了 lang_instruction，将其追加到 system prompt 前面。"""
    return lang_instruction + "\n\n" + system if lang_instruction else system


def _with_style(prompt_name: str, lang: str = "zh-hans", lang_instruction: str = "") -> str:
    """将风格规范追加到指定 prompt 后面。"""
    base = load_prompt(prompt_name, lang) + "\n\n" + load_prompt("review_style", lang)
    return _with_instruction(base, lang_instruction)


def _user_prompt(template_name: str, lang: str = "zh-hans", **kwargs) -> str:
    """加载用户 prompt 模板并填充 {placeholder}。"""
    template = load_prompt(template_name, lang)
    return template.format(**kwargs)


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# Pydantic 模型（LLM 结构化输出）
# ══════════════════════════════════════════════════════════


class ThemeOutput(BaseModel):
    """LLM 输出的Theme classification。"""

    themes: list[dict] = Field(
        description="List of themes, each with name, description, paper_indices",
    )


class RelationshipOutput(BaseModel):
    """Theme relationship analysis."""

    relationship_type: str = Field(
        description="Type: evolution / contrast / complement / parallel"
    )
    description: str = Field(
        description="Detailed description of relationships between papers"
    )


class CoverageOutput(BaseModel):
    """Coverage check result."""

    theme: str = Field(description="Theme name")
    sufficient: bool = Field(description="Whether coverage is sufficient")
    missing_aspects: str = Field(description="Missing aspects")


# ══════════════════════════════════════════════════════════
# 综述编排器
# ══════════════════════════════════════════════════════════


class ReviewOrchestrator:
    """综述撰写编排器。

    流程:
      1. classify() — 将论文分为 3-6 个主题
      2. assign()   — 每 papers归入对应主题
      3. analyze()  — 主题内Relationship analysis
      4. write()    — 逐节Writing review
      5. check()    — 覆盖检查，发现不足
      6. supplement() — 自动补搜 + 回到步骤 2
    """

    def __init__(
        self,
        llm: LLM,
        store: PaperStore,
        spec: TopicSpec,
        output_dir: Path,
        lang: str = "zh-hans",
        lang_instruction: str = "",
    ):
        self.llm = llm
        self.store = store
        self.spec = spec
        self.lang = lang
        self.lang_instruction = lang_instruction
        self.output_dir = output_dir
        self.papers: list[PaperInfo] = []
        self.themes: list[ThemeInfo] = []
        self.assignments: list[dict] = []  # {paper_id, theme, rationale}
        self.analyses: list[ThemeAnalysis] = []
        self.draft: Optional[ReviewDraft] = None

    async def run(self) -> Optional[ReviewDraft]:
        """全流程运行。"""
        await self._load_papers()

        # Step 1-4: 主流程
        await self._classify()
        await self._analyze()
        await self._write()

        # Saving
        await self._save()

        return self.draft

    async def _load_papers(self):
        """从 SQLite Loaded quality_passed 论文。"""
        records = self.store.list_papers(
            status="quality_passed",
            limit=100,
            sort_by="quality",
        )
        # 按标题去重，保留最新的一 papers
        seen: dict[str, PaperInfo] = {}
        for p in records:
            norm = p.title.strip().lower()
            if norm in seen:
                old = seen[norm]
                if p.year < old.year:
                    seen[norm] = PaperInfo(
                        id=p.id,
                        title=p.title,
                        abstract=p.abstract,
                        keywords=", ".join(p.keywords[:10]) if p.keywords else "",
                        year=p.year,
                        authors=", ".join(p.authors[:5]),
                    )
            else:
                seen[norm] = PaperInfo(
                    id=p.id,
                    title=p.title,
                    abstract=p.abstract,
                    keywords=", ".join(p.keywords[:10]) if p.keywords else "",
                    year=p.year,
                    authors=", ".join(p.authors[:5]),
                )
        self.papers = list(seen.values())
        self.papers.sort(key=lambda x: x.year, reverse=True)
        print(
            f"\n📄 Loaded {len(self.papers)} papers quality_passed papers (deduplicated by title, kept earliest)"
        )
        for i, p in enumerate(self.papers, 1):
            print(f"  [{i}] {p.title[:70]}")

    # ══════════════════════════════════════════════════════
    # Step 1: 分类
    # ══════════════════════════════════════════════════════

    async def _classify(self):
        """将论文分为 3-6 个主题维度。"""
        print(f"\n{'=' * 60}")
        print("  Step 1: Theme classification")
        print(f"{'=' * 60}")

        # 给 LLM 看有序号的论文列表
        papers_text = "\n".join(
            f"[{i + 1}] {p.title} ({p.year})\n"
            f"    关键词: {p.keywords[:100]}\n"
            f"    Abstract: {p.abstract[:300]}"
            for i, p in enumerate(self.papers)
        )

        prompt = _user_prompt(
            "review_classify_user",
            self.lang,
            research_question=self.spec.research_question,
            papers_text=papers_text,
        )

        result: ThemeOutput = self.llm.structured(
            prompt,
            output_type=ThemeOutput,
            system_prompt=_with_instruction(load_prompt("review_classify", self.lang), self.lang_instruction),
        )

        # 将序号映射回真实论文 ID（1-based → 0-based → paper）
        index_to_paper = {i + 1: p for i, p in enumerate(self.papers)}

        self.themes = []
        self.assignments = []
        used: set[int] = set()

        for t in result.themes:
            name = (t.get("name") or "").strip()
            desc = (t.get("description") or "").strip()
            indices = t.get("paper_indices") or []

            # 去重：只取有效且未被其他主题用过的序号
            valid = sorted(
                set(
                    i
                    for i in indices
                    if isinstance(i, int)
                    and 1 <= i <= len(self.papers)
                    and i not in used
                )
            )
            if not name or len(valid) < 2:
                continue

            paper_ids = [index_to_paper[i].id for i in valid]
            self.themes.append(
                ThemeInfo(name=name, description=desc, paper_ids=paper_ids)
            )
            used.update(valid)

            print(f"  📂 {name} ({len(paper_ids)}  papers): {desc[:60]}")
            for pid in paper_ids:
                self.assignments.append(
                    {
                        "paper_id": pid,
                        "theme": name,
                        "rationale": f"Assigned by LLM to category「{name}」",
                    }
                )

        # 未被任何类别覆盖的论文 → Auto-assigned第一个主题
        covered = {a["paper_id"] for a in self.assignments}
        for p in self.papers:
            if p.id not in covered:
                fallback = self.themes[0].name if self.themes else "未分类"
                self.assignments.append(
                    {
                        "paper_id": p.id,
                        "theme": fallback,
                        "rationale": "Auto-assigned（LLM 未覆盖）",
                    }
                )
                print(f"  📌 {p.title[:55]:55s} → {fallback} (Auto-assigned)")

    # ══════════════════════════════════════════════════════
    # Step 3: 主题内Relationship analysis
    # ══════════════════════════════════════════════════════

    async def _analyze(self):
        """并行分析各主题内论文关系。"""
        print(f"\n{'=' * 60}")
        print("  Step 3: Relationship analysis")
        print(f"{'=' * 60}")

        system = _with_instruction(load_prompt("review_relationship", self.lang), self.lang_instruction)

        async def _analyze_one(theme):
            ids = theme.paper_ids
            theme_papers = [p for p in self.papers if p.id in ids]
            if not theme_papers:
                return None

            papers_text = "\n".join(
                f"[{i + 1}] {p.title} ({p.year}): {p.abstract[:200]}"
                for i, p in enumerate(theme_papers)
            )

            prompt = _user_prompt(
                "review_relationship_user",
                self.lang,
                theme_name=theme.name,
                theme_desc=theme.description,
                papers_text=papers_text,
            )
            result: RelationshipOutput = self.llm.structured(
                prompt,
                output_type=RelationshipOutput,
                system_prompt=system,
            )
            return ThemeAnalysis(
                theme=theme.name,
                relationship=result.relationship_type,
                analysis=result.description,
            )

        results = [_analyze_one(t) for t in self.themes]
        gathered = await asyncio.gather(*results)
        self.analyses = [r for r in gathered if r is not None]

        for a in self.analyses:
            print(f"  🔗 {a.theme}: {a.relationship}")
            # 打印具体论文间关系
            for line in a.analysis.split("\n"):
                line = line.strip()
                if line:
                    print(f"       {line}")

    # ══════════════════════════════════════════════════════
    # Step 4: Writing review
    # ══════════════════════════════════════════════════════

    async def _write(self):
        """逐节Writing review。"""
        print(f"\n{'=' * 60}")
        print("  Step 4: Writing review")
        print(f"{'=' * 60}")

        sections = []

        # 建立全局引用编号（论文 → 序号）
        ref_map = {p.id: i + 1 for i, p in enumerate(self.papers)}

        # 并行写各主题章节
        async def _write_section(theme):
            theme_papers = [p for p in self.papers if p.id in theme.paper_ids]
            if not theme_papers:
                return None

            # 用全局引用编号传递论文
            papers_text = "\n\n".join(
                f"[{ref_map[p.id]}] {p.title} ({p.year})\n"
                f"作者: {p.authors}\nAbstract: {p.abstract}"
                for p in theme_papers
            )

            rel = next(
                (a.analysis for a in self.analyses if a.theme == theme.name),
                "",
            )

            content = self.llm.invoke(
                _user_prompt(
                    "review_section_user",
                    self.lang,
                    research_question=self.spec.research_question,
                    theme_name=theme.name,
                    theme_desc=theme.description,
                    relationship=rel,
                    papers_text=papers_text,
                ),
                system_prompt=_with_style("review_section", self.lang, self.lang_instruction),
            )

            return Section(
                title=theme.name,
                content=content,
                citations=[p.id for p in theme_papers],
            )

        tasks = [_write_section(t) for t in self.themes]
        results = await asyncio.gather(*tasks)
        sections = [r for r in results if r is not None]
        for s in sections:
            print(f"  ✍️ {s.title} ({len(s.content)}  chars)")

        # 写Introduction
        themes_overview = "\n".join(f"- {t.name}: {t.description}" for t in self.themes)
        intro = self.llm.invoke(
            _user_prompt(
                "review_intro_user",
                self.lang,
                research_question=self.spec.research_question,
                themes_overview=themes_overview,
            ),
            system_prompt=_with_style("review_intro", self.lang, self.lang_instruction),
        )
        print(f"  ✍️ Introduction ({len(intro)} chars)")

        sections_summary = "\n".join(
            f"## {s.title}\n{s.content[:200]}..." for s in sections
        )
        conclusion = self.llm.invoke(
            _user_prompt(
                "review_conclusion_user",
                self.lang,
                research_question=self.spec.research_question,
                sections_summary=sections_summary,
            ),
            system_prompt=_with_style("review_conclusion", self.lang, self.lang_instruction),
        )
        print(f"  ✍️ Conclusion ({len(conclusion)} chars)")

        sections_text = (
            "\n".join(f"- {s.title}: {s.content[:200]}..." for s in sections)
            if sections
            else "(No theme sections)"
        )
        title = f"{self.spec.research_question} Review"
        abstract = self.llm.invoke(
            _user_prompt(
                "review_abstract_user",
                self.lang,
                title=title,
                intro_preview=intro[:300] + "...",
                sections_preview=sections_text,
                conclusion_preview=conclusion[:300] + "...",
            ),
            system_prompt=_with_style("review_intro", self.lang, self.lang_instruction),
        )
        print(f"  ✍️ Abstract ({len(abstract)} chars)")

        # 组装
        refs = [
            {"id": p.id, "title": p.title, "authors": p.authors, "year": p.year}
            for p in self.papers
        ]

        self.draft = ReviewDraft(
            title=f"{self.spec.research_question} Review",
            abstract=abstract,
            introduction=intro,
            sections=sections,
            conclusion=conclusion,
            references=refs,
        )

        # 验证：检查所有文献是否在正文中被引用
        self._verify_citations()

    def _verify_citations(self):
        """检查每 papers论文是否在正文中被引用。"""
        if not self.draft:
            return
        body = self.draft.introduction
        for s in self.draft.sections:
            body += "\n" + s.content
        body += "\n" + self.draft.conclusion

        import re

        cited = set()
        for m in re.finditer(r"\[(\d+(?:\s*[,;:-]\s*\d+)*)\]", body):
            for part in re.split(r"[,;]\s*", m.group(1)):
                part = part.strip()
                if ":" in part or "-" in part or "–" in part:
                    sep = "–" if "–" in part else (":" if ":" in part else "-")
                    a, b = part.split(sep)
                    for i in range(int(a.strip()), int(b.strip()) + 1):
                        cited.add(i)
                else:
                    cited.add(int(part))

        uncited = [i + 1 for i, _ in enumerate(self.papers) if (i + 1) not in cited]
        if uncited:
            for n in uncited:
                p = self.papers[n - 1]
                print(f"  ⚠ Uncited: [{n}] {p.title[:60]} → Auto-inserted")

                # 找到这 papers论文所属的章节
                target_section = self.draft.sections[0] if self.draft.sections else None
                for a in self.assignments:
                    if a["paper_id"] == p.id:
                        for s in self.draft.sections:
                            if s.title == a["theme"]:
                                target_section = s
                                break
                        break

                if target_section:
                    target_section.content += _user_prompt(
                        "review_citation_line",
                        self.lang,
                        number=n,
                    )
        else:
            print("  ✅ All papers are cited")

    # ══════════════════════════════════════════════════════
    # Saving
    # ══════════════════════════════════════════════════════

    async def _save(self):
        """Saving综述到文件。"""
        if not self.draft:
            return

        lines = [
            f"# {self.draft.title}\n",
            "## Abstract\n",
            self.draft.abstract + "\n" if self.draft.abstract else "（待生成）\n",
            "## Introduction\n",
            self.draft.introduction,
        ]
        for section in self.draft.sections:
            lines.append(f"\n## {section.title}\n")
            lines.append(section.content)

        lines.append("\n## Conclusion and Future Work\n")
        lines.append(self.draft.conclusion)

        lines.append("\n## References\n")
        for i, ref in enumerate(self.draft.references, 1):
            lines.append(f"[{i}] {ref['authors']} ({ref['year']}). {ref['title']}.")

        text = "\n".join(lines)
        path = self.output_dir / "review.md"
        path.write_text(text, encoding="utf-8")
        print(f"\n📝 Review Saving: {path}")
