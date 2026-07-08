"""上下文注入 — 按需组装文本注入到 agent prompt 中。

提供:
  - ContextInjector.inject():       通用方法，将上下文数据填入模板
  - build_keyword_refine_context(): 关键词反哺专用，注入已通过论文的摘要+关键词
"""

from dataclasses import dataclass
from typing import Any

from src.store import PaperStore
from src.topic import TopicSpec

# ══════════════════════════════════════════════════════════
# 通用上下文注入器
# ══════════════════════════════════════════════════════════


class ContextInjector:
    """上下文注入器 — 将结构化数据按模板组装成文本。

    用法:
        ctx = ContextInjector()
        text = ctx.inject(
            "研究问题: {question}\n已有关键词: {keywords}",
            question="...", keywords="a, b, c",
        )
    """

    @staticmethod
    def inject(template: str, **kwargs: Any) -> str:
        """将 kwargs 中的值填入模板占位符。

        占位符格式: {name}
        自动处理 None 和空列表的显示。

        Args:
            template: 含 {placeholder} 的模板字符串。
            **kwargs: 要填入的值。

        Returns:
            填充后的文本。
        """
        formatted = {}
        for key, value in kwargs.items():
            if value is None:
                formatted[key] = "(无)"
            elif isinstance(value, list):
                if not value:
                    formatted[key] = "(空)"
                else:
                    formatted[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, bool):
                formatted[key] = "是" if value else "否"
            else:
                formatted[key] = str(value)
        return template.format(**formatted)

    @staticmethod
    def build_section(
        title: str,
        lines: list[str],
        indent: str = "  ",
    ) -> str:
        """构建一个带标题的文本段落。

        Args:
            title: 段落标题。
            lines: 内容行列表。
            indent: 每行缩进。

        Returns:
            格式化的段落文本。
        """
        if not lines:
            return ""
        body = "\n".join(f"{indent}{line}" for line in lines)
        return f"{title}\n{body}"


# ══════════════════════════════════════════════════════════
# 关键词反哺专用上下文
# ══════════════════════════════════════════════════════════


@dataclass
class KeywordRefineContext:
    """关键词反哺所需的结构化上下文数据。"""

    # 研究问题
    research_question: str = ""
    # 已有论文总数
    total_found: int = 0
    # 目标数
    target: int = 0
    # 已使用的关键词
    current_keywords: list[str] = None  # type: ignore[assignment]
    # 已通过的相关论文（标题 + 摘要 + 关键词）
    passed_papers: list[dict] = None  # type: ignore[assignment]


def build_keyword_refine_context(
    store: PaperStore,
    spec: TopicSpec,
    current_keywords: list[str],
    n_top_papers: int = 5,
    n_sample_papers: int = 15,
) -> str:
    """构建关键词反哺上下文：注入高质量已通过论文 + 现有关键词。

    从两个来源获取论文:
      1. quality_passed 论文（前 n_top_papers 篇，含完整摘要+关键词）
      2. searched 论文（前 n_sample_papers 篇，仅标题，用于覆盖面分析）

    Args:
        store: PaperStore 实例。
        spec: TopicSpec。
        current_keywords: 已使用过的关键词列表。
        n_top_papers: 展示多少篇已通过论文的详细信息。
        n_sample_papers: 展示多少篇搜索到论文的标题。

    Returns:
        格式化后的上下文文本，可直接注入到 refine_keywords 的 prompt。
    """

    # 获取质量分数最高的论文（用于展示摘要+关键词）
    # 不按状态过滤，直接用 quality_score 排名，确保即使还没到 quality 阶段也有内容可看
    top_papers = store.list_papers(
        limit=n_top_papers,
        sort_by="quality",
    )

    # 获取非拒绝论文（用于覆盖面分析）
    # 不限制状态，因为流水线排空后 searched/relevanced/quality_passed 都可能存在
    sample_papers = store.list_papers(
        limit=n_sample_papers,
        sort_by="relevance",
    )

    sections = []

    # ── 头部信息 ────────────────────────────────────────────
    header = ContextInjector.inject(
        "研究问题: {question}\n"
        "已搜索论文: {total} 篇 (目标: {target} 篇)\n"
        "已使用关键词: {keywords}\n",
        question=spec.research_question,
        total=store.count(),
        target=spec.min_papers,
        keywords=current_keywords,
    )
    sections.append(header)

    # ── 高质量论文详情（摘要+关键词） ──────────────────────
    if top_papers:
        paper_lines = []
        for i, p in enumerate(top_papers, 1):
            kw_str = ", ".join(p.keywords[:8]) if p.keywords else "(无)"
            abstract_preview = (
                p.abstract[:200].replace("\n", " ") if p.abstract else "(无摘要)"
            )
            paper_lines.append(
                f"[{i}] {p.title[:70]}\n"
                f"    关键词: {kw_str}\n"
                f"    摘要: {abstract_preview}..."
            )
        sections.append(
            ContextInjector.build_section(
                "\n✅ 质量最高的论文（参考其主题和关键词）:",
                paper_lines,
            )
        )
    else:
        sections.append("\n✅ 暂无论文数据。")

    # ── 搜索到的论文概览（仅标题） ────────────────────────
    if sample_papers:
        title_lines = [
            f"[{i + 1}] {p.id}: {p.title[:80]}" for i, p in enumerate(sample_papers)
        ]
        sections.append(
            ContextInjector.build_section(
                "\n📄 已搜索到的论文（查看当前覆盖范围）:",
                title_lines,
            )
        )

    # ── 结束指令 ────────────────────────────────────────────
    sections.append(
        "\n请分析以上信息，判断是否需要新的搜索关键词来覆盖未涉及的方向。\n"
        "如果当前关键词已经覆盖了主要方向，或者没有明显的新方向可搜，"
        "请将 should_continue 设为 false。"
    )

    return "\n".join(sections)
