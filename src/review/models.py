"""综述撰写数据模型。"""

from dataclasses import dataclass, field


@dataclass
class PaperInfo:
    """论文摘要信息（供 LLM 使用）。"""

    id: str
    title: str
    abstract: str
    keywords: str  # 逗号分隔
    year: int
    authors: str  # 逗号分隔


@dataclass
class ThemeInfo:
    """主题维度信息。"""

    name: str
    description: str
    paper_ids: list[str] = field(default_factory=list)


@dataclass
class PaperAssignment:
    """论文到主题的归属。"""

    paper_id: str
    theme: str
    rationale: str  # 归类依据


@dataclass
class ThemeAnalysis:
    """主题分析结果。"""

    theme: str
    relationship: str  # 演进/对比/互补描述
    analysis: str  # 综合分析段落


@dataclass
class Section:
    """综述章节。"""

    title: str
    content: str
    citations: list[str] = field(default_factory=list)  # 论文ID列表


@dataclass
class ReviewDraft:
    """完整综述草稿。"""

    title: str = ""
    abstract: str = ""
    introduction: str = ""
    sections: list[Section] = field(default_factory=list)
    conclusion: str = ""
    references: list[dict] = field(default_factory=list)


@dataclass
class CoverageGap:
    """覆盖不足的主题。"""

    theme: str
    missing_aspects: str
    suggested_keywords: list[str] = field(default_factory=list)
