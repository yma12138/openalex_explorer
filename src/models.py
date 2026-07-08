"""Data models for the literature review agent."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Author:
    """Author of a paper with position metadata."""

    name: str
    position: str = ""  # "first", "middle", "last", ""
    is_corresponding: bool = False
    orcid: Optional[str] = None


@dataclass
class PaperMeta:
    """Metadata for a single academic paper."""

    id: str  # Unique identifier (e.g., ArXiv ID)
    title: str  # Paper title
    authors: list[str]  # Author names only (backward compatible)
    abstract: str  # Paper abstract
    year: int  # Publication year
    author_details: list[Author] = field(default_factory=list)
    pdf_url: Optional[str] = None  # Direct PDF URL
    source: str = ""  # Source name, e.g., "arxiv"
    journal_ref: Optional[str] = None  # Journal reference (if published)
    doi: Optional[str] = None  # Digital Object Identifier
    categories: list[str] | None = None  # arXiv categories, e.g. ["cs.CV", "cs.AI"]
    keywords: list[str] | None = (
        None  # OpenAlex keywords, e.g. ["transformer", "attention"]
    )
    relevance_score: float = 0.0  # Relevance score (0-10), filled by relevance_filter
    quality_score: float = 0.0  # Quality score (0-100), filled later
    markdown_content: Optional[str] = None  # PDF parsed to Markdown


@dataclass
class SearchResult:
    """Result of a paper search."""

    papers: list[PaperMeta]
    total_found: int
    query: str
    source: str


@dataclass
class PaperSummary:
    """结构化论文摘要 — 从论文中提取的关键信息。"""

    research_question: str = ""  # 论文试图解决什么问题
    method: str = ""  # 使用的技术路线
    main_findings: list[str] = field(default_factory=list)  # 核心结论1-3条
    limitations: str = ""  # 作者提到的不足
    relevance_to_topic: str = ""  # 对综述的价值


@dataclass
class ParseResult:
    """Result of parsing a PDF to Markdown."""

    paper_id: str
    markdown: str
    num_pages: int
    error: Optional[str] = None
