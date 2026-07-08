"""OpenAlex API wrapper — primary paper source.

Provides: title, abstract, authors, year, PDF URL,
venue, citation count, topics, keywords.

同步/异步双模式:
  - search() / fetch_impact()       ← 同步兼容（旧调用者不变）
  - search_async() / fetch_impact_async()  ← 异步原生（用 HttpClient）
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from src.http_client import HttpClient
from src.models import PaperMeta, SearchResult

logger = logging.getLogger(__name__)

API_BASE = "https://api.openalex.org"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"


@dataclass
class OpenAlexImpact:
    """Impact metrics for a paper from OpenAlex."""

    paper_id: str
    title: str = ""
    cited_by_count: int = 0
    venue: str = ""
    venue_type: str = ""
    publication_date: str = ""
    is_retracted: bool = False
    is_oa: bool = False
    language: str = ""
    paper_type: str = ""
    topics: list[str] | None = None
    keywords: list[str] | None = None


SEARCH_FIELDS = (
    "id,title,authorships,publication_date,type,language,"
    "primary_location,open_access,best_oa_location,"
    "cited_by_count,topics,keywords,abstract_inverted_index"
)


class OpenAlexSource:
    """Primary paper source using OpenAlex API.

    Can search by keywords, return full metadata including
    citation counts and PDF URLs.

    Args:
        mailto: Optional email for OpenAlex polite pool (~100 req/s).
        api_key: Optional API key for higher rate limits.
        max_retries: Max retries on failure.
    """

    def __init__(
        self,
        mailto: str = "",
        api_key: str = "",
        max_retries: int = 2,
    ):
        from src.config import load_config

        cfg = load_config()
        self.source_name = "openalex"
        self.max_retries = max_retries
        self._mailto = mailto or cfg.openalex_mailto
        self._api_key = api_key or cfg.openalex_api_key

    def _auth_params(self) -> dict:
        """Return auth params (mailto + api_key) if configured."""
        params = {}
        if self._mailto:
            params["mailto"] = self._mailto
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    # ═══════════════════════════════════════════════════
    # 搜索
    # ═══════════════════════════════════════════════════

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance_score:desc",
        filters: Optional[dict[str, str]] = None,
    ) -> SearchResult:
        """同步搜索（兼容旧调用者）。"""
        return asyncio.run(self._search_async(query, max_results, sort, filters))

    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance_score:desc",
        filters: Optional[dict[str, str]] = None,
    ) -> SearchResult:
        """异步搜索 — 用 HttpClient（连接池 + 信号量限流）。"""
        return await self._search_async(query, max_results, sort, filters)

    async def _search_async(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance_score:desc",
        filters: Optional[dict[str, str]] = None,
    ) -> SearchResult:
        """搜索的核心实现（异步）。"""
        params: dict = {
            "search": query,
            "per_page": min(max_results, 50),
            "sort": sort,
            "select": SEARCH_FIELDS,
        }
        params.update(self._auth_params())
        if filters:
            params["filter"] = ",".join(f"{k}:{v}" for k, v in filters.items())

        # 重试循环：429 时指数退避
        raw_results: list = []
        for attempt in range(4):
            if attempt > 0:
                delay = 0.2 * (2 ** (attempt - 1))  # 0.2s, 0.4s, 0.8s
                logger.warning(f"OpenAlex 429 — retry {attempt} after {delay:.0f}s")
                await asyncio.sleep(delay)

            try:
                logger.debug(
                    "OpenAlex search: %s... auth=mailto:%s api_key:%s",
                    params.get("search", "")[:50],
                    self._mailto or "none",
                    "✓" if self._api_key else "✗",
                )
                response = await HttpClient.get(
                    f"{API_BASE}/works",
                    params=params,
                )
                if response.status_code == 429:
                    continue
                response.raise_for_status()
                raw_results = response.json().get("results", [])
                break
            except httpx.HTTPError as e:
                logger.warning(
                    "OpenAlex search failed (attempt %d): %s", attempt + 1, e
                )
                break

        if not raw_results:
            return SearchResult(
                papers=[],
                total_found=0,
                query=query,
                source=self.source_name,
            )

        papers = []
        for raw in raw_results[:max_results]:
            paper = self._raw_to_papermeta(raw)
            if paper:
                papers.append(paper)

        return SearchResult(
            papers=papers,
            total_found=len(papers),
            query=query,
            source=self.source_name,
        )

    # ═══════════════════════════════════════════════════
    # 引用/影响力数据
    # ═══════════════════════════════════════════════════

    def fetch_impact(self, arxiv_id: str) -> Optional[OpenAlexImpact]:
        """同步获取引用数据。"""
        return asyncio.run(self._fetch_impact_async(arxiv_id))

    async def fetch_impact_async(self, arxiv_id: str) -> Optional[OpenAlexImpact]:
        """异步获取引用数据 — 用 HttpClient。"""
        return await self._fetch_impact_async(arxiv_id)

    async def _fetch_impact_async(self, arxiv_id: str) -> Optional[OpenAlexImpact]:
        """获取引用数据的核心实现（异步）。"""
        doi = f"10.48550/arxiv.{arxiv_id}"
        url = f"{API_BASE}/works/doi:{doi}"
        params = self._auth_params()

        for attempt in range(self.max_retries):
            try:
                response = await HttpClient.get(url, params=params)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return self._raw_to_impact(arxiv_id, response.json())
            except httpx.HTTPError as e:
                logger.warning(
                    "OpenAlex fetch_impact failed (attempt %d): %s", attempt + 1, e
                )
                if attempt < self.max_retries - 1:
                    continue
                return None
        return None

    def fetch_impacts(
        self, arxiv_ids: list[str]
    ) -> dict[str, Optional[OpenAlexImpact]]:
        return {pid: self.fetch_impact(pid) for pid in arxiv_ids}

    # ---- PDF download (delegate to ArXiv) ----

    def download_pdf(self, paper_id: str, output_dir: Path) -> Optional[Path]:
        """Download PDF by paper ID.

        Falls back to ArXiv PDF URL if no direct OA URL available.
        """
        from src.sources.arxiv import ArXivSource

        return ArXivSource().download_pdf(paper_id, output_dir)

    # ---- Parsing ----

    def _raw_to_papermeta(self, raw: dict) -> Optional[PaperMeta]:
        """Convert OpenAlex raw result to PaperMeta."""
        try:
            # ID
            oa_id = raw.get("id", "")
            paper_id = oa_id.split("/")[-1] if oa_id else ""

            # Title
            title = raw.get("title", "")

            # Abstract (inverted index → text)
            abstract = self._inverted_index_to_text(raw.get("abstract_inverted_index"))

            # Authors
            from src.models import Author as AuthorM

            authors = []
            author_details = []
            for au in raw.get("authorships") or []:
                author_data = au.get("author") or {}
                name = author_data.get("display_name", "")
                if name:
                    authors.append(name)
                    author_details.append(
                        AuthorM(
                            name=name,
                            position=au.get("author_position", ""),
                            is_corresponding=au.get("is_corresponding", False),
                        )
                    )

            # Year
            pub_date = raw.get("publication_date", "")
            year = 0
            if pub_date:
                year = int(pub_date[:4])

            # PDF URL
            pdf_url = None
            bol = raw.get("best_oa_location") or {}
            if bol.get("pdf_url"):
                pdf_url = bol["pdf_url"]
            if not pdf_url:
                oa = raw.get("open_access") or {}
                if oa.get("oa_url"):
                    pdf_url = oa["oa_url"]

            # Categories (from topics)
            topics = raw.get("topics") or []
            categories = [t["display_name"] for t in topics[:5]]

            # Keywords (from keywords)
            kw_list = raw.get("keywords") or []
            keywords = [k["display_name"] for k in kw_list[:10]]

            # Venue (stored as journal_ref)
            loc = raw.get("primary_location") or {}
            src = loc.get("source") or {}
            journal_ref = src.get("display_name", "")

            return PaperMeta(
                id=paper_id,
                title=title,
                authors=authors,
                author_details=author_details,
                abstract=abstract,
                year=year,
                pdf_url=pdf_url,
                source=self.source_name,
                journal_ref=journal_ref or None,
                categories=categories or None,
                keywords=keywords or None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse OpenAlex result: {e}")
            return None

    def _raw_to_impact(self, arxiv_id: str, data: dict) -> OpenAlexImpact:
        loc = data.get("primary_location") or {}
        src = loc.get("source") or {}
        topics = data.get("topics") or []
        keywords = data.get("keywords") or []
        oa = data.get("open_access") or {}

        return OpenAlexImpact(
            paper_id=arxiv_id,
            title=data.get("title", ""),
            cited_by_count=data.get("cited_by_count", 0) or 0,
            venue=src.get("display_name", ""),
            venue_type=src.get("type", ""),
            publication_date=data.get("publication_date", ""),
            is_retracted=data.get("is_retracted", False),
            is_oa=oa.get("is_oa", False),
            language=data.get("language", ""),
            paper_type=data.get("type", ""),
            topics=[t["display_name"] for t in topics[:5]],
            keywords=[k["display_name"] for k in keywords[:5]],
        )

    @staticmethod
    def _inverted_index_to_text(
        index: Optional[dict],
    ) -> str:
        """Convert OpenAlex's abstract_inverted_index to plain text.

        OpenAlex stores abstract as a dict mapping word → [positions].
        E.g. {"the": [0, 5], "cat": [1], ...}
        """
        if not index:
            return ""
        word_by_pos: dict[int, str] = {}
        for word, positions in index.items():
            if isinstance(positions, list):
                for pos in positions:
                    word_by_pos[pos] = word
        if not word_by_pos:
            return ""
        max_pos = max(word_by_pos.keys())
        words = [word_by_pos.get(i, "") for i in range(max_pos + 1)]
        return " ".join(words).strip()
