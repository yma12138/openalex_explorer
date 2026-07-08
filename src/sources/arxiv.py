"""ArXiv paper source implementation."""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import httpx

from src.models import PaperMeta, SearchResult
from src.sources.base import PaperSource

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"


class ArXivSource(PaperSource):
    """Paper source that retrieves papers from ArXiv via its API."""

    def __init__(self, max_retries: int = 3):
        self.source_name = "arxiv"
        self.max_retries = max_retries

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """Search ArXiv API for papers."""
        if not query.strip():
            return SearchResult(
                papers=[], total_found=0, query=query, source=self.source_name
            )

        if max_results <= 0:
            return SearchResult(
                papers=[], total_found=0, query=query, source=self.source_name
            )

        safe_query = self._sanitize_query(query)

        params = {
            "search_query": f"all:{safe_query}",
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = httpx.get(ARXIV_API_BASE, params=params, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"ArXiv API request failed: {e}")
            return SearchResult(
                papers=[],
                total_found=0,
                query=query,
                source=self.source_name,
            )

        papers = self._parse_atom_response(response.text)
        return SearchResult(
            papers=papers[:max_results],
            total_found=len(papers),
            query=query,
            source=self.source_name,
        )

    def download_pdf(self, paper_id: str, output_dir: Path) -> Optional[Path]:
        """Download a PDF from ArXiv by paper ID.

        Args:
            paper_id: ArXiv ID (e.g., "1706.03762" or "1706.03762v1").
            output_dir: Directory to save the PDF.

        Returns:
            Path to downloaded PDF, or None on failure.
        """
        clean_id = re.sub(r"v\d+$", "", paper_id.strip())
        pdf_url = f"{ARXIV_PDF_BASE}/{clean_id}.pdf"
        output_path = output_dir / f"{paper_id}.pdf"

        if output_path.exists():
            logger.info(f"PDF already exists: {output_path}")
            return output_path

        output_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(self.max_retries):
            try:
                response = httpx.get(pdf_url, timeout=60.0, follow_redirects=True)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if (
                    "application/pdf" not in content_type
                    and not response.content[:5] == b"%PDF-"
                ):
                    logger.warning(
                        f"Response for {paper_id} is not a PDF "
                        f"(content-type: {content_type})"
                    )
                    if attempt < self.max_retries - 1:
                        continue
                    return None

                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.info(
                    f"Downloaded PDF: {output_path} ({len(response.content)} bytes)"
                )
                return output_path

            except httpx.HTTPError as e:
                logger.warning(
                    f"Download attempt {attempt + 1} failed for {paper_id}: {e}"
                )
                if attempt < self.max_retries - 1:
                    continue
                return None

        return None

    def _sanitize_query(self, query: str) -> str:
        """Remove or escape characters that break the ArXiv API."""
        safe = query.replace(":", " ").replace('"', " ").replace("&", " ")
        safe = re.sub(r"\s+", "+", safe.strip())
        return safe

    def _parse_atom_response(self, xml_text: str) -> list[PaperMeta]:
        """Parse ArXiv Atom XML response into PaperMeta list."""
        import xml.etree.ElementTree as ET

        papers = []
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv response XML: {e}")
            return []

        for entry in root.findall("atom:entry", ns):
            paper = self._parse_entry(entry, ns)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element, ns: dict) -> Optional[PaperMeta]:
        """Parse a single Atom entry into a PaperMeta."""
        try:
            id_tag = entry.find("atom:id", ns)
            full_id = id_tag.text if id_tag is not None else ""
            paper_id = full_id.strip().split("/")[-1] if full_id else ""

            title_tag = entry.find("atom:title", ns)
            title = self._clean_text(title_tag.text) if title_tag is not None else ""

            abstract_tag = entry.find("atom:summary", ns)
            abstract = (
                self._clean_text(abstract_tag.text) if abstract_tag is not None else ""
            )

            authors = []
            for author_elem in entry.findall("atom:author", ns):
                name_tag = author_elem.find("atom:name", ns)
                if name_tag is not None and name_tag.text:
                    authors.append(name_tag.text.strip())

            published_tag = entry.find("atom:published", ns)
            year = 0
            if published_tag is not None and published_tag.text:
                year_match = re.match(r"(\d{4})", published_tag.text.strip())
                if year_match:
                    year = int(year_match.group(1))

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

            pdf_url = pdf_url or f"{ARXIV_PDF_BASE}/{paper_id}.pdf"

            # Journal reference
            journal_tag = entry.find("arxiv:journal_ref", ns)
            journal_ref = (
                self._clean_text(journal_tag.text)
                if journal_tag is not None and journal_tag.text
                else None
            )

            # DOI
            doi_tag = entry.find("arxiv:doi", ns)
            doi = doi_tag.text.strip() if doi_tag is not None and doi_tag.text else None

            # Categories
            categories = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term")
                if term:
                    categories.append(term)

            return PaperMeta(
                id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                pdf_url=pdf_url,
                source=self.source_name,
                journal_ref=journal_ref,
                doi=doi,
                categories=categories if categories else None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse ArXiv entry: {e}")
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text from XML."""
        if not text:
            return ""
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
