"""Abstract interface for paper source providers.

All paper retrieval sources (ArXiv, Semantic Scholar, etc.)
implement the PaperSource abstract base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.models import SearchResult


class PaperSource(ABC):
    """Abstract interface for academic paper retrieval."""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> SearchResult:
        """Search for papers matching the given query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            SearchResult containing matched papers.
        """

    @abstractmethod
    def download_pdf(self, paper_id: str, output_dir: Path) -> Optional[Path]:
        """Download the PDF for a given paper.

        Args:
            paper_id: Unique identifier of the paper.
            output_dir: Directory to save the PDF file.

        Returns:
            Path to the downloaded PDF file, or None if download failed.
        """
