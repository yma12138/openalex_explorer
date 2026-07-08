"""Tests for PaperSource abstract interface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import PaperMeta
from src.sources.base import PaperSource, SearchResult


class DummySource(PaperSource):
    """Minimal PaperSource implementation for testing."""

    def __init__(self):
        self.search_called = False
        self.download_called = False

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        self.search_called = True
        papers = [
            PaperMeta(
                id="test-001",
                title="Test Paper",
                authors=["Alice", "Bob"],
                abstract="This is a test abstract.",
                year=2024,
                source="dummy",
            )
        ]
        return SearchResult(papers=papers, total_found=1, query=query, source="dummy")

    def download_pdf(self, paper_id: str, output_dir: Path) -> Path:
        self.download_called = True
        return output_dir / f"{paper_id}.pdf"


class TestPaperSourceInterface:
    """Test that PaperSource contract works correctly."""

    def test_search_returns_search_result(self):
        source = DummySource()
        result = source.search("test query", max_results=5)
        assert isinstance(result, SearchResult)
        assert len(result.papers) == 1
        assert result.total_found == 1
        assert result.query == "test query"
        assert source.search_called

    def test_download_pdf_returns_path(self, tmp_path):
        source = DummySource()
        path = source.download_pdf("test-001", tmp_path)
        assert path == tmp_path / "test-001.pdf"
        assert source.download_called

    def test_papers_have_required_fields(self):
        source = DummySource()
        result = source.search("test")
        paper = result.papers[0]
        assert paper.id
        assert paper.title
        assert paper.authors
        assert paper.abstract
        assert paper.year > 0
        assert paper.source == "dummy"
