"""Tests for ArXivSource."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.sources.arxiv import ArXivSource


class TestArXivSource:
    """Test ArXiv source. Integration tests require network."""

    def test_init(self):
        source = ArXivSource()
        assert source.source_name == "arxiv"
        assert source.max_retries == 3

    def test_search_empty_query(self):
        source = ArXivSource()
        result = source.search("", max_results=5)
        assert len(result.papers) == 0

    def test_search_zero_max_results(self):
        source = ArXivSource()
        result = source.search("machine learning", max_results=0)
        assert len(result.papers) == 0

    @pytest.mark.slow
    def test_search_real_query(self):
        """Integration test: search ArXiv for a real query."""
        source = ArXivSource()
        result = source.search("stock market prediction transformer", max_results=3)
        assert len(result.papers) > 0
        assert result.total_found > 0
        assert result.source == "arxiv"

        paper = result.papers[0]
        assert paper.id
        assert paper.title
        assert paper.abstract
        assert paper.year > 0

    @pytest.mark.slow
    def test_download_pdf_real(self, tmp_path):
        """Integration test: download a real ArXiv PDF."""
        source = ArXivSource()
        # Use a known paper: 1706.03762 (Attention Is All You Need)
        pdf_path = source.download_pdf("1706.03762", tmp_path)
        assert pdf_path is not None
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        # Should be larger than a minimal PDF header
        assert pdf_path.stat().st_size > 1000
