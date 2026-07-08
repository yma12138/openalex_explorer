"""Tests for PaperStore (SQLite-backed)."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import PaperMeta
from src.store import PaperStore


@pytest.fixture
def store():
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    s = PaperStore(db.name)
    yield s
    s.close()
    Path(db.name).unlink(missing_ok=True)


@pytest.fixture
def sample_paper():
    return PaperMeta(
        id="1706.03762",
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar"],
        abstract=(
            "The dominant sequence transduction models are based on"
            " attention mechanisms."
        ),
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        source="arxiv",
    )


class TestPaperStore:
    def test_insert_and_get(self, store, sample_paper):
        pid = store.insert_paper(sample_paper, search_query="transformer")
        assert pid == "1706.03762"

        record = store.get_paper("1706.03762")
        assert record is not None
        assert record.title == "Attention Is All You Need"
        assert record.authors == ["Vaswani", "Shazeer", "Parmar"]
        assert record.year == 2017
        assert record.status == "searched"
        assert record.search_query == "transformer"

    def test_get_nonexistent(self, store):
        record = store.get_paper("nonexistent")
        assert record is None

    def test_count(self, store, sample_paper):
        assert store.count() == 0
        store.insert_paper(sample_paper)
        assert store.count() == 1
        assert store.count(status="searched") == 1
        assert store.count(status="evaluated") == 0

    def test_duplicate_insert_preserves_state(self, store, sample_paper):
        store.insert_paper(sample_paper)
        store.update_status("1706.03762", "title_passed", relevance_score=4.0)

        # Insert again with updated metadata
        updated = PaperMeta(
            id="1706.03762",
            title="Attention Is All You Need (Updated)",
            authors=["Vaswani", "Shazeer", "Parmar"],
            abstract="Updated abstract.",
            year=2017,
            source="arxiv",
        )
        store.insert_paper(updated)

        # Pipeline state should be preserved
        record = store.get_paper("1706.03762")
        assert record.status == "title_passed"
        assert record.relevance_score == 4.0
        # But metadata should update
        assert record.title == "Attention Is All You Need (Updated)"

    def test_list_by_status(self, store, sample_paper):
        store.insert_paper(sample_paper)
        p2 = PaperMeta(
            id="2104.11502",
            title="Learning to Cluster Faces",
            authors=["Ye"],
            abstract="Face clustering.",
            year=2021,
            source="arxiv",
        )
        store.insert_paper(p2)
        store.update_status("2104.11502", "abstract_passed", relevance_score=4.5)

        discovered = store.list_papers(status="searched")
        assert len(discovered) == 1
        assert discovered[0].id == "1706.03762"

        passed = store.list_papers(status="abstract_passed")
        assert len(passed) == 1
        assert passed[0].id == "2104.11502"

    def test_update_status_with_extra(self, store, sample_paper):
        store.insert_paper(sample_paper)
        store.update_status(
            "1706.03762",
            "pdf_downloaded",
            relevance_score=4.5,
            pdf_path="output/1706.03762.pdf",
            quality_score=85.0,
        )

        record = store.get_paper("1706.03762")
        assert record.status == "pdf_downloaded"
        assert record.relevance_score == 4.5
        assert record.pdf_path == "output/1706.03762.pdf"
        assert record.quality_score == 85.0

    def test_list_sorted_by_relevance(self, store):
        papers = [
            PaperMeta(
                id=f"test-{i}",
                title=f"Paper {i}",
                authors=[],
                abstract="",
                year=2020,
                source="arxiv",
            )
            for i in range(3)
        ]
        for p in papers:
            store.insert_paper(p)

        store.update_status("test-0", "searched", relevance_score=3.0)
        store.update_status("test-1", "searched", relevance_score=5.0)
        store.update_status("test-2", "searched", relevance_score=1.0)

        listed = store.list_papers(limit=3)
        assert listed[0].relevance_score == 5.0
        assert listed[1].relevance_score == 3.0
        assert listed[2].relevance_score == 1.0
