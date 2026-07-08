"""Block 3: Keywords → OpenAlex → SQLite."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.search import search_keywords_to_store
from src.store import PaperStore
from src.topic import TopicSpec


@pytest.fixture
def store():
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    s = PaperStore(db.name)
    yield s
    s.close()
    Path(db.name).unlink(missing_ok=True)


SAMPLE_KEYWORDS = [
    ["transformer attention mechanism"],
    ["deep learning survey"],
    ["machine learning review"],
]

slow = pytest.mark.slow


class TestSearchToStore:
    """Integration: real OpenAlex API + SQLite."""

    @slow
    def test_search_stores_papers(self, store):
        """Basic search: papers go into store with status=searched."""
        kw = SAMPLE_KEYWORDS[0]
        print(f"\n  ▶ 搜索关键词: {kw[0]}")
        spec = TopicSpec(research_question="test")
        stats = search_keywords_to_store(
            keywords=kw,
            store=store,
            spec=spec,
            max_per_keyword=5,
        )
        print(
            f"  ✓ 找到 {stats['total_new']} 篇新论文, 累计 {stats['total_in_store']} 篇"
        )

        if stats["total_new"] == 0:
            pytest.skip("OpenAlex returned no results")
        assert stats["total_in_store"] > 0

        papers = store.list_papers(status="searched", limit=3)
        print(f"  ✓ 抽查前 {len(papers)} 篇记录 — 全部 status=searched")
        for p in papers:
            assert p.status == "searched"
            assert p.title
            assert p.year > 0
            print(f"    [{p.id}] {p.title[:60]}... ({p.year})")

    @slow
    def test_search_with_year_filter(self, store):
        """Year filter should restrict results."""
        kw = SAMPLE_KEYWORDS[1]
        print(f"\n  ▶ 搜索: {kw[0]} (限制 2023-2024)")
        spec = TopicSpec(research_question="test", start_year=2023, end_year=2024)
        stats = search_keywords_to_store(
            keywords=kw,
            store=store,
            spec=spec,
            max_per_keyword=5,
        )
        print(f"  ✓ 找到 {stats['total_in_store']} 篇")

        if stats["total_in_store"] == 0:
            pytest.skip("OpenAlex returned no results")
        papers = store.list_papers(status="searched", limit=10)
        years = set()
        for p in papers:
            assert 2023 <= p.year <= 2024, f"违规: {p.id} year={p.year}"
            years.add(p.year)
        print(f"  ✓ 全部在年份范围内, 分布: {sorted(years)}")

    @slow
    def test_dedup(self, store):
        """Same keyword searched twice should not add duplicate papers."""
        kw = SAMPLE_KEYWORDS[2]
        print(f"\n  ▶ 第一次搜索: {kw[0]}")
        stats1 = search_keywords_to_store(
            keywords=kw,
            store=store,
            spec=TopicSpec(research_question="test"),
            max_per_keyword=5,
        )
        first_count = stats1["total_in_store"]
        print(f"  ✓ 第一次: {first_count} 篇")
        if first_count == 0:
            pytest.skip("OpenAlex returned no results")

        print("  ▶ 第二次搜索 (相同关键词, 期望 0 新篇)")
        stats2 = search_keywords_to_store(
            keywords=kw,
            store=store,
            spec=TopicSpec(research_question="test"),
            max_per_keyword=5,
        )
        print(
            f"  ✓ 新增 {stats2['total_new']} 篇 (应为 0), 累计 {stats2['total_in_store']} 篇 (应等于 {first_count})"  # noqa: E501
        )
        assert stats2["total_new"] == 0
        assert stats2["total_in_store"] == first_count

    @slow
    def test_multiple_keywords(self, store):
        """Multiple keywords should find different papers."""
        print(f"\n  ▶ 使用 {len(SAMPLE_KEYWORDS)} 组不同关键词搜索")
        spec = TopicSpec(research_question="test")
        all_ids = set()
        final_stats = {"total_in_store": 0}
        for kw in SAMPLE_KEYWORDS:
            print(f"    搜索: {kw[0]}")
            stats = search_keywords_to_store(
                keywords=kw,
                store=store,
                spec=spec,
                max_per_keyword=3,
            )
            final_stats = stats
            papers = store.list_papers(status="searched", limit=50)
            ids = {p.id for p in papers}
            all_ids.update(ids)
            print(
                f"    → 当前去重 {len(all_ids)} 篇 / 累计 {stats['total_in_store']} 篇"
            )

        assert len(all_ids) >= final_stats["total_in_store"]
        max_possible = len(SAMPLE_KEYWORDS) * 3
        if final_stats["total_in_store"] > 0:
            assert len(all_ids) <= max_possible
        print(
            f"  ✓ 最终: {final_stats['total_in_store']} 篇存储, {len(all_ids)} 篇去重 (上限 {max_possible})"  # noqa: E501
        )


class TestSearchNonLLM:
    """Tests that don't need OpenAlex."""

    def test_empty_keywords(self, store):
        print("  ▶ 空关键词列表")
        stats = search_keywords_to_store(
            keywords=[], store=store, spec=TopicSpec(research_question="test")
        )
        assert stats["total_new"] == 0
        assert stats["total_in_store"] == 0
        print("  ✓ 正确返回 0 结果")
