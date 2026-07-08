"""Tests for OpenAlexSource."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sources.openalex import OpenAlexSource


class TestOpenAlex:
    def test_init(self):
        source = OpenAlexSource()
        assert source.source_name == "openalex"
        assert source.max_retries == 2
        print("  ✓ OpenAlexSource 初始化正确")

    @pytest.mark.slow
    def test_fetch_impact_found(self):
        source = OpenAlexSource()
        print("\n  ▶ 查询 ResNet (1512.03385) 的 OpenAlex 数据")
        impact = source.fetch_impact("1512.03385")
        if impact is None:
            pytest.skip("OpenAlex returned no data")
        print(f"  ✓ 标题: {impact.title}")
        print(f"  ✓ 引用: {impact.cited_by_count} 次")
        print(f"  ✓ 期刊: {impact.venue} ({impact.venue_type})")
        print(f"  ✓ 主题: {impact.topics}")
        assert impact.title
        assert "Deep Residual" in impact.title

    @pytest.mark.slow
    def test_fetch_impact_not_found(self):
        source = OpenAlexSource()
        impact = source.fetch_impact("9999.99999")
        assert impact is None
        print("  ✓ 不存在的论文返回 None")

    def test_fetch_impacts(self):
        source = OpenAlexSource()
        results = source.fetch_impacts(["1512.03385"])
        assert isinstance(results, dict)
        print(f"  ✓ fetch_impacts 返回 dict ({len(results)} 条)")

    @pytest.mark.slow
    def test_search_returns_results(self):
        source = OpenAlexSource()
        print('\n  ▶ 搜索 "machine learning transformer"')
        result = source.search("machine learning transformer", max_results=3)
        if len(result.papers) == 0:
            pytest.skip("OpenAlex returned no results")
        paper = result.papers[0]
        print(f"  ✓ 结果 1: {paper.title[:60]}...")
        print(f"  ✓ 作者: {paper.authors[:3]}...")
        print(f"  ✓ 年份: {paper.year}")
        assert paper.title

    @pytest.mark.slow
    def test_fetch_impact_has_citations_and_topics(self):
        source = OpenAlexSource()
        print("\n  ▶ 查询 ResNet 引用和主题")
        impact = source.fetch_impact("1512.03385")
        if impact is None:
            pytest.skip("OpenAlex returned no data")
        print(f"  ✓ 引用数: {impact.cited_by_count}")
        print(f"  ✓ 主题: {impact.topics}")
        assert impact.cited_by_count > 1000
        assert impact.topics and len(impact.topics) > 0
