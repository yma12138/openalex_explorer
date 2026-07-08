"""Tests for ContextInjector and build_keyword_refine_context."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.context import ContextInjector, build_keyword_refine_context
from src.store import PaperStore
from src.topic import TopicSpec


class TestContextInjector:
    """Test the general-purpose context injector."""

    def test_inject_basic(self):
        text = ContextInjector.inject(
            "问题: {question}\n年份: {year}",
            question="LLM推理",
            year=2024,
        )
        assert "问题: LLM推理" in text
        assert "年份: 2024" in text
        print("  ✓ inject 基本替换正确")

    def test_inject_none_becomes_placeholder(self):
        text = ContextInjector.inject("关键词: {kw}", kw=None)
        assert "(无)" in text
        print("  ✓ None 显示为 (无)")

    def test_inject_empty_list(self):
        text = ContextInjector.inject("列表: {items}", items=[])
        assert "(空)" in text
        print("  ✓ 空列表显示为 (空)")

    def test_inject_nonempty_list(self):
        text = ContextInjector.inject("关键词: {kw}", kw=["a", "b", "c"])
        assert "关键词: a, b, c" in text
        print("  ✓ 非空列表合并为逗号分隔")

    def test_build_section(self):
        section = ContextInjector.build_section(
            "标题:",
            ["行1", "行2", "行3"],
        )
        assert "标题:" in section
        assert "  行1" in section
        assert "  行3" in section
        print("  ✓ build_section 格式正确")

    def test_build_section_empty_lines(self):
        section = ContextInjector.build_section("标题:", [])
        assert section == ""
        print("  ✓ build_section 空行返回空字符串")


class TestBuildKeywordRefineContext:
    """Test building keyword refinement context with real store."""

    @pytest.fixture
    def store(self):
        db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db.close()
        s = PaperStore(db.name)
        yield s
        s.close()
        Path(db.name).unlink(missing_ok=True)

    @pytest.fixture
    def spec(self):
        return TopicSpec(
            research_question="测试问题",
            min_papers=10,
        )

    def test_empty_store(self, store, spec):
        """Empty store should not crash."""
        text = build_keyword_refine_context(
            store=store,
            spec=spec,
            current_keywords=["test"],
            n_top_papers=3,
            n_sample_papers=5,
        )
        assert "测试问题" in text
        assert "已使用关键词" in text
        assert "暂无论文数据" in text
        print("  ✓ 空库不崩溃")

    def test_with_inserted_papers(self, store, spec):
        """Store with papers should show paper details."""
        from src.models import PaperMeta

        for i in range(3):
            store.insert_paper(
                PaperMeta(
                    id=f"test-{i}",
                    title=f"Test Paper {i} About Machine Learning",
                    authors=["Author A"],
                    abstract=f"This is abstract {i} about deep learning.",
                    year=2024,
                    source="openalex",
                    keywords=["machine learning", "deep learning", "transformer"],
                    categories=["cs.AI", "cs.LG"],
                )
            )
            store.update_status(
                f"test-{i}",
                "quality_passed",
                relevance_score=8.0,
                quality_score=85.0,
            )

        text = build_keyword_refine_context(
            store=store,
            spec=spec,
            current_keywords=["machine learning", "deep learning"],
            n_top_papers=2,
            n_sample_papers=5,
        )

        assert "测试问题" in text
        assert "machine learning, deep learning" in text
        assert "Test Paper" in text
        assert "abstract" in text or "摘要" in text
        assert "transformer" in text  # from keywords
        print("  ✓ 有论文时上下文包含标题、摘要、关键词")
        print(f"\n  生成文本 ({len(text)} 字符):")
        for line in text.split("\n")[:8]:
            print(f"    {line}")

    def test_keywords_appear_in_context(self, store, spec):
        """Paper keywords should be visible in context."""
        from src.models import PaperMeta

        store.insert_paper(
            PaperMeta(
                id="kw-test-1",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                abstract="The dominant sequence transduction models...",
                year=2017,
                source="openalex",
                keywords=["transformer", "attention mechanism", "seq2seq"],
            )
        )
        store.update_status(
            "kw-test-1", "quality_passed", relevance_score=9.0, quality_score=95.0
        )

        text = build_keyword_refine_context(
            store=store,
            spec=spec,
            current_keywords=["transformer"],
        )

        assert "transformer" in text
        assert "attention mechanism" in text
        print("  ✓ 论文关键词出现在上下文中")
