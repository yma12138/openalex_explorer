"""Block 2: TopicSpec → brainstorm_keywords."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.keywords import KeywordSet, brainstorm_keywords
from src.llm import LLM
from src.topic import TopicSpec

slow = pytest.mark.slow


def _has_api_key() -> bool:
    """Check if llm_config.json exists and has an API key."""
    cfg_path = Path(__file__).resolve().parent.parent.parent / "llm_config.json"
    if not cfg_path.exists():
        return False
    import json
    with open(cfg_path) as f:
        return bool(json.load(f).get("api_key", ""))


@pytest.fixture(scope="module")
def llm():
    if not _has_api_key():
        pytest.skip("No LLM API key configured (create llm_config.json)")
    cfg_path = Path(__file__).resolve().parent.parent.parent / "llm_config.json"
    import json
    with open(cfg_path) as f:
        config = json.load(f)
    return LLM(config=config)


SPECS = [
    TopicSpec(
        research_question="绿色科技创新中的机构间合作模式研究综述",
        start_year=2020,
        end_year=2025,
        min_papers=20,
        method_preference="social network analysis",
        research_field="innovation management",
    ),
    TopicSpec(
        research_question="大语言模型的推理能力改进方法综述",
        start_year=2022,
        end_year=2026,
        min_papers=15,
    ),
    TopicSpec(
        research_question=(
            "A survey of privacy-preserving techniques in federated learning"
        ),
        start_year=2020,
        end_year=2025,
        min_papers=15,
        method_preference="differential privacy, secure multi-party computation",
    ),
]


class TestBrainstormKeywords:
    """Integration tests — require real LLM call."""

    @slow
    def test_returns_list_of_strings(self, llm):
        print(f"\n  ▶ 提问: {SPECS[0].research_question[:30]}...")
        keywords = brainstorm_keywords(llm, SPECS[0])
        print(f"  ✓ 生成 {len(keywords)} 组关键词:")
        for i, kw in enumerate(keywords, 1):
            print(f"    [{i}] {kw}")
        assert isinstance(keywords, list)
        assert 3 <= len(keywords) <= 8
        for kw in keywords:
            assert isinstance(kw, str)
            assert len(kw.split()) >= 2

    @slow
    def test_covers_different_aspects(self, llm):
        print(f"\n  ▶ 提问: {SPECS[1].research_question[:30]}...")
        keywords = brainstorm_keywords(llm, SPECS[1])
        first_words = {kw.split()[0].lower() for kw in keywords if kw.split()}
        print(
            f"  ✓ {len(keywords)} 组关键词, 首词多样性: {len(first_words)}/{len(keywords)}"  # noqa: E501
        )
        for kw in keywords:
            print(f"    - {kw}")
        assert len(first_words) >= 2, f"关键词过于相似: {keywords}"

    @slow
    def test_english_spec_produces_english_keywords(self, llm):
        print(f"\n  ▶ 英文提问: {SPECS[2].research_question[:40]}...")
        keywords = brainstorm_keywords(llm, SPECS[2])
        print(f"  ✓ {len(keywords)} 组关键词:")
        for kw in keywords:
            assert kw.isascii()
            print(f"    - {kw}")


class TestKeywordSetModel:
    """Test the KeywordSet Pydantic model."""

    def test_valid_keyword_set(self):
        ks = KeywordSet(keywords=[f"kw {i} group a" for i in range(5)])
        assert len(ks.keywords) == 5
        print("  ✓ KeywordSet 正常创建 (5 个关键词)")

    def test_too_few_keywords_rejected(self):
        with pytest.raises(ValueError):
            KeywordSet(keywords=["only one"])
        print("  ✓ 关键词 < 5 被正确拒绝")

    def test_many_keywords_accepted(self):
        """不再有上限，20 组关键词也应该通过。"""
        ks = KeywordSet(keywords=[f"kw {i} group" for i in range(20)])
        assert len(ks.keywords) == 20
        print("  ✓ 20 组关键词通过 (无上限限制)")
