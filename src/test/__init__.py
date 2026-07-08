"""Test suite for the literature review agent.

每个测试文件对应一个独立板块（block），按正向流顺序命名：
  test_keywords.py       Block 1 — TopicSpec → 搜索关键词
  test_search.py         Block 2 — 关键词 → OpenAlex → SQLite
  test_search.py         Block 3 — 关键词 → OpenAlex → SQLite
  test_refine_loop.py    Block 4 — 关键词反哺 → 再搜索 → 够数为止
  test_pipeline.py       Block 5 — 以上全部串联（LangGraph）

需要 API Key 的测试用 @pytest.mark.skipif 跳过。
"""
