"""Search flow — keywords → OpenAlex → SQLite.

同步/异步双模式:
  - search_keywords_to_store()           ← 同步
  - search_keywords_to_store_async()     ← 异步（多个关键词并行搜索）
"""

import asyncio
import logging

from src.sources.openalex import OpenAlexSource
from src.store import PaperStore
from src.topic import TopicSpec

logger = logging.getLogger(__name__)


def search_keywords_to_store(
    keywords: list[str],
    store: PaperStore,
    spec: TopicSpec,
    max_per_keyword: int = 20,
) -> dict[str, int]:
    """Search OpenAlex for each keyword and store results in SQLite.

    Args:
        keywords: List of search query strings.
        store: Initialized PaperStore instance.
        spec: TopicSpec for year filters etc.
        max_per_keyword: Max results per keyword search.

    Returns:
        Dict with stats: {"total_new": int, "total_in_store": int}
    """
    source = OpenAlexSource()
    total_new = 0

    filters = {}
    if spec.start_year:
        filters["from_publication_date"] = f"{spec.start_year}-01-01"
    if spec.end_year:
        filters["to_publication_date"] = f"{spec.end_year}-12-31"

    for kw in keywords:
        result = source.search(
            query=kw,
            max_results=max_per_keyword,
            sort="relevance_score:desc",
            filters=filters if filters else None,
        )

        if not result.papers:
            logger.info(f"Keyword {kw!r}: 0 results")
            continue

        before_kw = store.count()
        store.insert_papers(result.papers, search_query=kw)
        new_kw = store.count() - before_kw
        total_new += new_kw
        logger.info(f"Keyword {kw!r}: {len(result.papers)} found, {new_kw} new")

    return {"total_new": total_new, "total_in_store": store.count()}


async def search_keywords_to_store_async(
    keywords: list[str],
    store: PaperStore,
    spec: TopicSpec,
    max_per_keyword: int = 20,
) -> dict[str, int]:
    """异步搜索：多个关键词并行搜索 OpenAlex 并存入 SQLite。"""
    source = OpenAlexSource()

    filters = {}
    if spec.start_year:
        filters["from_publication_date"] = f"{spec.start_year}-01-01"
    if spec.end_year:
        filters["to_publication_date"] = f"{spec.end_year}-12-31"

    async def _search_one(kw: str) -> list:
        result = await source.search_async(
            query=kw,
            max_results=max_per_keyword,
            sort="relevance_score:desc",
            filters=filters if filters else None,
        )
        return result.papers

    results = await asyncio.gather(*[_search_one(kw) for kw in keywords])

    total_new = 0
    for kw, papers in zip(keywords, results):
        if not papers:
            logger.info("Keyword %r: 0 results", kw)
            continue
        before = store.count()
        store.insert_papers(papers, search_query=kw)
        added = store.count() - before
        total_new += added
        logger.info("Keyword %r: %d found, %d new", kw, len(papers), added)

    return {"total_new": total_new, "total_in_store": store.count()}
