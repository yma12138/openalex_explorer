"""Paper profile — enrich a paper with author stats, venue metrics, etc.

同步/异步双模式:
  - build_paper_profile()           ← 同步兼容
  - build_paper_profile_async()     ← 异步原生（asyncio.gather 并行）
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from src.config import load_config
from src.http_client import HttpClient
from src.models import PaperMeta

logger = logging.getLogger(__name__)

API_BASE = "https://api.openalex.org"

# 认证参数缓存（mailto + api_key）
_AUTH_CACHE: dict = {}


def _auth_params() -> dict:
    global _AUTH_CACHE
    if not _AUTH_CACHE:
        cfg = load_config()
        params = {}
        if cfg.openalex_mailto:
            params["mailto"] = cfg.openalex_mailto
        if cfg.openalex_api_key:
            params["api_key"] = cfg.openalex_api_key
        _AUTH_CACHE = params
    return _AUTH_CACHE


def build_paper_profile(paper: PaperMeta) -> dict[str, Any]:
    """同步：构建论文画像（兼容旧调用者）。"""
    return asyncio.run(_build_profile_async(paper))


async def build_paper_profile_async(paper: PaperMeta) -> dict[str, Any]:
    """异步：构建论文画像，work/venue/authors 并行请求。"""
    return await _build_profile_async(paper)


# ══════════════════════════════════════════════════════
# 核心实现
# ══════════════════════════════════════════════════════


async def _build_profile_async(paper: PaperMeta) -> dict[str, Any]:
    """构建论文画像的核心实现（异步 + 并行）。"""
    profile: dict[str, Any] = {
        "paper_id": paper.id,
        "title": paper.title,
        "year": paper.year,
        "source": paper.source,
    }

    if paper.source != "openalex":
        return profile

    # ── Step 1: 获取论文元数据（work） ───────────────────────────
    work_data = await _fetch_work_async(paper.id)
    if work_data is None:
        return profile

    profile["cited_by_count"] = work_data.get("cited_by_count", 0)
    profile["publication_date"] = work_data.get("publication_date", "")
    profile["type"] = work_data.get("type", "")

    # 距今月数
    pub_date = work_data.get("publication_date", "")
    if pub_date:
        try:
            dt = datetime.strptime(pub_date[:10], "%Y-%m-%d")
            now = datetime.now()
            months = (now.year - dt.year) * 12 + now.month - dt.month
            profile["months_since_publication"] = months
        except ValueError:
            pass

    # ── Step 2+3: 并行获取 venue + authors ──────────────────────
    venue_id = work_data.get("venue_id", "")
    author_ids = work_data.get("author_ids", [])  # list of (name, openalex_id)

    tasks = []
    if venue_id:
        tasks.append(_fetch_venue_stats_async(venue_id))

    # 最多取前 10 个作者
    author_tasks = [_fetch_author_stats_async(aid) for _name, aid in author_ids[:10]]
    tasks.extend(author_tasks)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 第一个结果是 venue（如果有）
        idx = 0
        if venue_id:
            venue_result = results[0]
            if isinstance(venue_result, dict) and not isinstance(
                venue_result, BaseException
            ):
                profile["venue_stats"] = venue_result
            idx = 1

        # 剩余结果是 authors
        author_profiles = []
        for j, (name, _aid) in enumerate(author_ids[:10]):
            if idx + j < len(results):
                r = results[idx + j]
                if isinstance(r, dict) and not isinstance(r, BaseException):
                    author_profiles.append(r)
                else:
                    author_profiles.append(
                        {
                            "name": name,
                            "id": _aid,
                            "error": "fetch failed",
                        }
                    )
        if author_profiles:
            profile["authors"] = author_profiles

    return profile


# ══════════════════════════════════════════════════════
# 子请求
# ══════════════════════════════════════════════════════


async def _fetch_work_async(paper_id: str) -> Optional[dict]:
    """获取论文元数据（引用数、venue_id、作者列表）。"""
    url = f"{API_BASE}/works/{paper_id}"
    params = _auth_params()

    try:
        r = await HttpClient.get(url, params=params)
        if r.status_code != 200:
            logger.warning("Failed to fetch work %s: %s", paper_id, r.status_code)
            return None
        data = r.json()
        loc = data.get("primary_location") or {}
        src = loc.get("source") or {}
        venue_id = src.get("id", "")

        author_ids = []
        for au in data.get("authorships") or []:
            a = au.get("author") or {}
            aid = a.get("id", "")
            name = a.get("display_name", "")
            if aid and name:
                author_ids.append((name, aid))

        return {
            "cited_by_count": data.get("cited_by_count", 0),
            "publication_date": data.get("publication_date", ""),
            "type": data.get("type", ""),
            "venue_id": venue_id,
            "venue_name": src.get("display_name", ""),
            "venue_type": src.get("type", ""),
            "author_ids": author_ids,
        }
    except httpx.HTTPError as e:
        logger.warning("HTTP error fetching work %s: %s", paper_id, e)
        return None


async def _fetch_venue_stats_async(source_id: str) -> Optional[dict]:
    """获取期刊/会议统计指标。"""
    sid = source_id.rstrip("/").split("/")[-1]
    url = f"{API_BASE}/sources/{sid}"
    params = {
        "select": "id,display_name,type,summary_stats,works_count,cited_by_count",
    }
    params.update(_auth_params())

    try:
        r = await HttpClient.get(url, params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        ss = data.get("summary_stats") or {}
        return {
            "name": data.get("display_name", ""),
            "type": data.get("type", ""),
            "works_count": data.get("works_count", 0),
            "cited_by_count": data.get("cited_by_count", 0),
            "2yr_mean_citedness": ss.get("2yr_mean_citedness", 0),
            "h_index": ss.get("h_index", 0),
            "i10_index": ss.get("i10_index", 0),
        }
    except httpx.HTTPError as e:
        logger.warning("HTTP error fetching venue %s: %s", source_id, e)
        return None


async def _fetch_author_stats_async(author_id: str) -> Optional[dict]:
    """获取作者统计指标。"""
    aid = author_id.rstrip("/").split("/")[-1]
    url = f"{API_BASE}/authors/{aid}"
    params = {
        "select": (
            "id,display_name,orcid,works_count,cited_by_count,last_known_institutions"
        ),
    }
    params.update(_auth_params())

    try:
        r = await HttpClient.get(url, params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        wc = data.get("works_count", 0) or 1
        insts = data.get("last_known_institutions") or []
        return {
            "name": data.get("display_name", ""),
            "id": data.get("id", ""),
            "orcid": data.get("orcid", ""),
            "works_count": wc,
            "cited_by_count": data.get("cited_by_count", 0) or 0,
            "avg_citations_per_paper": round(
                (data.get("cited_by_count", 0) or 0) / wc, 1
            ),
            "institutions": [i.get("display_name", "") for i in insts[:3]],
        }
    except httpx.HTTPError as e:
        logger.warning("HTTP error fetching author %s: %s", author_id, e)
        return None
