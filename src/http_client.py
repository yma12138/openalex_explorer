"""全局异步 HTTP 客户端。

职责:
  - 持有 httpx.AsyncClient 单例（连接池复用）
  - asyncio.Semaphore 限流替代 time.sleep
  - 统一的日志和超时管理

用法:
    await HttpClient.request("GET", url, params={...})

生命周期:
    await HttpClient.init()   ← 应用启动时
    ...
    await HttpClient.close()  ← 应用关闭时
"""

import asyncio
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_CONCURRENT = 100


class HttpClient:
    """全局异步 HTTP 客户端。

    单例模式，全局共用连接池和限流信号量。

    Attributes:
        client: httpx.AsyncClient 实例。
        semaphore: 并发控制信号量。
    """

    client: Optional[httpx.AsyncClient] = None
    semaphore: Optional[asyncio.Semaphore] = None

    # ── 生命周期 ─────────────────────────────────────────────────

    @classmethod
    async def init(
        cls,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """初始化全局客户端。

        Args:
            max_concurrent: 最大并发请求数（默认 20）。
            timeout: 请求超时秒数（默认 30s）。
        """
        if cls.client is not None:
            logger.warning("HttpClient 重复初始化，先关闭旧实例")
            await cls.close()

        limits = httpx.Limits(
            max_keepalive_connections=max_concurrent,
            max_connections=max_concurrent * 2,
        )
        cls.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=limits,
        )
        cls.semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(
            f"HttpClient 初始化: max_concurrent={max_concurrent}, timeout={timeout}s"
        )

    @classmethod
    async def close(cls) -> None:
        """关闭全局客户端，释放连接池。"""
        if cls.client is not None:
            await cls.client.aclose()
            cls.client = None
            cls.semaphore = None
            logger.info("HttpClient 已关闭")

    @classmethod
    def is_ready(cls) -> bool:
        """客户端是否已初始化。"""
        return cls.client is not None and cls.semaphore is not None

    # ── 请求 API ─────────────────────────────────────────────────

    @classmethod
    async def request(
        cls,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """发起一个受信号量控制的 HTTP 请求。

        自动懒初始化：第一次调用时用默认参数初始化。

        Args:
            method: HTTP 方法（GET, POST, ...）。
            url: 请求 URL。
            **kwargs: 传给 httpx.AsyncClient.request 的参数。

        Returns:
            httpx.Response 对象。
        """
        if not cls.is_ready():
            logger.info("HttpClient 懒初始化")
            await cls.init()

        async with cls.semaphore:  # type: ignore[union-attr]
            response = await cls.client.request(  # type: ignore[union-attr]
                method, url, **kwargs
            )
            return response

    @classmethod
    async def get(cls, url: str, **kwargs: Any) -> httpx.Response:
        """GET 请求的快捷方法。"""
        return await cls.request("GET", url, **kwargs)

    @classmethod
    async def _ensure_ready(cls) -> None:
        """如果未初始化，用默认参数自动初始化。"""
        if not cls.is_ready():
            await cls.init()
