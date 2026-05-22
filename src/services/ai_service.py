from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
from typing import Any

import numpy as np

import ai
from ai.providers.base import ProviderError
from ai.schemas import ItemDescription

from src.core.exceptions import AIServiceError
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class AIService:

    def __init__(
        self,
        settings: Settings | None = None,
        vlm: Any = None,
        embedder: Any = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._vlm = vlm          # injected provider (None → use env-selected default)
        self._embedder = embedder  # injected embedder (None → use env-selected default)
        self._semaphore = asyncio.Semaphore(self._cfg.ai_concurrency_limit)
        self._embed_cache: dict[str, np.ndarray] = {}
        logger.debug(
            "AIService ready (concurrency=%d, timeout=%.1fs, retries=%d)",
            self._cfg.ai_concurrency_limit,
            self._cfg.ai_timeout_seconds,
            self._cfg.ai_max_retries,
        )

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def describe_item(self, image_path: str, user_text: str) -> ItemDescription:

        logger.info("describe_item: %s | text=%r", image_path, user_text[:60])
        return await self._with_retry(
            "describe_item",
            functools.partial(
                ai.describe_item, image_path, user_text, vlm=self._vlm
            ),
        )

    async def embed(self, text: str) -> np.ndarray:

        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self._embed_cache:
            logger.debug("embed cache hit for key %s", cache_key[:8])
            return self._embed_cache[cache_key]

        logger.debug("embed: %r…", text[:60])
        vec = await self._with_retry(
            "embed",
            functools.partial(ai.embed, text, embedder=self._embedder),
        )
        self._embed_cache[cache_key] = vec
        return vec

    def top_k(
        self,
        query: np.ndarray,
        candidates: list[np.ndarray],
        k: int = 3,
    ) -> list[Any]:
        """Synchronous wrapper — similarity is pure NumPy, no I/O."""
        return ai.top_k(query, candidates, k=k)

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Synchronous wrapper."""
        return ai.cosine(a, b)

    def clear_cache(self) -> None:
        """Purge the embedding cache (useful in tests)."""
        self._embed_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _with_retry(self, name: str, fn: Any) -> Any:
        loop = asyncio.get_running_loop()
        last_exc: Exception | None = None
        max_retries = self._cfg.ai_max_retries
        timeout = self._cfg.ai_timeout_seconds

        for attempt in range(max_retries + 1):
            async with self._semaphore:
                try:
                    t0 = time.perf_counter()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, fn),
                        timeout=timeout,
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info("%s OK in %.2fs (attempt %d)", name, elapsed, attempt + 1)
                    return result

                except asyncio.TimeoutError as exc:
                    last_exc = exc
                    logger.warning(
                        "%s timed out after %.1fs (attempt %d/%d)",
                        name, timeout, attempt + 1, max_retries + 1,
                    )

                except (ProviderError, ValueError) as exc:
                    last_exc = exc
                    logger.warning(
                        "%s failed (attempt %d/%d): %s",
                        name, attempt + 1, max_retries + 1, exc,
                    )

                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.error(
                        "%s unexpected error (attempt %d/%d): %s",
                        name, attempt + 1, max_retries + 1, exc,
                        exc_info=True,
                    )

            if attempt < max_retries:
                backoff = 0.5 * (2 ** attempt)
                logger.debug("%s back-off %.1fs before retry", name, backoff)
                await asyncio.sleep(backoff)

        raise AIServiceError(
            f"{name} failed after {max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc
