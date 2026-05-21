"""Unit tests for AIService wrapper (retries, cache, semaphore)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from src.config import Settings
from src.core.exceptions import AIServiceError
from src.services.ai_service import AIService


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    base = dict(
        llm_provider="offline",
        embedding_provider="offline",
        ai_max_retries=2,
        ai_timeout_seconds=5.0,
        ai_concurrency_limit=4,
        image_storage_dir="/tmp/test_images",
        database_url="sqlite+aiosqlite:///./test.db",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def ai_service(fake_vlm, fake_embedder):
    settings = _make_settings()
    return AIService(settings=settings, vlm=fake_vlm, embedder=fake_embedder)


# ---------------------------------------------------------------------------
# describe_item
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_describe_item_returns_item_description(ai_service, sample_image):
    from ai.schemas import ItemDescription
    result = await ai_service.describe_item(sample_image, "test text")
    assert isinstance(result, ItemDescription)
    assert result.object_class == "umbrella"


@pytest.mark.asyncio
async def test_describe_item_passes_vlm(ai_service, fake_vlm, sample_image):
    await ai_service.describe_item(sample_image, "hello")
    assert len(fake_vlm.calls) == 1
    assert fake_vlm.calls[0][0] == sample_image


# ---------------------------------------------------------------------------
# embed + cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_returns_numpy_array(ai_service):
    vec = await ai_service.embed("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


@pytest.mark.asyncio
async def test_embed_cache_hit(ai_service, fake_embedder):
    await ai_service.embed("same text")
    await ai_service.embed("same text")
    # FakeEmbedder.embed should be called only once (second is from cache)
    # We verify via cache size
    assert len(ai_service._embed_cache) == 1


@pytest.mark.asyncio
async def test_embed_cache_different_texts(ai_service):
    await ai_service.embed("alpha")
    await ai_service.embed("beta")
    assert len(ai_service._embed_cache) == 2


@pytest.mark.asyncio
async def test_clear_cache(ai_service):
    await ai_service.embed("hello")
    assert len(ai_service._embed_cache) == 1
    ai_service.clear_cache()
    assert len(ai_service._embed_cache) == 0


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retries_on_provider_error(fake_embedder):
    """Service should retry and eventually succeed."""
    from ai.providers.base import ProviderError
    settings = _make_settings(ai_max_retries=2, ai_timeout_seconds=5.0)

    call_count = 0

    def flaky_embed(text, *, embedder=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderError("transient error")
        return fake_embedder.embed(text)

    service = AIService(settings=settings, embedder=fake_embedder)
    with patch("src.services.ai_service.ai.embed", side_effect=flaky_embed):
        result = await service.embed("test")
    assert isinstance(result, np.ndarray)
    assert call_count == 3


@pytest.mark.asyncio
async def test_raises_ai_service_error_after_exhausted_retries(fake_embedder):
    """All retries fail → AIServiceError raised."""
    from ai.providers.base import ProviderError
    settings = _make_settings(ai_max_retries=1, ai_timeout_seconds=5.0)

    def always_fail(text, *, embedder=None):
        raise ProviderError("always fails")

    service = AIService(settings=settings, embedder=fake_embedder)
    with patch("src.services.ai_service.ai.embed", side_effect=always_fail):
        with pytest.raises(AIServiceError):
            await service.embed("test")


# ---------------------------------------------------------------------------
# top_k / cosine (synchronous wrappers)
# ---------------------------------------------------------------------------

def test_top_k_wrapper(ai_service):
    q = np.array([1.0, 0.0], dtype=np.float32)
    cands = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    results = ai_service.top_k(q, cands, k=2)
    assert results[0].candidate_id == 0
    assert results[0].score > results[1].score


def test_cosine_wrapper(ai_service):
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(ai_service.cosine(a, b) - 1.0) < 1e-5
