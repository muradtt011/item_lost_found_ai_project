
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.providers.base import EmbeddingProvider, VLMProvider

from src.api import app
from src.config import get_settings
from src.core.matching_service import MatchingService
from src.services.ai_service import AIService
from src.storage.repository import InMemoryRepository


class FakeVLM(VLMProvider):
    """Returns a fixed JSON response. No network."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "object_class": "umbrella",
            "colors": ["black"],
            "brand": "Fulton",
            "distinguishing_marks": ["bent rib"],
            "location_hints": ["library entrance"],
            "confidence": 0.85,
        }
        self.calls: list[tuple[str, str]] = []

    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        self.calls.append((image_path, prompt))
        return json.dumps(self.payload)


class FakeEmbedder(EmbeddingProvider):
    """Deterministic toy embedder: 8-D unit vectors derived from a hash."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            text = "empty-description"
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


@pytest.fixture
def fake_vlm() -> FakeVLM:
    return FakeVLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_image(tmp_path):
    """A tiny but valid PNG file."""
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )
    p = tmp_path / "tiny.png"
    p.write_bytes(png_bytes)
    return str(p)


@pytest.fixture
def client(fake_vlm, fake_embedder):
    """FastAPI test client using fake AI providers, no live API calls."""
    settings = get_settings()
    repo = InMemoryRepository()
    ai_service = AIService(settings=settings, vlm=fake_vlm, embedder=fake_embedder)

    with TestClient(app) as test_client:
        app.state.matching = MatchingService(
            repository=repo,
            ai_service=ai_service,
            settings=settings,
        )
        app.state.repo = repo
        yield test_client

    if hasattr(app.state, "matching"):
        delattr(app.state, "matching")
    if hasattr(app.state, "repo"):
        delattr(app.state, "repo")