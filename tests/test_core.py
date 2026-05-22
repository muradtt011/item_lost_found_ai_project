

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.config import Settings
from src.core.exceptions import (
    AIServiceError,
    ImageValidationError,
    ItemNotFoundError,
    MatchingError,
)
from src.core.matching_service import MatchingService, _build_reason
from src.core.validation import validate_image, validate_user_text
from src.models import ItemRecord, ItemStatus
from src.storage.repository import InMemoryRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**kw) -> Settings:
    defaults = dict(
        llm_provider="offline",
        embedding_provider="offline",
        max_image_size_mb=1.0,
        ai_max_retries=0,
        ai_timeout_seconds=5.0,
        ai_concurrency_limit=2,
        image_storage_dir="/tmp/test_store",
        database_url="sqlite+aiosqlite:///./test.db",
    )
    defaults.update(kw)
    return Settings(**defaults)


def _make_png(tmp_path: Path, name: str = "test.png") -> Path:
    """Create a minimal valid 1×1 PNG."""
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )
    p = tmp_path / name
    p.write_bytes(png_bytes)
    return p


def _make_jpeg(tmp_path: Path, name: str = "test.jpg") -> Path:
    """Create a minimal JPEG (magic bytes + minimal body)."""
    # FF D8 FF E0 ... minimal valid-ish JPEG
    jpeg_bytes = bytes.fromhex("ffd8ffe000104a46494600010100000100010000")
    p = tmp_path / name
    p.write_bytes(jpeg_bytes)
    return p


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:
    def test_valid_png_accepted(self, tmp_path):
        p = _make_png(tmp_path)
        result = validate_image(p, settings=_make_settings())
        assert result == p.resolve()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ImageValidationError, match="does not exist"):
            validate_image(tmp_path / "nonexistent.png", settings=_make_settings())

    def test_wrong_extension_raises(self, tmp_path):
        p = tmp_path / "image.gif"
        p.write_bytes(b"GIF89a")
        with pytest.raises(ImageValidationError, match="Unsupported"):
            validate_image(p, settings=_make_settings())

    def test_oversized_file_raises(self, tmp_path):
        p = tmp_path / "big.png"
        # Write enough PNG-like bytes to exceed 1 MB limit
        png_header = bytes.fromhex("89504e470d0a1a0a")
        p.write_bytes(png_header + b"\x00" * (1024 * 1024 + 1))
        with pytest.raises(ImageValidationError, match="exceeds limit"):
            validate_image(p, settings=_make_settings(max_image_size_mb=1.0))

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image(p, settings=_make_settings())

    def test_bad_png_magic_raises(self, tmp_path):
        p = tmp_path / "fake.png"
        p.write_bytes(b"NOTAPNG" + b"\x00" * 30)
        with pytest.raises(ImageValidationError, match="magic"):
            validate_image(p, settings=_make_settings())

    def test_valid_jpeg_accepted(self, tmp_path):
        p = _make_jpeg(tmp_path)
        result = validate_image(p, settings=_make_settings())
        assert result == p.resolve()

    def test_bad_jpeg_magic_raises(self, tmp_path):
        p = tmp_path / "fake.jpg"
        p.write_bytes(b"NOTAJPEG" + b"\x00" * 30)
        with pytest.raises(ImageValidationError, match="magic"):
            validate_image(p, settings=_make_settings())


# ---------------------------------------------------------------------------
# validate_user_text
# ---------------------------------------------------------------------------

class TestValidateUserText:
    def test_strips_whitespace(self):
        assert validate_user_text("  hello  ") == "hello"

    def test_empty_is_allowed(self):
        assert validate_user_text("") == ""

    def test_too_long_raises(self):
        from src.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="too long"):
            validate_user_text("x" * 2001)


# ---------------------------------------------------------------------------
# Matching service
# ---------------------------------------------------------------------------

def _fake_ai_service(fake_vlm, fake_embedder):
    from src.services.ai_service import AIService
    return AIService(
        settings=_make_settings(),
        vlm=fake_vlm,
        embedder=fake_embedder,
    )


@pytest.mark.asyncio
async def test_register_lost_creates_record(tmp_path, fake_vlm, fake_embedder, sample_image):
    repo = InMemoryRepository()
    ai = _fake_ai_service(fake_vlm, fake_embedder)
    svc = MatchingService(
        repository=repo,
        ai_service=ai,
        settings=_make_settings(image_storage_dir=str(tmp_path / "store")),
    )

    record = await svc.register_item(
        status=ItemStatus.lost,
        image_path=sample_image,
        user_text="lost my umbrella",
    )
    assert record.status == ItemStatus.lost
    assert record.embedding is not None
    assert record.vlm_description is not None
    assert record.id
    assert len(repo) == 1


@pytest.mark.asyncio
async def test_register_found_creates_record(tmp_path, fake_vlm, fake_embedder, sample_image):
    repo = InMemoryRepository()
    ai = _fake_ai_service(fake_vlm, fake_embedder)
    svc = MatchingService(
        repository=repo,
        ai_service=ai,
        settings=_make_settings(image_storage_dir=str(tmp_path / "store")),
    )
    record = await svc.register_item(
        status=ItemStatus.found,
        image_path=sample_image,
        user_text="found near bus stop",
    )
    assert record.status == ItemStatus.found


@pytest.mark.asyncio
async def test_find_matches_returns_opposite_pool(
    tmp_path, fake_vlm, fake_embedder, sample_image
):
    repo = InMemoryRepository()
    ai = _fake_ai_service(fake_vlm, fake_embedder)
    store_dir = str(tmp_path / "store")
    svc = MatchingService(
        repository=repo, ai_service=ai,
        settings=_make_settings(image_storage_dir=store_dir),
    )

    lost_rec = await svc.register_item(ItemStatus.lost, sample_image, "lost umbrella")

    # Register two found items
    found1 = await svc.register_item(ItemStatus.found, sample_image, "found black umbrella")
    found2 = await svc.register_item(ItemStatus.found, sample_image, "found something")

    response = await svc.find_matches(lost_rec.id, k=3)
    assert response.query_item_id == lost_rec.id
    assert len(response.matches) == 2
    # All matches should be from the found pool
    for m in response.matches:
        assert m.status == ItemStatus.found


@pytest.mark.asyncio
async def test_find_matches_no_opposite_pool(tmp_path, fake_vlm, fake_embedder, sample_image):
    repo = InMemoryRepository()
    ai = _fake_ai_service(fake_vlm, fake_embedder)
    svc = MatchingService(
        repository=repo, ai_service=ai,
        settings=_make_settings(image_storage_dir=str(tmp_path / "store")),
    )
    lost_rec = await svc.register_item(ItemStatus.lost, sample_image, "lost something")
    response = await svc.find_matches(lost_rec.id, k=3)
    assert response.matches == []


@pytest.mark.asyncio
async def test_find_matches_item_not_found(tmp_path, fake_vlm, fake_embedder):
    repo = InMemoryRepository()
    ai = _fake_ai_service(fake_vlm, fake_embedder)
    svc = MatchingService(
        repository=repo, ai_service=ai,
        settings=_make_settings(image_storage_dir=str(tmp_path / "store")),
    )
    with pytest.raises(ItemNotFoundError):
        await svc.find_matches("nonexistent-id", k=3)


@pytest.mark.asyncio
async def test_find_matches_no_embedding(tmp_path):
    repo = InMemoryRepository()
    record = ItemRecord(
        status=ItemStatus.lost,
        user_text="no embedding",
        image_path="/fake/path.png",
        embedding=None,
    )
    await repo.save(record)
    from src.services.ai_service import AIService
    ai = AIService(settings=_make_settings())
    svc = MatchingService(
        repository=repo, ai_service=ai,
        settings=_make_settings(image_storage_dir=str(tmp_path / "store")),
    )
    with pytest.raises(MatchingError):
        await svc.find_matches(record.id)


# ---------------------------------------------------------------------------
# _build_reason
# ---------------------------------------------------------------------------

def _record_with_desc(desc: dict) -> ItemRecord:
    return ItemRecord(
        status=ItemStatus.lost,
        user_text="test",
        image_path="/fake.png",
        vlm_description=desc,
    )


def test_build_reason_same_class():
    q = _record_with_desc({"object_class": "umbrella", "colors": ["black"]})
    c = _record_with_desc({"object_class": "umbrella", "colors": ["black"]})
    reason = _build_reason(q, c, 0.95)
    assert "same object class" in reason
    assert "shared colors" in reason


def test_build_reason_different_class():
    q = _record_with_desc({"object_class": "phone", "colors": []})
    c = _record_with_desc({"object_class": "wallet", "colors": []})
    reason = _build_reason(q, c, 0.4)
    assert "phone" in reason
    assert "wallet" in reason
