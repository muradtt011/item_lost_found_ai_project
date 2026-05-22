
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import Settings
from src.core.matching_service import MatchingService
from src.models import ItemStatus
from src.services.ai_service import AIService
from src.storage.repository import InMemoryRepository


def _make_settings(**kw) -> Settings:
    defaults = dict(
        llm_provider="offline",
        embedding_provider="offline",
        ai_max_retries=0,
        ai_timeout_seconds=5.0,
        ai_concurrency_limit=2,
        image_storage_dir="/tmp/e2e_test_store",
        database_url="sqlite+aiosqlite:///./test.db",
    )
    defaults.update(kw)
    return Settings(**defaults)


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )


@pytest.fixture
def client(tmp_path, fake_vlm, fake_embedder):
    settings = _make_settings(image_storage_dir=str(tmp_path / "store"))
    repo = InMemoryRepository()
    ai_svc = AIService(settings=settings, vlm=fake_vlm, embedder=fake_embedder)
    matching = MatchingService(repository=repo, ai_service=ai_svc, settings=settings)

    with TestClient(app) as c:
        app.state.matching = matching
        app.state.repo = repo
        yield c

    if hasattr(app.state, "matching"):
        del app.state.matching
    if hasattr(app.state, "repo"):
        del app.state.repo


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_lost_item(client, tmp_path):
    img = tmp_path / "umbrella.png"
    img.write_bytes(_png_bytes())
    with open(img, "rb") as f:
        resp = client.post(
            "/items/lost",
            data={"user_text": "lost my umbrella"},
            files={"image": ("umbrella.png", f, "image/png")},
        )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "lost"
    assert body["id"]


def test_register_found_item(client, tmp_path):
    img = tmp_path / "found.png"
    img.write_bytes(_png_bytes())
    with open(img, "rb") as f:
        resp = client.post(
            "/items/found",
            data={"user_text": "found near bus stop"},
            files={"image": ("found.png", f, "image/png")},
        )
    assert resp.status_code == 201
    assert resp.json()["status"] == "found"


def test_register_bad_extension_returns_422(client, tmp_path):
    img = tmp_path / "image.gif"
    img.write_bytes(b"GIF89a" + b"\x00" * 20)
    with open(img, "rb") as f:
        resp = client.post(
            "/items/lost",
            data={"user_text": ""},
            files={"image": ("image.gif", f, "image/gif")},
        )
    assert resp.status_code == 422
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_list_items_empty(client):
    resp = client.get("/items")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_list_items_after_register(client, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(_png_bytes())
    with open(img, "rb") as f:
        client.post("/items/lost", data={"user_text": ""}, files={"image": ("x.png", f, "image/png")})

    resp = client.get("/items")
    assert resp.json()["total"] == 1


def test_list_items_filter_status(client, tmp_path):
    img = tmp_path / "y.png"
    img.write_bytes(_png_bytes())
    with open(img, "rb") as f:
        client.post("/items/lost", data={"user_text": ""}, files={"image": ("y.png", f, "image/png")})

    resp = client.get("/items?status=found")
    assert resp.json()["total"] == 0

    resp = client.get("/items?status=lost")
    assert resp.json()["total"] == 1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_matches_not_found_item(client):
    resp = client.get("/items/nonexistent/matches")
    assert resp.status_code == 404


def test_full_match_flow(client, tmp_path):
    img = tmp_path / "z.png"
    img.write_bytes(_png_bytes())

    with open(img, "rb") as f:
        r1 = client.post("/items/lost", data={"user_text": "lost umbrella"}, files={"image": ("z.png", f, "image/png")})
    lost_id = r1.json()["id"]

    with open(img, "rb") as f:
        client.post("/items/found", data={"user_text": "found umbrella"}, files={"image": ("z.png", f, "image/png")})

    resp = client.get(f"/items/{lost_id}/matches?k=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_item_id"] == lost_id
    assert len(body["matches"]) == 1
    assert body["matches"][0]["score"] >= -1.0
