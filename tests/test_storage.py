"""Unit tests for the storage / repository layer."""

from __future__ import annotations

import pytest

from src.core.exceptions import ItemNotFoundError, StorageError
from src.models import ItemRecord, ItemStatus
from src.storage.repository import InMemoryRepository, JsonRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**kw) -> ItemRecord:
    defaults = dict(status=ItemStatus.lost, user_text="test", image_path="/fake.png")
    defaults.update(kw)
    return ItemRecord(**defaults)


# ===========================================================================
# InMemoryRepository
# ===========================================================================

class TestInMemoryRepository:

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        repo = InMemoryRepository()
        rec = _make_record()
        saved = await repo.save(rec)
        fetched = await repo.get(saved.id)
        assert fetched.id == saved.id
        assert fetched.status == ItemStatus.lost

    @pytest.mark.asyncio
    async def test_duplicate_save_raises(self):
        repo = InMemoryRepository()
        rec = _make_record()
        await repo.save(rec)
        with pytest.raises(StorageError, match="already exists"):
            await repo.save(rec)

    @pytest.mark.asyncio
    async def test_get_missing_raises(self):
        repo = InMemoryRepository()
        with pytest.raises(ItemNotFoundError):
            await repo.get("does-not-exist")

    @pytest.mark.asyncio
    async def test_list_all(self):
        repo = InMemoryRepository()
        await repo.save(_make_record(status=ItemStatus.lost))
        await repo.save(_make_record(status=ItemStatus.found))
        items = await repo.list_items()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_list_filtered_lost(self):
        repo = InMemoryRepository()
        await repo.save(_make_record(status=ItemStatus.lost))
        await repo.save(_make_record(status=ItemStatus.found))
        items = await repo.list_items(status=ItemStatus.lost)
        assert all(i.status == ItemStatus.lost for i in items)
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_list_filtered_found(self):
        repo = InMemoryRepository()
        await repo.save(_make_record(status=ItemStatus.found))
        items = await repo.list_items(status=ItemStatus.found)
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_update(self):
        repo = InMemoryRepository()
        rec = _make_record()
        await repo.save(rec)
        updated = rec.model_copy(update={"user_text": "updated text"})
        result = await repo.update(updated)
        assert result.user_text == "updated text"
        fetched = await repo.get(rec.id)
        assert fetched.user_text == "updated text"

    @pytest.mark.asyncio
    async def test_update_missing_raises(self):
        repo = InMemoryRepository()
        with pytest.raises(ItemNotFoundError):
            await repo.update(_make_record())

    @pytest.mark.asyncio
    async def test_delete(self):
        repo = InMemoryRepository()
        rec = _make_record()
        await repo.save(rec)
        await repo.delete(rec.id)
        with pytest.raises(ItemNotFoundError):
            await repo.get(rec.id)

    @pytest.mark.asyncio
    async def test_delete_missing_raises(self):
        repo = InMemoryRepository()
        with pytest.raises(ItemNotFoundError):
            await repo.delete("does-not-exist")

    @pytest.mark.asyncio
    async def test_len(self):
        repo = InMemoryRepository()
        assert len(repo) == 0
        await repo.save(_make_record())
        assert len(repo) == 1


# ===========================================================================
# JsonRepository
# ===========================================================================

class TestJsonRepository:

    @pytest.mark.asyncio
    async def test_save_and_get(self, tmp_path):
        repo = JsonRepository(json_path=tmp_path / "store.json")
        rec = _make_record()
        await repo.save(rec)
        fetched = await repo.get(rec.id)
        assert fetched.id == rec.id

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "store.json"
        repo1 = JsonRepository(json_path=path)
        rec = _make_record()
        await repo1.save(rec)

        repo2 = JsonRepository(json_path=path)
        fetched = await repo2.get(rec.id)
        assert fetched.id == rec.id

    @pytest.mark.asyncio
    async def test_duplicate_save_raises(self, tmp_path):
        repo = JsonRepository(json_path=tmp_path / "store.json")
        rec = _make_record()
        await repo.save(rec)
        with pytest.raises(StorageError):
            await repo.save(rec)

    @pytest.mark.asyncio
    async def test_list_filtered(self, tmp_path):
        repo = JsonRepository(json_path=tmp_path / "store.json")
        await repo.save(_make_record(status=ItemStatus.lost))
        await repo.save(_make_record(status=ItemStatus.found))
        lost = await repo.list_items(status=ItemStatus.lost)
        assert len(lost) == 1
        assert lost[0].status == ItemStatus.lost

    @pytest.mark.asyncio
    async def test_update(self, tmp_path):
        repo = JsonRepository(json_path=tmp_path / "store.json")
        rec = _make_record()
        await repo.save(rec)
        updated = rec.model_copy(update={"user_text": "new text"})
        await repo.update(updated)
        fetched = await repo.get(rec.id)
        assert fetched.user_text == "new text"

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        repo = JsonRepository(json_path=tmp_path / "store.json")
        rec = _make_record()
        await repo.save(rec)
        await repo.delete(rec.id)
        with pytest.raises(ItemNotFoundError):
            await repo.get(rec.id)

    @pytest.mark.asyncio
    async def test_get_missing_raises(self, tmp_path):
        repo = JsonRepository(json_path=tmp_path / "store.json")
        with pytest.raises(ItemNotFoundError):
            await repo.get("nope")


# ===========================================================================
# store_image helper
# ===========================================================================

@pytest.mark.asyncio
async def test_store_image_copies_file(tmp_path, sample_image):
    repo = InMemoryRepository()
    storage_dir = tmp_path / "images"
    dest = await repo.store_image(
        source_path=__import__("pathlib").Path(sample_image),
        storage_dir=storage_dir,
    )
    assert dest.exists()
    assert dest.parent == storage_dir
