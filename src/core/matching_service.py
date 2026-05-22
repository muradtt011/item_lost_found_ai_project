

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Settings, get_settings
from src.core.exceptions import ItemNotFoundError, MatchingError
from src.core.validation import validate_image, validate_user_text
from src.models import ItemRecord, ItemStatus, MatchDetail, MatchResponse
from src.services.ai_service import AIService
from src.storage.repository import ItemRepository

logger = logging.getLogger(__name__)


class MatchingService:

    def __init__(
        self,
        repository: ItemRepository,
        ai_service: AIService,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repository
        self._ai = ai_service
        self._cfg = settings or get_settings()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_item(
        self,
        status: ItemStatus,
        image_path: str | Path,
        user_text: str = "",
    ) -> ItemRecord:
        
        user_text = validate_user_text(user_text)
        validated_path = validate_image(image_path, settings=self._cfg)

        # Copy image into permanent storage dir
        storage_dir = Path(self._cfg.image_storage_dir)
        dest_path = await self._repo.store_image(validated_path, storage_dir)

        logger.info(
            "Registering %s item | image=%s | text=%r",
            status.value, dest_path.name, user_text[:60],
        )

        # AI pipeline
        description = await self._ai.describe_item(str(dest_path), user_text)
        search_text = description.to_search_text()
        if user_text:
            search_text = f"{search_text} | {user_text}"

        embedding = await self._ai.embed(search_text)

        record = ItemRecord(
            status=status,
            user_text=user_text,
            image_path=str(dest_path),
            vlm_description=description.model_dump(),
            embedding=embedding.tolist(),
        )

        saved = await self._repo.save(record)
        logger.info("Registered item %s (status=%s)", saved.id, status.value)
        return saved

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    async def find_matches(
        self,
        item_id: str,
        k: int = 3,
    ) -> MatchResponse:
        
        if k <= 0:
            raise ValueError("k must be a positive integer")

        query_item = await self._repo.get(item_id)

        if query_item.embedding is None:
            raise MatchingError(
                f"Item {item_id!r} has no embedding; was it fully processed?"
            )

        query_vec = np.array(query_item.embedding, dtype=np.float32)

        # Search in the opposite pool
        opposite_status = (
            ItemStatus.found if query_item.status == ItemStatus.lost else ItemStatus.lost
        )
        candidates = await self._repo.list_items(status=opposite_status)

        if not candidates:
            logger.info("No %s items to match against", opposite_status.value)
            return MatchResponse(query_item_id=item_id, matches=[])

        # Filter to items that have embeddings
        valid = [c for c in candidates if c.embedding is not None]
        if not valid:
            logger.info("No %s items with embeddings to match", opposite_status.value)
            return MatchResponse(query_item_id=item_id, matches=[])

        cand_vecs = [np.array(c.embedding, dtype=np.float32) for c in valid]

        raw_matches = self._ai.top_k(query_vec, cand_vecs, k=min(k, len(valid)))

        details: list[MatchDetail] = []
        for m in raw_matches:
            candidate = valid[m.candidate_id]
            reason = _build_reason(query_item, candidate, m.score)
            details.append(
                MatchDetail(
                    item_id=candidate.id,
                    score=float(m.score),
                    reason=reason,
                    status=candidate.status,
                    user_text=candidate.user_text,
                    vlm_description=candidate.vlm_description,
                    image_path=candidate.image_path,
                    created_at=candidate.created_at,
                )
            )

        logger.info(
            "find_matches: query=%s, pool=%s, k=%d -> %d results",
            item_id, opposite_status.value, k, len(details),
        )
        return MatchResponse(query_item_id=item_id, matches=details)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_reason(
    query: ItemRecord,
    candidate: ItemRecord,
    score: float,
) -> str:
    """Produce a human-readable reason string from two VLM descriptions."""
    parts: list[str] = [f"similarity={score:.3f}"]

    q_desc = query.vlm_description or {}
    c_desc = candidate.vlm_description or {}

    q_class = q_desc.get("object_class", "")
    c_class = c_desc.get("object_class", "")
    if q_class and c_class:
        if q_class.lower() == c_class.lower():
            parts.append(f"same object class ({q_class})")
        else:
            parts.append(f"object: {q_class} vs {c_class}")

    q_colors = set(q_desc.get("colors", []))
    c_colors = set(c_desc.get("colors", []))
    common_colors = q_colors & c_colors
    if common_colors:
        parts.append(f"shared colors: {', '.join(sorted(common_colors))}")

    q_brand = q_desc.get("brand")
    c_brand = c_desc.get("brand")
    if q_brand and c_brand and q_brand.lower() == c_brand.lower():
        parts.append(f"brand match: {q_brand}")

    return "; ".join(parts)
