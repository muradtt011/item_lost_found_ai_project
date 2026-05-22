
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ItemStatus(str, Enum):
    lost = "lost"
    found = "found"


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class ItemRecord(BaseModel):
    
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ItemStatus
    user_text: str
    image_path: str  # path on disk after copy to storage dir
    vlm_description: dict[str, Any] | None = None  # serialised ItemDescription
    embedding: list[float] | None = None            # stored as list for JSON compat
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def description_text(self) -> str:
        if self.vlm_description:
            object_class = self.vlm_description.get("object_class", "")
            colors = ", ".join(self.vlm_description.get("colors", []))
            brand = self.vlm_description.get("brand") or ""
            marks = ", ".join(self.vlm_description.get("distinguishing_marks", []))
            parts = [p for p in [object_class, colors, brand, marks] if p]
            return " | ".join(parts) if parts else self.user_text
        return self.user_text


# ---------------------------------------------------------------------------
# API request / response shapes
# ---------------------------------------------------------------------------

class RegisterItemRequest(BaseModel):

    model_config = ConfigDict(extra="ignore")

    user_text: str = Field(default="", description="Free-text description from the user")


class MatchDetail(BaseModel):

    model_config = ConfigDict(extra="forbid")

    item_id: str
    score: float = Field(ge=-1.0, le=1.0)
    reason: str = ""
    status: ItemStatus
    user_text: str
    vlm_description: dict[str, Any] | None = None
    image_path: str
    created_at: datetime


class MatchResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    query_item_id: str
    matches: list[MatchDetail]


class ItemResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: str
    status: ItemStatus
    user_text: str
    image_path: str
    vlm_description: dict[str, Any] | None = None
    created_at: datetime

    @classmethod
    def from_record(cls, r: ItemRecord) -> "ItemResponse":
        return cls(
            id=r.id,
            status=r.status,
            user_text=r.user_text,
            image_path=r.image_path,
            vlm_description=r.vlm_description,
            created_at=r.created_at,
        )


class ItemListResponse(BaseModel):


    model_config = ConfigDict(extra="forbid")

    items: list[ItemResponse]
    total: int


class ErrorResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str = ""
