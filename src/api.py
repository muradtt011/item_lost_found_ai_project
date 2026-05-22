
from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from src.config import configure_logging, get_settings
from src.core.exceptions import (
    AIServiceError,
    ImageValidationError,
    ItemNotFoundError,
    LostFoundError,
    MatchingError,
    ValidationError,
)
from src.core.matching_service import MatchingService
from src.models import (
    ErrorResponse,
    ItemListResponse,
    ItemResponse,
    ItemStatus,
    MatchResponse,
)
from src.services.ai_service import AIService
from src.storage.repository import PostgreSQLRepository, make_repository

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    repo = make_repository(database_url=settings.database_url)
    ai_service = AIService(settings=settings)
    matching = MatchingService(
        repository=repo, ai_service=ai_service, settings=settings
    )

    # Connect PostgreSQL if applicable
    if isinstance(repo, PostgreSQLRepository):
        try:
            await repo.connect()
            logger.info("PostgreSQL connected")
        except Exception as exc:
            logger.warning(
                "PostgreSQL unavailable (%s); falling back to JSON store", exc
            )
            from src.storage.repository import JsonRepository
            repo = JsonRepository()
            matching = MatchingService(
                repository=repo, ai_service=ai_service, settings=settings
            )

    app.state.matching = matching
    app.state.repo = repo

    yield  # ← application runs here

    if isinstance(repo, PostgreSQLRepository):
        await repo.close()
        logger.info("PostgreSQL pool closed")


app = FastAPI(
    title="Smart Lost & Found API",
    version="1.0.0",
    description="Register lost/found items with images and find matches using AI.",
    lifespan=_lifespan,
)


def _get_matching() -> MatchingService:
    """Return the active MatchingService (allows test override via app.state)."""
    return app.state.matching


# ---------------------------------------------------------------------------
# Exception handlers — clean JSON, no stack traces
# ---------------------------------------------------------------------------

@app.exception_handler(ImageValidationError)
async def _image_validation_handler(request, exc: ImageValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="image_validation_error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(ValidationError)
async def _validation_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(ItemNotFoundError)
async def _not_found_handler(request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(error="not_found", detail=str(exc)).model_dump(),
    )


@app.exception_handler(AIServiceError)
async def _ai_error_handler(request, exc: AIServiceError):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(error="ai_service_error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(LostFoundError)
async def _generic_error_handler(request, exc: LostFoundError):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="internal_error", detail=str(exc)).model_dump(),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Item registration endpoints
# ---------------------------------------------------------------------------

async def _register(status: ItemStatus, image: UploadFile, user_text: str) -> ItemResponse:
    """Shared logic for POST /items/lost and POST /items/found."""
    matching = _get_matching()

    # Write the upload to a temp file so validate_image can inspect it
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        record = await matching.register_item(
            status=status,
            image_path=tmp_path,
            user_text=user_text,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return ItemResponse.from_record(record)


@app.post(
    "/items/lost",
    response_model=ItemResponse,
    status_code=201,
    tags=["items"],
    summary="Register a lost item",
)
async def register_lost(
    image: UploadFile = File(..., description="JPEG or PNG image of the lost item"),
    user_text: str = Form(default="", description="Free-text description"),
) -> ItemResponse:
    return await _register(ItemStatus.lost, image, user_text)


@app.post(
    "/items/found",
    response_model=ItemResponse,
    status_code=201,
    tags=["items"],
    summary="Register a found item",
)
async def register_found(
    image: UploadFile = File(..., description="JPEG or PNG image of the found item"),
    user_text: str = Form(default="", description="Free-text description"),
) -> ItemResponse:
    return await _register(ItemStatus.found, image, user_text)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@app.get(
    "/items/{item_id}/matches",
    response_model=MatchResponse,
    tags=["matching"],
    summary="Get top-k matches for an item",
)
async def get_matches(
    item_id: str,
    k: int = Query(default=3, ge=1, le=50, description="Number of matches to return"),
) -> MatchResponse:
    matching = _get_matching()
    return await matching.find_matches(item_id, k=k)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

@app.get(
    "/items",
    response_model=ItemListResponse,
    tags=["items"],
    summary="List all items",
)
async def list_items(
    status: Optional[Literal["lost", "found", "all"]] = Query(
        default="all", description="Filter by status"
    ),
) -> ItemListResponse:
    matching = _get_matching()
    repo = matching._repo  # noqa: SLF001 — intentional internal access

    filter_status: Optional[ItemStatus] = None
    if status in ("lost", "found"):
        filter_status = ItemStatus(status)

    records = await repo.list_items(status=filter_status)
    return ItemListResponse(
        items=[ItemResponse.from_record(r) for r in records],
        total=len(records),
    )
