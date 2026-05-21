"""Image and request validation.

All validation failures raise subclasses of LostFoundError so the API
layer can map them to clean HTTP 4xx responses without leaking stack traces.
"""

from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path

from src.config import Settings, get_settings
from src.core.exceptions import ImageValidationError, ValidationError

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Minimal magic-byte signatures
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_JPEG_SIG = b"\xff\xd8\xff"


def validate_image(
    image_path: str | Path,
    settings: Settings | None = None,
) -> Path:
    """Validate an image file and return its resolved Path.

    Checks:
    - File exists
    - Extension is .jpg / .jpeg / .png
    - File size <= max_image_size_bytes
    - Magic bytes match the declared format
    - For PNG: IDAT chunk present (not truncated)

    Parameters
    ----------
    image_path:
        Path to the image on disk.
    settings:
        Optional settings override (uses global singleton if omitted).

    Returns
    -------
    Path
        The resolved path, confirmed valid.

    Raises
    ------
    ImageValidationError
        On any validation failure.
    """
    cfg = settings or get_settings()
    path = Path(image_path).resolve()

    logger.debug("Validating image: %s", path)

    if not path.exists():
        raise ImageValidationError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ImageValidationError(f"Path is not a regular file: {path}")

    ext = path.suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image format {ext!r}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
        )

    size = path.stat().st_size
    if size == 0:
        raise ImageValidationError("Image file is empty (0 bytes)")

    max_bytes = cfg.max_image_size_bytes
    if size > max_bytes:
        raise ImageValidationError(
            f"Image size {size / 1_048_576:.2f} MB exceeds limit "
            f"{cfg.max_image_size_mb} MB"
        )

    # Read enough bytes to check magic + basic structure
    header = path.read_bytes()[:8]

    if ext == ".png":
        _validate_png(path, header)
    elif ext in (".jpg", ".jpeg"):
        _validate_jpeg(header)

    logger.debug("Image validated OK: %s (%.1f KB)", path.name, size / 1024)
    return path


def _validate_png(path: Path, header: bytes) -> None:
    if not header.startswith(_PNG_SIG):
        raise ImageValidationError(
            f"File {path.name!r} has .png extension but invalid PNG magic bytes"
        )
    # Try to confirm the file is not truncated by checking it contains IDAT
    data = path.read_bytes()
    if b"IDAT" not in data:
        raise ImageValidationError(
            f"PNG file {path.name!r} appears corrupted (no IDAT chunk)"
        )


def _validate_jpeg(header: bytes) -> None:
    if not header.startswith(_JPEG_SIG):
        raise ImageValidationError(
            "File has JPEG extension but invalid JPEG magic bytes (not FF D8 FF)"
        )


def validate_user_text(text: str) -> str:
    """Normalise and validate user-supplied free text.

    Returns the stripped string (which may be empty — that is allowed).
    Raises ValidationError if the text is unreasonably long.
    """
    text = text.strip()
    if len(text) > 2000:
        raise ValidationError(
            f"user_text is too long ({len(text)} chars; max 2000)"
        )
    return text
