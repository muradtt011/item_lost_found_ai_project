

from __future__ import annotations


class LostFoundError(Exception):
    """Base class for all application-level errors."""


class ValidationError(LostFoundError):
    """Input failed validation (bad image format, size, etc.)."""


class ImageValidationError(ValidationError):
    """Specific to image file problems."""


class ItemNotFoundError(LostFoundError):
    """Raised when an item ID does not exist in the repository."""

    def __init__(self, item_id: str) -> None:
        super().__init__(f"Item not found: {item_id!r}")
        self.item_id = item_id


class AIServiceError(LostFoundError):
    """Raised when the AI wrapper cannot complete a call after all retries."""


class StorageError(LostFoundError):
    """Raised when the storage layer encounters an unrecoverable error."""


class MatchingError(LostFoundError):
    """Raised when matching logic encounters an unexpected state."""
