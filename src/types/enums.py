"""Enums for various supported types."""

from enum import Enum


class SupportedExtensions(Enum):
    """Supported file extensions for OCR processing."""

    PDF = ".pdf"
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"
