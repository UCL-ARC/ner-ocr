"""Enums for various supported types."""

from enum import Enum


# TO DO: maybe just directly move this to the file that needs it?
class SupportedExtensions(Enum):
    """Supported file extensions for OCR processing."""

    PDF = ".pdf"
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"
