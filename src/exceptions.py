"""Custom exceptions."""


class OCRError(Exception):
    """Custom exception for OCR-related errors."""


class ImageProcessingError(OCRError):
    """Exception for image processing errors."""


class PDFProcessingError(OCRError):
    """Exception for PDF processing errors."""
