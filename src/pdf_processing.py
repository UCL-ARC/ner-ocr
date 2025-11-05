"""PDF processing utilities."""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pypdfium2 as pdfium
from loguru import logger

from .exceptions import PDFProcessingError


class PDFProcessor:
    """Handles PDF processing operations."""

    def __init__(self) -> None:
        """Initialise PDF processor with thread lock."""
        self._lock = Lock()

    def pdf_to_images(
        self, pdf_path: Path, max_num_imgs: int | None = None, zoom: float = 2.0
    ) -> list[np.ndarray]:
        """
        Convert PDF pages to images using pdfium (same as PaddleOCR).

        Args:
            pdf_path: Path to PDF file
            max_num_imgs: Maximum number of images to extract
            zoom: Zoom factor for rendering (higher = better quality)

        Returns:
            List of numpy arrays (cv2 images), one per page

        Raises:
            PDFProcessingError: If PDF processing fails

        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        images: list[np.ndarray] = []

        try:
            # Read PDF bytes
            with Path.open(pdf_path, "rb") as f:
                bytes_ = f.read()

            with self._lock:
                doc = pdfium.PdfDocument(bytes_)
                try:
                    for page_idx, page in enumerate(doc):
                        if max_num_imgs is not None and len(images) >= max_num_imgs:
                            break

                        logger.debug(f"Processing PDF page {page_idx}")

                        # Use same settings as PaddleOCR
                        deg = 0
                        image = page.render(scale=zoom, rotation=deg).to_pil()
                        image = image.convert("RGB")
                        image = np.array(image)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        images.append(image)

                finally:
                    doc.close()

        except Exception as e:
            error_message = f"Error processing PDF {pdf_path}: {e}"
            raise PDFProcessingError(error_message) from e

        else:
            logger.info(f"Converted {len(images)} pages to images from PDF")
            return images
