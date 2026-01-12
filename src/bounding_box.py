"""Utilities for bounding box extraction and OCR processing."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR

from .custom_types import BaseOCRProcessor, OCRResult, PageResult, SupportedExtensions
from .exceptions import ImageProcessingError, OCRError
from .image_processing import ImageProcessor
from .pdf_processing import PDFProcessor
from .utils import run_with_timeout


class BoundingBoxExtractor:
    """Handles bounding box extraction from images."""

    @staticmethod
    def extract_bounding_box_image(
        image: np.ndarray, polygon: np.ndarray
    ) -> np.ndarray:
        """
        Extract the image section defined by the polygon bounding box.

        Args:
            image: Source image as numpy array
            polygon: Polygon coordinates defining the bounding box

        Returns:
            Extracted image section with transparent background

        Raises:
            ImageProcessingError: If extraction fails

        """
        logger.debug("Extracting bounding box image")

        # Convert polygon to numpy array
        poly = np.array(polygon, dtype=np.int32)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(poly)

        # Validate bounds
        img_height, img_width = image.shape[:2]
        if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
            logger.warning(
                f"Bounding box extends beyond image bounds: ({x}, {y}, {w}, {h})"
            )
            # Clip to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)

        if w <= 0 or h <= 0:
            error_message = f"Invalid bounding box dimensions: {w}x{h}"
            raise ImageProcessingError(error_message)

        # Extract the rectangular region
        rect_crop = image[y : y + h, x : x + w]

        # Create a mask for the polygon within the bounding rectangle
        mask = np.zeros((h, w), dtype=np.uint8)
        poly_shifted = poly - [x, y]  # Shift polygon to new coordinate system

        # Ensure shifted polygon is within bounds
        poly_shifted = np.clip(poly_shifted, [0, 0], [w - 1, h - 1])

        cv2.fillPoly(mask, [poly_shifted], 255)

        # Apply mask to create transparent background
        rect_crop_rgba = cv2.cvtColor(rect_crop, cv2.COLOR_BGR2RGBA)
        rect_crop_rgba[:, :, 3] = mask  # Set alpha channel

        return rect_crop_rgba


class PaddleOCRWrapper(BaseOCRProcessor):
    """
    Wrapper around PaddleOCR to handle PDFs and extract bounding box images.

    This class provides a unified interface for OCR processing of both PDF and image files,
    with automatic resizing and bounding box extraction capabilities.
    """

    def __init__(
        self, max_side_limit: int = 1500, ocr_timeout: int = 120, **paddle_kwargs
    ) -> None:
        """
        Initialize the PaddleOCR wrapper.

        Args:
            max_side_limit: Maximum allowed size for any side of processed images
            ocr_timeout: Timeout in seconds for OCR operations
            **paddle_kwargs: Additional arguments to pass to PaddleOCR
                Note: 'device' is converted to 'use_gpu' for PaddleOCR compatibility
                      ('gpu' or 'cuda' -> use_gpu=True, 'cpu' -> use_gpu=False)

        """
        # Convert device parameter to use_gpu for PaddleOCR compatibility
        if "device" in paddle_kwargs:
            device = paddle_kwargs.pop("device").lower()
            paddle_kwargs["use_gpu"] = device in ("gpu", "cuda")
            logger.debug(
                f"Converted device='{device}' to use_gpu={paddle_kwargs['use_gpu']}"
            )

        try:
            self.ocr = PaddleOCR(**paddle_kwargs)
        except Exception as e:
            error_message = f"Failed to initialize PaddleOCR: {e}"
            raise OCRError(error_message) from e

        self.max_side_limit = max_side_limit
        self.ocr_timeout = ocr_timeout
        self.image_processor = ImageProcessor()
        self.bbox_extractor = BoundingBoxExtractor()
        self.pdf_processor = PDFProcessor()

        logger.info(
            f"PaddleOCR wrapper initialized with max_side_limit={max_side_limit}"
        )

    def _parse_ocr_result(
        self, result: dict[str, Any], original_image: np.ndarray | None = None
    ) -> list[OCRResult]:
        """
        Parse PaddleOCR result into structured format with bounding box images.

        Args:
            result: Single page result from PaddleOCR
            original_image: Original cv2 image for extracting bounding box images

        Returns:
            List of OCRResult objects

        """
        logger.debug("Parsing OCR result")
        parsed_results = []

        for dt_polys, rec_texts, rec_scores, rec_boxes in zip(
            result["dt_polys"],
            result["rec_texts"],
            result["rec_scores"],
            result["rec_boxes"],
            strict=False,
        ):
            bbox_image = None
            if original_image is not None:
                try:
                    bbox_image = self.bbox_extractor.extract_bounding_box_image(
                        original_image, dt_polys
                    )
                except ImageProcessingError as e:
                    logger.warning(f"Failed to extract bounding box: {e}")

            ocr_result = OCRResult(
                poly=dt_polys.tolist(),
                text=rec_texts,
                score=float(rec_scores),
                box=rec_boxes.tolist(),
                bbox_image=bbox_image,
            )
            parsed_results.append(ocr_result)

        return parsed_results

    def _process_image_file(self, file_path: Path) -> list[PageResult]:
        """
        Process a single image file.

        Args:
            file_path: Path to image file

        Returns:
            List containing single PageResult

        """
        # Resize image if needed
        processed_path, processed_image = self.image_processor.resize_image_to_limit(
            file_path, self.max_side_limit
        )

        logger.info("Running OCR prediction on image...")

        try:
            results = run_with_timeout(
                self.ocr.predict, self.ocr_timeout, str(processed_path)
            )

            logger.info(f"OCR prediction completed, found {len(results)} page(s)")

        except TimeoutError as e:
            error_message = f"OCR prediction timed out after {self.ocr_timeout} seconds"
            raise OCRError(error_message) from e
        except Exception as e:
            error_message = f"OCR prediction failed: {e}"
            raise OCRError(error_message) from e

        # Process results
        page_results = []
        for result in results:
            parsed_result = self._parse_ocr_result(result, processed_image)
            page_result = PageResult(
                page=1,  # Images have a single page
                data=parsed_result,
                original_image=processed_image,
            )
            page_results.append(page_result)

        return page_results

    def _process_pdf_file(self, file_path: Path) -> list[PageResult]:
        """
        Process a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of PageResult objects, one per page

        """
        # Convert PDF to images
        pdf_images = self.pdf_processor.pdf_to_images(file_path)

        logger.info("Running OCR prediction on PDF...")

        try:
            results = run_with_timeout(
                self.ocr.predict, self.ocr_timeout, str(file_path)
            )

            logger.info(f"OCR prediction completed, found {len(results)} page(s)")

        except TimeoutError as e:
            error_message = f"OCR prediction timed out after {self.ocr_timeout} seconds"
            raise OCRError(error_message) from e
        except Exception as e:
            error_message = f"OCR prediction failed: {e}"
            raise OCRError(error_message) from e

        # Process results
        page_results = []
        for i, result in enumerate(results):
            # Get corresponding image
            page_image = pdf_images[i] if i < len(pdf_images) else None
            if page_image is None:
                logger.warning(f"No corresponding image found for page {i}")

            parsed_result = self._parse_ocr_result(result, page_image)
            page_result = PageResult(
                page=result["page_index"] + 1,
                data=parsed_result,
                original_image=page_image,
            )
            page_results.append(page_result)

        return page_results

    def extract(self, file_path: str | Path) -> list[PageResult]:
        """
        Extract OCR results from a file (PDF or image).

        Args:
            file_path: Path to PDF or image file

        Returns:
            List of PageResult objects, one per page

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is not supported
            OCRError: If OCR processing fails

        """
        # Validate input
        validated_path = self.image_processor.validate_image_path(file_path)
        extension = validated_path.suffix.lower()

        logger.info(f"Processing file: {validated_path} (extension: {extension})")

        try:
            if extension == SupportedExtensions.PDF.value:
                return self._process_pdf_file(validated_path)
            return self._process_image_file(validated_path)

        except Exception as e:
            if isinstance(e, (OCRError | FileNotFoundError | ValueError)):
                raise
            error_message = f"Unexpected error processing {validated_path}: {e}"
            raise OCRError(error_message) from e
