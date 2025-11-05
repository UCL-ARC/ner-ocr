"""Utilities for bounding box extraction and OCR processing."""

import signal
import types
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pypdfium2 as pdfium
from loguru import logger
from paddleocr import PaddleOCR


class SupportedExtensions(Enum):
    # TO DO: check what this does
    """Supported file extensions for OCR processing."""

    PDF = ".pdf"
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"


@dataclass
class OCRResult:
    """Structured OCR result for a single text detection."""

    poly: list[list[float]]
    text: str
    score: float
    box: list[float]
    bbox_image: np.ndarray | None = None


@dataclass
class PageResult:
    """Results for a single page."""

    page: int
    data: list[OCRResult]
    original_image: np.ndarray | None = None


class OCRError(Exception):
    """Custom exception for OCR-related errors."""


class ImageProcessingError(OCRError):
    """Exception for image processing errors."""


class PDFProcessingError(OCRError):
    """Exception for PDF processing errors."""


# TO do check how ABC works and what the point of this is
class BaseOCRProcessor(ABC):
    """Abstract base class for OCR processors."""

    @abstractmethod
    def extract(self, file_path: str | Path) -> list[PageResult]:
        """Extract OCR results from a file."""


class ImageProcessor:
    """Handles image processing operations."""

    # TO DO: check how staticmethod works
    @staticmethod
    def validate_image_path(image_path: str | Path) -> Path:
        """
        Validate that the image path exists and has a supported extension.

        Args:
            image_path: Path to the image file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is not supported

        """
        file_path = Path(image_path)

        if not file_path.exists():
            error_message = f"File not found: {image_path}"
            raise FileNotFoundError(error_message)

        extension = file_path.suffix.lower()
        supported_extensions = {ext.value for ext in SupportedExtensions}

        if extension not in supported_extensions:
            error_message = (
                f"Unsupported file extension: {extension}. "
                f"Supported extensions: {supported_extensions}"
            )
            raise ValueError(error_message)

        return file_path

    @staticmethod
    def calculate_resize_dimensions(
        original_width: int, original_height: int, max_side_limit: int
    ) -> tuple[int, int, float]:
        """
        Calculate new dimensions for resizing while maintaining aspect ratio.

        Args:
            original_width: Original image width
            original_height: Original image height
            max_side_limit: Maximum allowed size for any side

        Returns:
            Tuple of (new_width, new_height, scale_factor)

        """
        if max(original_height, original_width) <= max_side_limit:
            return original_width, original_height, 1.0

        scale_factor = max_side_limit / max(original_height, original_width)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        return new_width, new_height, scale_factor

    @staticmethod
    def resize_image_to_limit(
        image_path: Path, max_side_limit: int = 1500
    ) -> tuple[Path, np.ndarray]:
        """
        Resize image if any side exceeds the max_side_limit while maintaining aspect ratio.

        Args:
            image_path: Path to input image
            max_side_limit: Maximum allowed size for any side

        Returns:
            Tuple of (path_to_processed_image, processed_image_array)

        Raises:
            ImageProcessingError: If image cannot be loaded or processed

        """
        try:
            image = cv2.imread(str(image_path))
        except Exception as e:
            msg = f"Could not load image: {image_path}, error: {e}"
            raise ImageProcessingError(msg) from e

        if image is None:
            msg = f"Could not load image: {image_path}"
            raise ImageProcessingError(msg)

        height, width = image.shape[:2]
        new_width, new_height, scale_factor = (
            ImageProcessor.calculate_resize_dimensions(width, height, max_side_limit)
        )

        if scale_factor == 1.0:
            logger.info(
                f"Image size ({width}x{height}) is within limit ({max_side_limit})"
            )
            return image_path, image

        logger.info(
            f"Image size ({width}x{height}) exceeds max_side_limit of {max_side_limit}. "
            f"Resizing to ({new_width}x{new_height})"
        )

        # Resize with high quality interpolation
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Save resized image to a temporary path
        resized_image_path = image_path.parent / f"resized_{image_path.name}"
        success = cv2.imwrite(str(resized_image_path), resized_image)

        if not success:
            msg = f"Failed to save resized image to: {resized_image_path}"
            raise ImageProcessingError(msg)

        logger.info(f"Resized image saved to: {resized_image_path}")
        return resized_image_path, resized_image


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


@contextmanager
def timeout_context(duration: int) -> Generator[None, None, None]:
    """Context manager for adding timeout to operations."""

    def timeout_handler(_signum: int, _frame: types.FrameType | None) -> None:
        error_message = f"Operation timed out after {duration} seconds"
        raise TimeoutError(error_message)

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


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

        """
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
            with timeout_context(self.ocr_timeout):
                results = self.ocr.predict(str(processed_path))

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
                page=result["page_index"],
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
            with timeout_context(self.ocr_timeout):
                results = self.ocr.predict(str(file_path))

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
                page=result["page_index"], data=parsed_result, original_image=page_image
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


def display_results(results: list[PageResult], max_samples_per_page: int = 5) -> None:
    """
    Display OCR results with matplotlib.

    Args:
        results: List of PageResult objects
        max_samples_per_page: Maximum number of samples to show per page

    """
    for page_result in results:
        logger.info(
            f"Page {page_result.page}: {len(page_result.data)} text regions found"
        )

        samples_shown = 0
        for i, item in enumerate(page_result.data):
            if samples_shown >= max_samples_per_page:
                break

            logger.info(f"Text {i+1}: '{item.text}' (confidence: {item.score:.3f})")

            if item.bbox_image is not None:
                try:
                    # Convert BGRA to RGB for display
                    bbox_image = cv2.cvtColor(item.bbox_image, cv2.COLOR_BGRA2RGB)

                    # Show the image
                    plt.figure(figsize=(8, 3))
                    plt.imshow(bbox_image)
                    plt.title(f"Page {page_result.page}: '{item.text}'")
                    plt.axis("off")
                    plt.show()

                    samples_shown += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to display bounding box image: {e}")
            else:
                logger.info("  No bounding box image available")


if __name__ == "__main__":
    # Initialize the wrapper with robust configuration
    try:
        ocr = PaddleOCRWrapper(
            max_side_limit=1500,
            ocr_timeout=400,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            return_word_box=True,
            device="cpu",  # Use CPU for Mac compatibility
        )

        def test_file(file_path: str | Path) -> None:
            """Test OCR extraction on a single file."""
            try:
                logger.info(f"Testing OCR on: {file_path}")
                results = ocr.extract(file_path)

                if results:
                    display_results(results, max_samples_per_page=3)
                    logger.success(f"Successfully processed {file_path}")
                else:
                    logger.warning(f"No results found for {file_path}")

            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to process {file_path}: {e}")

        # Test files
        test_files = [
            "data/input/pms_annotated-1.pdf",
            "data/input/pms_annotated-1.png",
            "data/input/pms_annotated-1.jpg",
        ]

        for test_file_path in test_files:
            test_file(test_file_path)

        logger.success("All tests completed!")

    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to initialize OCR wrapper: {e}")
