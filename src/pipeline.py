"""OCR pipeline implementation."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from loguru import logger

from .bounding_box import PaddleOCRWrapper
from .custom_types import PageResult


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
