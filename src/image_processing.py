"""Image processing utilities."""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .exceptions import ImageProcessingError
from .types.enums import SupportedExtensions


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
