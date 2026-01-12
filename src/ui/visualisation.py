"""Visualisation utilities for the UI."""

import cv2
import numpy as np
from PIL import Image

from src.custom_types import OCRResult

# Image channel constants (standard values for image processing)
GRAYSCALE_DIMS = 2  # Grayscale images have 2 dimensions
RGB_CHANNELS = 3  # RGB images have 3 channels
RGBA_CHANNELS = 4  # RGBA images have 4 channels

# Display constants
TEXT_TRUNCATE_LENGTH = 30


def draw_ocr_boxes(
    image: np.ndarray,
    ocr_results: list[OCRResult],
    highlight_indices: list[int] | None = None,
    *,
    show_text: bool = False,
) -> np.ndarray:
    """
    Draw OCR bounding boxes on an image.

    Args:
        image: Original image as numpy array
        ocr_results: List of OCR results with polygon coordinates
        highlight_indices: Indices of results to highlight (e.g., search matches)
        show_text: Whether to draw text labels on boxes

    Returns:
        Image with bounding boxes drawn

    """
    # Make a copy to avoid modifying original
    img = image.copy()

    # Convert to BGR if needed for OpenCV drawing
    if len(img.shape) == GRAYSCALE_DIMS:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == RGBA_CHANNELS:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.shape[2] == RGB_CHANNELS:
        # Assume RGB, convert to BGR for cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    highlight_indices = highlight_indices or []

    for idx, result in enumerate(ocr_results):
        poly = np.array(result.poly, dtype=np.int32)

        # Choose color based on whether this is highlighted
        if idx in highlight_indices:
            color = (0, 255, 0)  # Green for highlighted (BGR)
            thickness = 3
        else:
            color = (255, 0, 0)  # Blue for normal (BGR)
            thickness = 1

        # Draw polygon
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness)

        # Draw text label if requested
        if show_text and result.text:
            x, y = int(poly[0][0]), int(poly[0][1]) - 5
            # Truncate long text
            display_text = (
                result.text[:TEXT_TRUNCATE_LENGTH] + "..."
                if len(result.text) > TEXT_TRUNCATE_LENGTH
                else result.text
            )
            cv2.putText(
                img,
                display_text,
                (x, max(y, 10)),  # Ensure y is not negative
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

    # Convert back to RGB for display in Gradio
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_comparison_view(
    original_text: str,
    enhanced_text: str,
    bbox_image: np.ndarray | None = None,
) -> dict:
    """Create a comparison view between original and enhanced OCR."""
    result = {
        "original": original_text,
        "enhanced": enhanced_text,
        "match": original_text.lower().strip() == enhanced_text.lower().strip(),
    }

    if bbox_image is not None:
        # Convert to PIL for display
        if (
            len(bbox_image.shape) == RGB_CHANNELS
            and bbox_image.shape[2] == RGBA_CHANNELS
        ):
            result["image"] = Image.fromarray(bbox_image, mode="RGBA")
        elif len(bbox_image.shape) == RGB_CHANNELS:
            result["image"] = Image.fromarray(bbox_image)
        else:
            result["image"] = Image.fromarray(bbox_image)

    return result


def render_results_table(ocr_results: list[OCRResult]) -> list[list[str]]:
    """
    Convert OCR results to a table format for display.

    Returns:
        List of rows: [index, text, enhanced_text, score, enhanced_score]

    """
    rows = []
    for idx, result in enumerate(ocr_results):
        rows.append(
            [
                str(idx),
                result.text,
                result.transformer_text or "-",
                f"{result.score:.3f}",
                f"{result.transformer_score:.3f}" if result.transformer_score else "-",
            ]
        )
    return rows


def render_entity_results(entity_results: list[dict]) -> str:
    """Render entity extraction results as formatted markdown."""
    if not entity_results:
        return "No entities extracted yet."

    md_parts = []
    for page_result in entity_results:
        page_num = page_result.get("page", "?")
        md_parts.append(f"## Page {page_num}\n")

        entities = page_result.get("entities", {})
        if not entities:
            md_parts.append("*No entities found on this page.*\n")
            continue

        for entity_type, entity_data in entities.items():
            md_parts.append(f"### {entity_type}\n")

            if isinstance(entity_data, dict):
                if "addresses" in entity_data:
                    for addr in entity_data["addresses"]:
                        md_parts.append(f"- **Raw**: {addr.get('raw_text', 'N/A')}")
                        if addr.get("street"):
                            md_parts.append(f"  - Street: {addr.get('street')}")
                        if addr.get("city"):
                            md_parts.append(f"  - City: {addr.get('city')}")
                        if addr.get("state"):
                            md_parts.append(f"  - State: {addr.get('state')}")
                        if addr.get("postal_code"):
                            md_parts.append(
                                f"  - Postal Code: {addr.get('postal_code')}"
                            )
                        if addr.get("country"):
                            md_parts.append(f"  - Country: {addr.get('country')}")
                        if addr.get("address_type"):
                            md_parts.append(f"  - Type: {addr.get('address_type')}")
                        md_parts.append("")
                else:
                    for key, value in entity_data.items():
                        if value:
                            md_parts.append(f"- **{key}**: {value}")
            md_parts.append("")

    return "\n".join(md_parts)


def get_highlighted_indices(
    ocr_data: list[OCRResult],
    search_data: list[OCRResult],
) -> list[int]:
    """
    Find indices in ocr_data that match items in search_data.

    Matches by comparing polygon coordinates.
    """
    highlighted = []
    for idx, ocr_item in enumerate(ocr_data):
        for search_item in search_data:
            # Compare by polygon (first point should be sufficient)
            if ocr_item.poly == search_item.poly:
                highlighted.append(idx)
                break
    return highlighted
