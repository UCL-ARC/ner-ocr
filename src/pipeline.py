"""Pipeline module for data processing."""

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import time

import numpy as np
from loguru import logger

from .bounding_box import PaddleOCRWrapper
from .custom_types import PositionalQuery, SearchResult, SemanticQuery
from .rpa import RPAProcessor
from .transformer_ocr import TrOCRModels, TrOCRWrapper


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    returns: argparse.Namespace with input and output paths.
    """
    parser = argparse.ArgumentParser(
        description="OCR Pipeline for document processing with semantic/positional search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/input",
        help="Input directory or file path containing documents to process",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Output directory for saving results",
    )

    return parser.parse_args()


def parse_result(result: SearchResult) -> dict:
    """
    Convert a SearchResult (or nested OCR dataclasses) into a JSON-safe dict,
    dropping heavy image data and converting NumPy types to native Python.

    Args:
        result: SearchResult or nested OCR dataclass
    returns: cleaned dict suitable for JSON serialization

    """

    def _clean(obj):  # noqa: ANN202, ANN001
        """
        Recursive helper to go down dictionary tree and remove the elements we don't want.
        and convert numpy arrays and numpy scalars to native Python types.
        """
        # Handle dataclasses
        if is_dataclass(obj):
            obj = asdict(obj)
        # Handle dicts
        if isinstance(obj, dict):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if k not in {"original_image", "bbox_image"}
            }
        # Handle lists and tuples
        if isinstance(obj, list | tuple):
            return [_clean(v) for v in obj]
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numpy scalars
        if isinstance(obj, np.float32 | np.float64 | np.int32 | np.int64):
            return obj.item()
        # Everything else — leave as is
        return obj

    # for now we just do this later more may be added hence the unnecessary func in func
    return _clean(result)


def to_markdown(data: list[dict], line_threshold: int = 10) -> None:
    """
    Convert OCR JSON with bounding boxes into Markdown.

    Args:
        data: list of page results with bounding boxes and text
        line_threshold: max difference in y_min to consider same line

    """
    markdown_pages = []

    for page_entry in data:
        page_result = page_entry["page_result"]
        page_number = page_result["page"]
        items = page_result["data"]

        # Sort items top-to-bottom
        items_sorted = sorted(items, key=lambda x: x["box"][1])  # sort by y_min

        lines = []
        current_line = []
        last_y = None

        for item in items_sorted:
            y_min = item["box"][1]
            x_min = item["box"][0]
            text = item.get("transformer_text", item["text"])

            if last_y is None or abs(y_min - last_y) <= line_threshold:
                current_line.append((x_min, text))
            else:
                # finish current line
                current_line.sort(key=lambda x: x[0])
                lines.append("   ".join([t[1] for t in current_line]))
                current_line = [(x_min, text)]

            last_y = y_min

        # Add last line
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append("   ".join([t[1] for t in current_line]))

        markdown_page = "\n".join(lines)
        markdown_pages.append(f"### Page {page_number}\n\n{markdown_page}")

    logger.info("\n\n".join(markdown_pages))


def pipeline(
    file_path: str | Path,
    ocr: PaddleOCRWrapper,
    model: TrOCRWrapper,
    rpa: RPAProcessor,
    query: SemanticQuery | PositionalQuery,
) -> list[SearchResult]:
    """Pipeline to process a document and extract information based on a query."""
    start_time = time()
    logger.info(f"Starting pipeline for file: {file_path}")

    # Step 1: Run OCR to get initial text detections
    ocr_results = ocr.extract(file_path)
    logger.info(
        f"OCR completed. Detected {sum(len(page.data) for page in ocr_results)} text regions."
    )

    # Step 2: Search for relevant information
    searched_results = []
    for page in ocr_results:
        searched_page = rpa.search(page, query)
        searched_results.append(searched_page)

    # Step 3: Enhance OCR results using Transformer OCR
    for searched_page in searched_results:
        for item in searched_page.page_result.data:
            transformer_output = model.predict(item.bbox_image)
            logger.info(
                f"Original Text: '{item.text}' | Transformer Text: '{transformer_output.transformer_text}'"
            )
            item.transformer_text = transformer_output.transformer_text
            item.transformer_score = transformer_output.score

    end_time = time()
    elapsed_time = end_time - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds.")
    return searched_results


if __name__ == "__main__":
    # Init pipeline components
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    ocr = PaddleOCRWrapper(
        max_side_limit=1500,
        ocr_timeout=400,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        device="cpu",  # Use CPU for Mac compatibility
    )

    model = TrOCRWrapper(
        model=TrOCRModels.LARGE_HANDWRITTEN,
        device="mps",  # Use mps for faster inference
        cache_dir="../models",
    )

    # rpa = RPAProcessor(search_type="semantic", search_kwargs={}, verbose=True)
    # query = SemanticQuery(text="NCDSID", threshold=0.9, search_type="fuzzy", search_padding=50.0)
    rpa = RPAProcessor(search_type="positional", search_kwargs={}, verbose=True)
    query = PositionalQuery(x=300, y=300, search_radius=150)

    # Get input files
    input_path = Path(args.input)

    if input_path.is_file():
        all_files = [input_path]
        logger.info(f"Processing single file: {input_path}")
    elif input_path.is_dir():
        all_files = list(input_path.iterdir())
        all_files = [f for f in all_files if f.is_file()]
        logger.info(f"Processing {len(all_files)} files from directory: {input_path}")
    else:
        logger.error(f"Input path does not exist: {input_path}")
        error_msg = f"Input path does not exist: {input_path}"
        raise FileNotFoundError(error_msg)

    for file_path in all_files:
        try:
            results = pipeline(file_path, ocr, model, rpa, query)
            if results:
                logger.success(f"Successfully processed {file_path}")
                # save results as json
                parsed_results = [parse_result(res) for res in results]
                to_markdown(parsed_results, line_threshold=10)

                json_output_path = output_dir / f"{file_path.stem}_results.json"
                with Path.open(json_output_path, "w") as f:
                    json.dump(parsed_results, f, indent=4)

            else:
                logger.warning(f"No results found for {file_path}")
        except Exception as e:  # noqa: BLE001
            error_msg = f"Failed to process {file_path}: {e}"
            logger.error(error_msg)
