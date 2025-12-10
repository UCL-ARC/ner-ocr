"""Pipeline module for data processing."""

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import time

import numpy as np
import yaml
from loguru import logger

from .bounding_box import PaddleOCRWrapper
from .rpa import RPAProcessor
from .transformer_ocr import TrOCRModels, TrOCRWrapper
from .types import PositionalQuery, SearchResult, SemanticQuery


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
        default="data/intermediate",
        help="Output directory for saving OCR results",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
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


def pipeline(
    file_path: str | Path,
    ocr: PaddleOCRWrapper,
    model: TrOCRWrapper,
    queries: list[dict],
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
    for query in queries:
        query_obj: SemanticQuery | PositionalQuery
        logger.info(f"Processing query: {query['task']}")
        if query["query_type"] == "semantic":
            rpa = RPAProcessor(
                search_type="semantic",
                search_kwargs=query.get("search_kwargs", {}),
                verbose=True,
            )
            query_obj = SemanticQuery(
                text=query["query_kwargs"]["text"],
                threshold=query["query_kwargs"].get("threshold", 0.8),
                search_type=query["query_kwargs"].get("search_type", "fuzzy"),
                search_padding=query["query_kwargs"].get("search_padding", 50.0),
            )
        elif query["query_type"] == "positional":
            rpa = RPAProcessor(
                search_type="positional",
                search_kwargs=query.get("search_kwargs", {}),
                verbose=True,
            )
            query_obj = PositionalQuery(
                x=query["query_kwargs"]["x"],
                y=query["query_kwargs"]["y"],
                search_radius=query["query_kwargs"].get("search_radius", 100),
            )
        else:
            logger.error(f"Unknown query type: {query['query_type']}")
            continue

        for page in ocr_results:
            searched_page = rpa.search(page, query_obj, task=query.get("task"))
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

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    logger.info(f"Loaded config: {config}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    ocr = PaddleOCRWrapper(**config["ocr"])

    model = TrOCRWrapper(
        model=TrOCRModels[config["transformer_ocr"]["model"]],
        device=config["transformer_ocr"].get("device", None),
    )

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
            results = pipeline(file_path, ocr, model, config["queries"])
            if results:
                logger.success(f"Successfully processed {file_path}")
                # save results as json
                parsed_results = [parse_result(res) for res in results]

                json_output_path = output_dir / f"{file_path.stem}_ocr_results.json"
                with Path.open(json_output_path, "w") as f:
                    json.dump(parsed_results, f, indent=4)

            else:
                logger.warning(f"No results found for {file_path}")
        except Exception as e:  # noqa: BLE001
            error_msg = f"Failed to process {file_path}: {e}"
            logger.error(error_msg)
