"""Pipeline module for data processing."""

import argparse
import json
from pathlib import Path

import yaml
from loguru import logger

from .entities import ENTITY_REGISTRY
from .entity_extraction import QwenEntityExtractor, QwenModels


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
        default="data/intermediate",
        help="Input directory or file path containing documents to process",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output",
        help="Output directory for saving results",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    return parser.parse_args()


def to_markdown(
    page_entry: dict,
    line_threshold: int = 10,
    gap_threshold: int = 40,  # if two lines are farther apart than this, add an extra blank line
) -> str:
    """
    Convert OCR JSON with bounding boxes into Markdown.

    Uses only the 'box' field:
        box = [x_min, y_min, x_max, y_max]

    Args:
        page_entry: single element from the parsed JSON (with "page_result")
        line_threshold: max difference in y_min to consider items on same line
        gap_threshold: vertical gap between lines that triggers an extra blank line

    """
    page_result = page_entry["page_result"]
    items = page_result.get("data", [])

    processed: list[dict] = []
    for item in items:
        box = item.get("box")
        if not box or len(box) != 4:  # noqa: PLR2004
            continue  # skip if no usable box

        x_min, y_min, _, _ = box
        text = item.get("transformer_text") or item.get("text") or ""
        text = text.strip()
        if not text:
            continue

        processed.append({"x": x_min, "y": y_min, "text": text})

    if not processed:
        return ""

    # Sort roughly top-to-bottom, left-to-right
    processed.sort(key=lambda it: (it["y"], it["x"]))

    # Build lines with their average y
    lines: list[tuple[float, list[dict]]] = []
    current_line: list[dict] = []
    last_y: int | None = None

    for it in processed:
        y = it["y"]
        if last_y is None or abs(y - last_y) <= line_threshold:
            current_line.append(it)
        else:
            # finish previous line
            current_line.sort(key=lambda x: x["x"])
            avg_y = sum(tok["y"] for tok in current_line) / len(current_line)
            lines.append((avg_y, current_line))
            current_line = [it]

        last_y = y

    # add last line
    if current_line:
        current_line.sort(key=lambda x: x["x"])
        avg_y = sum(tok["y"] for tok in current_line) / len(current_line)
        lines.append((avg_y, current_line))

    # Now build markdown, inserting extra blank lines when gaps are large
    markdown_lines: list[str] = []
    last_line_y: float | None = None

    for line_y, line_tokens in lines:
        if last_line_y is not None:
            gap = line_y - last_line_y
            if gap > gap_threshold:
                # add an extra blank line for big gaps
                # TO DO: this should be proportional to the gap size
                markdown_lines.append("")
                markdown_lines.append("")
                markdown_lines.append("")

        markdown_lines.append("   ".join(tok["text"] for tok in line_tokens))
        last_line_y = line_y

    return "\n".join(markdown_lines)


def pipeline(
    file_path: Path,
    extractor: QwenEntityExtractor,
    entities: list[str],
) -> list[dict]:
    """Run main pipeline function for extracting single entities from a file."""
    # read in json file path
    with Path.open(file_path, "r") as f:
        parsed_result = json.load(f)

    page_results = []
    # TO DO HERE THIS NEEDS TO BE FIXED!!!!!! COMPLETELY BROKEN NOW
    for page in parsed_result:
        markdown_page = to_markdown(page)

        extracted_entities = {}
        for entity in entities:
            result = extractor.extract_entities(
                markdown_page, entity_model=ENTITY_REGISTRY[entity]
            )
            if result["content"]:
                extracted_entities[entity] = result[
                    "content"
                ].model_dump()  # note this requires it to be pydantic object
            else:
                extracted_entities[entity] = {}

        logger.info(f"\n\n### Page {page['page_result']['page']}\n\n{markdown_page}")
        logger.info(f"Extracted Entities: {extracted_entities}")
        page_result = {
            "page": page["page_result"]["page"],
            "page_text": markdown_page,
            "entities": extracted_entities,
        }
        page_results.append(page_result)

    return page_results


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

    extractor = QwenEntityExtractor(
        model=QwenModels[config["entity_extraction"]["model"]],
        device=config["entity_extraction"].get("device", None),
        local=True,
    )

    for file_path in all_files:
        try:
            results = pipeline(
                file_path, extractor, entities=config["entity_extraction"]["entities"]
            )

            json_output_path = output_dir / f"{file_path.stem}_entity_results.json"
            with Path.open(json_output_path, "w") as f:
                json.dump(results, f, indent=4)
        except Exception as e:  # noqa: BLE001
            error_msg = f"Failed to process {file_path}: {e}"
            logger.error(error_msg)
