"""Entrypoint script for NER-OCR container."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.config import load_config
from src.pipelines import EntityExtractionPipeline, OCRPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NER-OCR container entrypoint")

    parser.add_argument(
        "--mode",
        choices=["ocr", "entity"],
        default="ocr",
        help="Pipeline mode: ocr or entity",
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory/file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")

    return parser.parse_args()


def main() -> int:
    """Run the main entrypoint."""
    args = parse_args()

    # Validate paths
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return 1

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    try:
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    except Exception as e:  # noqa: BLE001
        error_msg = f"Failed to load config: {e}"
        logger.error(error_msg)
        return 1

    # Run pipeline(s)
    pipeline: OCRPipeline | EntityExtractionPipeline

    try:
        if args.mode == "ocr":
            logger.info("Running OCR pipeline")
            pipeline = OCRPipeline(config)
            pipeline.run(input_path, output_path)

        elif args.mode == "entity":
            logger.info("Running entity extraction pipeline")
            pipeline = EntityExtractionPipeline(config)
            pipeline.run(input_path, output_path)

    except Exception as e:  # noqa: BLE001
        error_msg = f"Pipeline failed: {e}"
        logger.error(error_msg)
        return 1
    else:
        logger.info("Pipeline completed successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
