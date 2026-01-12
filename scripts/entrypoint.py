"""Entrypoint script for NER-OCR container."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.config import load_config
from src.pipelines import EntityExtractionPipeline, OCRPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NER-OCR container entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  ocr        - Run OCR pipeline on documents
  entity     - Run entity extraction pipeline
  workbench  - Launch interactive web UI (Gradio)

Examples:
  # Run OCR pipeline
  python entrypoint.py --mode ocr -i /data/input -o /data/output

  # Run entity extraction
  python entrypoint.py --mode entity -i /data/input -o /data/output

  # Launch workbench UI
  python entrypoint.py --mode workbench --port 7860
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["ocr", "entity", "workbench"],
        default="ocr",
        help="Pipeline mode: ocr, entity, or workbench (interactive UI)",
    )

    # Pipeline mode arguments
    parser.add_argument(
        "-i", "--input", help="Input directory/file (required for ocr/entity modes)"
    )
    parser.add_argument(
        "-o", "--output", help="Output directory (required for ocr/entity modes)"
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")

    # Workbench mode arguments
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Host for workbench mode (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for workbench mode (default: 7860)",
    )
    parser.add_argument(
        "--auth-user",
        help="Username for workbench authentication",
    )
    parser.add_argument(
        "--auth-pass",
        help="Password for workbench authentication",
    )

    return parser.parse_args()


def run_workbench(args: argparse.Namespace) -> int:
    """Launch the workbench UI."""
    from src.ui.app import launch_workbench

    logger.info("=" * 60)
    logger.info("NER-OCR Workbench")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Auth: {'enabled' if args.auth_user else 'disabled'}")
    logger.info("=" * 60)

    auth = None
    if args.auth_user and args.auth_pass:
        auth = (args.auth_user, args.auth_pass)

    try:
        launch_workbench(
            host=args.host,
            port=args.port,
            share=False,  # Never share in TRE
            auth=auth,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down workbench...")
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Workbench failed: {e}")
        return 1
    else:
        return 0


# TO DO: resolve PLR0911
def run_pipeline(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Run OCR or entity extraction pipeline."""
    # Validate required arguments for pipeline modes
    if not args.input:
        logger.error("--input/-i is required for ocr/entity modes")
        return 1
    if not args.output:
        logger.error("--output/-o is required for ocr/entity modes")
        return 1

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


def main() -> int:
    """Run the main entrypoint."""
    args = parse_args()

    if args.mode == "workbench":
        return run_workbench(args)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
