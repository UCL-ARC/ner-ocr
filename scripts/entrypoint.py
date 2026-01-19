"""Entrypoint script for NER-OCR pipelines and workbench UI."""

import argparse
import sys
from pathlib import Path

from loguru import logger


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands for each mode."""
    parser = argparse.ArgumentParser(
        description="NER-OCR - Document OCR and Entity Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        title="modes",
        description="Available modes",
        help="Run 'entrypoint.py <mode> --help' for mode-specific options",
    )

    # OCR subcommand
    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Run OCR pipeline on documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entrypoint.py ocr -i /data/input -o /data/output
  python entrypoint.py ocr -i ./docs -o ./results --config custom.yaml
        """,
    )
    _add_pipeline_args(ocr_parser)

    # Entity extraction subcommand
    entity_parser = subparsers.add_parser(
        "entity",
        help="Run entity extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entrypoint.py entity -i /data/input -o /data/output
  python entrypoint.py entity -i ./docs -o ./results --config custom.yaml
        """,
    )
    _add_pipeline_args(entity_parser)

    # Workbench subcommand
    workbench_parser = subparsers.add_parser(
        "workbench",
        help="Launch interactive web UI (Gradio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entrypoint.py workbench
  python entrypoint.py workbench --port 8080
  python entrypoint.py workbench --auth admin password
        """,
    )
    workbench_parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Host to bind to (default: 0.0.0.0)",
    )
    workbench_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)",
    )
    workbench_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (not recommended for TRE)",
    )

    return parser


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add common pipeline arguments to a subparser."""
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory or file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config YAML path (default: config.yaml)",
    )


def run_workbench(args: argparse.Namespace) -> int:
    """Launch the workbench UI."""
    # Import here to avoid loading heavy dependencies on --help
    from src.ui.app import launch_workbench

    logger.info("=" * 60)
    logger.info("NER-OCR Workbench")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info("=" * 60)

    try:
        launch_workbench(
            host=args.host,
            port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down workbench...")
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Workbench failed: {e}")
        return 1
    else:
        return 0


def run_pipeline(args: argparse.Namespace) -> int:
    """Run OCR or entity extraction pipeline."""
    from src.config import load_config
    from src.pipelines import EntityExtractionPipeline, OCRPipeline

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
        logger.error(f"Failed to load config: {e}")
        return 1

    # Run pipeline
    pipeline: OCRPipeline | EntityExtractionPipeline

    try:
        if args.mode == "ocr":
            logger.info("Running OCR pipeline")
            pipeline = OCRPipeline(config)
        else:
            logger.info("Running entity extraction pipeline")
            pipeline = EntityExtractionPipeline(config)

        pipeline.run(input_path, output_path)

    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed: {e}")
        return 1

    logger.info("Pipeline completed successfully")
    return 0


def main() -> int:
    """Run the main entrypoint."""
    parser = create_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return 0

    if args.mode == "workbench":
        return run_workbench(args)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
