"""Run the NER-OCR Workbench web UI."""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NER-OCR Workbench - Interactive Pipeline UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ui.py                    # Run on localhost:7860
  python scripts/run_ui.py --port 8080        # Run on custom port
  python scripts/run_ui.py --auth admin pass  # Enable basic auth
        """,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (not recommended for TRE)",
    )
    parser.add_argument(
        "--auth",
        nargs=2,
        metavar=("USERNAME", "PASSWORD"),
        help="Enable basic authentication with username and password",
    )

    return parser.parse_args()


def main() -> int:
    """Launch the Gradio UI."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NER-OCR Workbench")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Auth: {'enabled' if args.auth else 'disabled'}")
    logger.info("=" * 60)

    # Import here to avoid loading heavy dependencies on --help
    from src.ui.app import launch_workbench

    auth = tuple(args.auth) if args.auth else None

    try:
        launch_workbench(
            host=args.host,
            port=args.port,
            share=args.share,
            auth=auth,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down workbench...")
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Failed to start workbench: {e}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
