"""Entrypoint script for NER-OCR container."""

import argparse
import shutil
import subprocess
from pathlib import Path

from loguru import logger

PADDLE_OCR_DEST = Path("/root/.paddleocr/whl")
PADDLEX_DEST = Path("/root/.paddlex/official_models")
HF_HUB_DEST = Path("/root/.cache/huggingface/hub")


def copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree from src to dst."""
    if not src.exists():
        logger.warning(f"Source path does not exist, skipping: {src}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Copying {src} -> {dst}")
    shutil.copytree(src, dst)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NER-OCR container entrypoint")

    parser.add_argument(
        "--mode",
        choices=["pipeline-ocr", "pipeline-entity-extraction"],
        default="pipeline-ocr",
        help="Which pipeline to run",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory for pipeline",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for pipeline",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file",
    )

    return parser.parse_args()


def main() -> None:
    """Run the main entrypoint function."""
    args = parse_args()

    if args.mode == "pipeline-ocr":
        module = "src.pipeline_ocr"
    elif args.mode == "pipeline-entity-extraction":
        module = "src.pipeline_entity_extraction"
    else:
        error_msg = f"Unknown mode: {args.mode}"
        raise ValueError(error_msg)

    cmd = [
        "python3",
        "-m",
        module,
        "-i",
        args.input,
        "-o",
        args.output,
        "--config",
        args.config,
    ]
    logger.info(f"Running pipeline: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
