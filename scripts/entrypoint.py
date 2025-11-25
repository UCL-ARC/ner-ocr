"""Entrypoint script for NER-OCR container."""

import argparse
import os
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
        "--paddle-models-dir",
        default=os.environ.get("PADDLE_MODELS_DIR", ""),
        help="Dir with PaddleOCR models (copied to /root/.paddleocr/whl)",
    )
    parser.add_argument(
        "--paddlex-models-dir",
        default=os.environ.get("PADDLEX_MODELS_DIR", ""),
        help="Dir with PaddleX models (copied to /root/.paddlex/official_models)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=os.environ.get("HF_CACHE_DIR", ""),
        help="Dir with HF hub cache (copied to /root/.cache/huggingface/hub)",
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

    return parser.parse_args()


def main() -> None:
    """Run the main entrypoint function."""
    args = parse_args()

    if args.paddle_models_dir:
        copy_tree(Path(args.paddle_models_dir), PADDLE_OCR_DEST)

    if args.paddlex_models_dir:
        copy_tree(Path(args.paddlex_models_dir), PADDLEX_DEST)

    if args.hf_cache_dir:
        copy_tree(Path(args.hf_cache_dir), HF_HUB_DEST)

    os.environ["PADDLEOCR_HOME"] = str(PADDLE_OCR_DEST.parent)
    os.environ["PADDLEX_HOME"] = str(PADDLEX_DEST.parent)
    os.environ["HF_HOME"] = str(HF_HUB_DEST.parent)

    cmd = [
        "python",
        "-m",
        "src.pipeline",
        "-i",
        args.input,
        "-o",
        args.output,
    ]
    logger.info(f"Running pipeline: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
