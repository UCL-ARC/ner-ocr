# NER OCR

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/macknix/ner-ocr/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/macknix/ner-ocr/actions/workflows/tests.yml
[linting-badge]:            https://github.com/macknix/ner-ocr/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/macknix/ner-ocr/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/macknix/ner-ocr/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/macknix/ner-ocr/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

A pipeline for NER using OCR

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## About

### Project Team

Mack Nixon ([mack.nixon@ucl.ac.uk](mailto:mack.nixon@ucl.ac.uk))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`ner-ocr` requires Python 3.13&ndash;3.11.

### Installation

<!-- How to build or install the application. -->

### Installing `uv`

[uv](https://docs.astral.sh/uv) is used for Python dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
uv sync
```

### Activate your Python environment

```sh
source .venv/bin/activate
```

### Installing pre-commit hooks

Install `pre-commit` locally (in your activated `venv`) to aid code consistency (if you're looking to contribute).

```sh
pre-commit install
```

# Docker Usage

This document explains how to build and run the `ner-ocr` Docker image using **pre‑downloaded models** stored on your local filesystem (or in a TRE).

The image:

- Does **not** bake models in at build time.
- Expects you to **mount** a models directory at runtime and tell it where to find:
  - PaddleOCR models
  - PaddleX models
  - Hugging Face (HF) cache (for TrOCR and Qwen)

---

## 1. Prerequisites

- Docker installed (Docker Desktop on macOS is fine).
- Python (optional, only needed to generate `requirements.txt` and download models the first time).

Project structure (simplified):

```text
ner-ocr/
  src/
  scripts/
    entrypoint.py
  data/
    input/    # your PDFs/images go here
    output/   # pipeline writes results here
  models/
    paddle_models/    # PaddleOCR cache (optional but recommended)
    paddlex_models/   # PaddleX models (optional but recommended)
    hf_cache/         # Hugging Face hub cache (TrOCR + Qwen)
  Dockerfile
  requirements.txt
```

You can choose any host path for `models/`; using `./models` is just a convenient default.

---

## 2. Preparing models (one‑time, on your host)

You need to download all required models once on your host machine, then store them under `models/` so they can be mounted into the container.

### 2.1. Download PaddleOCR / PaddleX models

In your local Python environment (not inside Docker):

```bash
cd /path/to/ner-ocr
python - << 'PYCODE'
from paddleocr import PaddleOCR

# Match your runtime settings (lang, ocr_version, etc.)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    ocr_version="PP-OCRv5",
)
print("PaddleOCR models downloaded.")
PYCODE
```

This will populate:

- `~/.paddleocr/whl`
- `~/.paddlex/official_models`

Copy them into your project `models/` directory:

```bash
mkdir -p models/paddle_models models/paddlex_models

cp -R ~/.paddleocr/whl/. models/paddle_models/
cp -R ~/.paddlex/official_models/. models/paddlex_models/
```

### 2.2. Download TrOCR and Qwen models (Hugging Face)

In the same environment:

```bash
python - << 'PYCODE'
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# TrOCR
trocr_name = "microsoft/trocr-large-handwritten"
TrOCRProcessor.from_pretrained(trocr_name)
VisionEncoderDecoderModel.from_pretrained(trocr_name)

# Qwen models used by entity_extraction
qwen_models = [
    "Qwen/Qwen3-4B-Instruct-2507",
    # add/remove here as needed
]

for name in qwen_models:
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)

print("TrOCR + Qwen models downloaded to HF cache.")
PYCODE
```

This will populate `~/.cache/huggingface/hub`.

Copy the HF cache into `models/hf_cache`:

```bash
mkdir -p models/hf_cache
cp -R ~/.cache/huggingface/hub/. models/hf_cache/
```

Now your `models/` tree should look like:

```text
models/
  paddle_models/
    ... (paddle OCR files) ...
  paddlex_models/
    ... (PaddleX official_models) ...
  hf_cache/
    ... (HF hub repos: trocr, Qwen, etc.) ...
```

---

## 3. Build the Docker image

From the project root:

```bash
cd /path/to/ner-ocr
docker build -t ner-ocr:latest .
```

This uses:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/entrypoint.py ./entrypoint.py
COPY data/ ./data/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "entrypoint.py"]
```

The image now contains only code + dependencies; **no models**.

---

## 4. Running the container

You must:

- Mount your local `models/` directory.
- Mount input/output data directories.
- Pass the model directory paths to `entrypoint.py`.

### 4.1. Local run example (macOS)

Assuming:

- Project root: `/Users/you/Projects/ner-ocr`
- Models under `./models`
- Input PDFs/images under `./data/input`
- Output dir: `./data/output`

Create input/output dirs if needed:

```bash
mkdir -p data/input data/output
# copy some test PDFs/images into data/input
```

Run:

```bash
docker run --rm -it \
  -v "$PWD/models":/models \
  -v "$PWD/data/input":/data/input \
  -v "$PWD/data/output":/data/output \
  ner-ocr:latest \
  --paddle-models-dir /models/paddle_models \
  --paddlex-models-dir /models/paddlex_models \
  --hf-cache-dir /models/hf_cache \
  -i /data/input \
  -o /data/output
```

Explanation:

- `-v "$PWD/models":/models` mounts your host `./models` folder to `/models` in the container.
- `--paddle-models-dir /models/paddle_models` tells the entrypoint where PaddleOCR models are.
- `--paddlex-models-dir /models/paddlex_models` does the same for PaddleX.
- `--hf-cache-dir /models/hf_cache` points to the HF hub cache (TrOCR + Qwen).
- `-i /data/input` and `-o /data/output` are pipeline input/output paths inside the container, mapped to your host `./data/input` and `./data/output`.

The `scripts/entrypoint.py` script will:

1. Optionally copy models into the expected runtime locations (e.g. `/root/.paddleocr/whl`, `/root/.paddlex/official_models`, `/root/.cache/huggingface/hub`), **or** just set env vars if you choose that approach.
2. Set `PADDLEOCR_HOME`, `PADDLEX_HOME`, and `HF_HOME`.
3. Run `python -m src.pipeline` with your `-i` and `-o`.

---

## 5. Running in a TRE

In a TRE you do the same thing conceptually:

1. Ensure your models storage is available inside the container (for example, mounted at `/mnt/models`).
2. Ensure input and output storage are mounted (`/mnt/input`, `/mnt/output`).
3. Configure the container command like:

```bash
python entrypoint.py \
  --paddle-models-dir /mnt/models/paddle_models \
  --paddlex-models-dir /mnt/models/paddlex_models \
  --hf-cache-dir /mnt/models/hf_cache \
  -i /mnt/input \
  -o /mnt/output
```

No rebuild is required when models change; you just update the mounted models directory.

---

## 6. Notes

- If you run out of memory (SIGKILL / exit code 137), reduce model sizes (e.g. use `microsoft/trocr-base-handwritten` instead of the large model) or increase Docker memory (Docker Desktop → Settings → Resources).
- You can also skip copying models at startup and directly mount them into the standard locations:
  - `/root/.paddleocr`
  - `/root/.paddlex`
  - `/root/.cache/huggingface`
  if you prefer, as long as the directory structures match what the libraries expect.