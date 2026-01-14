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

A modular pipeline for extracting named entities from scanned documents using OCR and LLMs.

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## About

### Project Team

Mack Nixon ([mack.nixon@ucl.ac.uk](mailto:mack.nixon@ucl.ac.uk))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

---

## Pipeline Overview

The NER-OCR pipeline processes documents through four stages:

```                   
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌───────────────────┐
│   1. OCR    │───▶│  2. Search  │───▶│  3. Enhancement │───▶│ 4. Entity Extract │
│ (PaddleOCR) │    │ (RPA/Query) │    │    (TrOCR)      │    │     (Qwen LLM)    │
└─────────────┘    └─────────────┘    └─────────────────┘    └───────────────────┘
```

| Stage | Purpose | Model |
|-------|---------|-------|
| **1. OCR** | Extract text and bounding boxes from document images | PaddleOCR v5 |
| **2. Search** | Filter regions of interest using semantic or positional queries | Fuzzy matching / coordinates |
| **3. Enhancement** | Improve OCR text quality using transformer models | Microsoft TrOCR |
| **4. Entity Extraction** | Extract structured entities from text using LLM | Qwen 3 |

---

## Configuration

The pipeline is configured via two YAML files:

| File | Purpose |
|------|---------|
| `config.yaml` | Pipeline settings (models, devices, queries) |
| `entities.yaml` | Custom entity definitions |

### config.yaml

```yaml
# OCR Configuration (PaddleOCR)
ocr:
  max_side_limit: 1500        # Max image dimension (pixels)
  ocr_timeout: 400            # Timeout in seconds
  use_doc_orientation_classify: false  # Auto-rotate documents
  use_doc_unwarping: false    # Dewarp curved documents
  use_textline_orientation: false      # Detect text line angles
  return_word_box: true       # Return word-level boxes (vs line-level)
  device: cpu                 # 'cpu' or 'gpu'

# Transformer OCR Configuration (TrOCR)
transformer_ocr:
  model: "LARGE_HANDWRITTEN"  # TrOCR model variant
  device: cpu                 # 'cpu', 'cuda', or 'mps'
  max_new_tokens: 128         # Max tokens to generate (increase for longer text)

# Search Queries
queries:
  - task: "Extract address"
    query_type: "semantic"    # 'semantic' or 'positional'
    query_kwargs:
      text: "address"         # Search term
      threshold: 0.9          # Match confidence (0-1)
      search_type: "fuzzy"    # 'fuzzy' or 'exact'
      search_padding: 50.0    # Expand search region (pixels)

# Entity Extraction Configuration (Qwen)
entity_extraction:
  model: "QWEN3_1_7B"         # Qwen model variant
  device: "cpu"               # 'cpu', 'cuda', or 'mps'
  entities:                   # Entities to extract
    - AddressEntityList
  line_threshold: 10          # Y-distance for same-line grouping
  gap_threshold: 40           # Y-distance for paragraph breaks
```

---

## Pipeline Stages

### Stage 1: OCR (PaddleOCR)

Extracts text and bounding boxes from PDF pages or images using PaddleOCR v5.

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_side_limit` | int | 1500 | Maximum image dimension. Larger values = better accuracy, more memory |
| `ocr_timeout` | int | 400 | Timeout in seconds per page |
| `use_doc_orientation_classify` | bool | false | Auto-detect and correct document rotation |
| `use_doc_unwarping` | bool | false | Correct warped/curved documents (e.g., book spines) |
| `use_textline_orientation` | bool | false | Detect text line angles for skewed text |
| `return_word_box` | bool | true | Return word-level boxes. Set `false` for line-level |
| `device` | str | "cpu" | `cpu` or `gpu` (PaddleOCR uses 'gpu', not 'cuda') |

#### Output

Each detected text region includes:
- Bounding box coordinates `[x_min, y_min, x_max, y_max]`
- OCR text
- Confidence score

---

### Stage 2: Search (RPA/Query)

Filters OCR results to regions of interest using semantic or positional queries.

#### Query Types

**Semantic Query** - Find text matching a search term:
```yaml
queries:
  - task: "Find addresses"
    query_type: "semantic"
    query_kwargs:
      text: "address"         # Search term
      threshold: 0.9          # Minimum match score (0-1)
      search_type: "fuzzy"    # 'fuzzy' or 'exact'
      search_padding: 50.0    # Expand region around match (pixels)
```

**Positional Query** - Find text at specific coordinates:
```yaml
queries:
  - task: "Top-left region"
    query_type: "positional"
    query_kwargs:
      x: 100                  # X coordinate
      y: 200                  # Y coordinate
      search_radius: 50       # Search radius (pixels)
```

---

### Stage 3: Enhancement (TrOCR)

Improves OCR text quality by re-processing cropped text regions through Microsoft's TrOCR transformer model. Particularly effective for handwritten text.

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | str | "LARGE_HANDWRITTEN" | TrOCR model variant (see below) |
| `device` | str | "cpu" | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `max_new_tokens` | int | 128 | Maximum tokens to generate (increase for longer text) |

#### Available Models

| Model | Use Case | Size |
|-------|----------|------|
| `BASE_HANDWRITTEN` | Handwritten text (faster) | ~330MB |
| `BASE_PRINTED` | Printed text (faster) | ~330MB |
| `LARGE_HANDWRITTEN` | Handwritten text (better accuracy) | ~560MB |
| `LARGE_PRINTED` | Printed text (better accuracy) | ~560MB |
| `BASE_STR` | Scene text (signs, labels) | ~330MB |
| `LARGE_STR` | Scene text (better accuracy) | ~560MB |

---

### Stage 4: Entity Extraction (Qwen LLM)

Extracts structured entities from text using Qwen large language models. The LLM receives the OCR text and entity schema, returning structured JSON.

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | str | "QWEN3_1_7B" | Qwen model variant (see below) |
| `device` | str | "cpu" | `cpu`, `cuda`, or `mps` |
| `entities` | list | ["AddressEntityList"] | Entity types to extract |
| `line_threshold` | int | 10 | Y-distance for grouping text on same line |
| `gap_threshold` | int | 40 | Y-distance for inserting paragraph breaks |

#### Available Models

| Model | Parameters | Memory | Speed |
|-------|------------|--------|-------|
| `QWEN3_1_7B` | 1.7B | ~4GB | Fast |
| `QWEN3_4B_INSTRUCT_2507` | 4B | ~8GB | Medium |
| `QWEN3_8B` | 8B | ~16GB | Slow |

#### Text Formatting Options

The `line_threshold` and `gap_threshold` control how OCR results are formatted into text for the LLM:

- **`line_threshold`**: Items within this Y-distance are joined on the same line
- **`gap_threshold`**: Gaps larger than this insert paragraph breaks

Lower values = more line breaks. Higher values = denser text blocks.

---

## Custom Entities

Define custom entities in `entities.yaml`. These are converted to Pydantic models at runtime.

### entities.yaml Format

```yaml
entities:
  PersonEntity:
    description: "Data model for a person entity"
    create_list: true    # Also creates PersonEntityList
    fields:
      first_name:
        type: "str | None"
        description: "Person's first name"
      last_name:
        type: "str | None"
        description: "Person's last name"
      date_of_birth:
        type: "str | None"
        description: "Date of birth in any format"
      raw_text:
        type: "str"
        description: "Raw text containing person information"
        required: true
```

### Field Types

| Type | Description |
|------|-------------|
| `str` | Required string |
| `str \| None` | Optional string |
| `int` | Required integer |
| `int \| None` | Optional integer |
| `float` | Required float |
| `float \| None` | Optional float |
| `bool` | Required boolean |
| `bool \| None` | Optional boolean |

### Field Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | str | "str \| None" | Field type |
| `description` | str | "" | Description (shown to LLM - be specific!) |
| `required` | bool | false | If true, field cannot be None |

### Using Custom Entities

After defining entities in `entities.yaml`, reference them in `config.yaml`:

```yaml
entity_extraction:
  entities:
    - PersonEntityList      # Your custom entity
    - AddressEntityList     # Built-in entity
```

### Docker Usage

Mount your custom entities file:

```bash
docker run -p 7860:7860 \
  -v ./entities.yaml:/app/entities.yaml \
  -v ./config.yaml:/app/config.yaml \
  ner-ocr:latest --mode workbench
```

### Built-in Entities

The following entities are always available:

- **`AddressEntity`**: Street, city, state, postal code, country, address type
- **`AddressEntityList`**: List of AddressEntity

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv) for dependency management
- Docker (for containerized deployment)

### Installing uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Running Environments

This project supports three deployment modes. Choose based on your use case:

| Environment | Use Case | Hardware | Performance |
|-------------|----------|----------|-------------|
| **Local (Mac)** | Development & testing | Apple Silicon with MPS | Fast (GPU accelerated) |
| **Docker Dev** | Local containerized testing | CPU only | Slower |
| **Docker TRE** | Production deployment | NVIDIA GPU with CUDA | Fast (GPU accelerated) |

---

## 1. Local Development (Mac with MPS)

**Best for:** Day-to-day development with fast iteration. Uses Apple's Metal Performance Shaders (MPS) for GPU acceleration.

### Setup

```bash
# Install all dependencies including PyTorch with MPS support
uv sync --group base --group dev

# Activate the virtual environment
source .venv/bin/activate

# (Optional) Install pre-commit hooks for contributors
pre-commit install
```

### Running the Workbench UI

```bash
uv run python -m scripts.run_ui
```

Then open http://localhost:7860 in your browser.

### Running the Pipeline CLI

```bash
uv run python scripts/entrypoint.py --mode ocr -i data/input -o data/output
```

### Verify MPS is Working

```bash
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## 2. Docker Dev (CPU-only, Cross-platform)

**Best for:** Testing the containerized application locally before TRE deployment. Works on Mac, Windows, and Linux.

### Build

```bash
docker build -f Dockerfile.dev -t ner-ocr:dev .
```

### Run Workbench UI

```bash
docker run -p 7860:7860 \
  -v "$PWD/data":/app/data \
  -v "$PWD/models":/app/models \
  -v "$PWD/config.yaml":/app/config.yaml \
  -v "$PWD/entities.yaml":/app/entities.yaml \
  ner-ocr:dev --mode workbench
```

Then open http://localhost:7860 in your browser.

### Run Pipeline

```bash
docker run --rm \
  -v "$PWD/data":/app/data \
  -v "$PWD/models":/app/models \
  -v "$PWD/config.yaml":/app/config.yaml \
  -v "$PWD/entities.yaml":/app/entities.yaml \
  ner-ocr:dev --mode ocr -i /app/data/input -o /app/data/output
```

> ⚠️ **Note:** CPU inference is slow, especially for entity extraction (~5-10 min per document). For faster development, use local Mac environment with MPS.

---

## 3. Docker TRE (GPU, Production)

**Best for:** Production deployment in a Trusted Research Environment with NVIDIA GPUs.

### Build

```bash
docker build -t ner-ocr:latest .
```

### Build for TRE Export (x86_64)

If building on Apple Silicon for deployment to x86_64 TRE:

```bash
docker build --platform linux/amd64 -t ner-ocr:amd64 .
docker save ner-ocr:amd64 | gzip > ner-ocr-amd64.tar.gz
```

Load in TRE:

```bash
gzip -dc ner-ocr-amd64.tar.gz | docker load
```

### Run with GPU

```bash
docker run --gpus all -p 7860:7860 \
  -v /mnt/data:/app/data \
  -v /mnt/models:/app/models \
  ner-ocr:latest --mode workbench
```

### Network-Isolated TRE Deployment

The Docker image is configured for network-isolated environments:

- **Gradio:** Analytics and update checks are disabled via `GRADIO_ANALYTICS_ENABLED=False` and `GRADIO_CHECK_UPDATE=False`
- **Hugging Face:** Offline mode is enabled via `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`

These environment variables are set in the Dockerfile. Models must be pre-cached in the image or mounted from the host.

To run with explicit offline settings (if not baked into image):

```bash
docker run --gpus all -p 7860:7860 \
  -e GRADIO_ANALYTICS_ENABLED=False \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/models:/app/models \
  ner-ocr:latest --mode workbench
```

### Accessing the UI in TRE

The UI binds to `0.0.0.0:7860` by default. To access it:

1. Find the host IP: `hostname -I` or `ip addr show`
2. Open in browser: `http://<host-ip>:7860`

> ⚠️ **Note:** `localhost` may not work in some TRE setups. Use the actual IP address.

---

## Dependency Management

This project uses `uv` with dependency groups to handle different environments cleanly.

### File Structure

| File | Purpose |
|------|---------|
| `pyproject.toml` | Single source of truth for all dependencies |
| `requirements-docker.txt` | Docker deps (excludes torch/paddle) |
| `requirements.txt` | Full deps (reference only) |

### Dependency Groups

```toml
[dependency-groups]
base = ["torch", "paddlepaddle", "accelerate", "torchvision"]  # ML frameworks
dev = ["pytest", "ruff", "mypy", "pre-commit"]                  # Development tools
```

### Regenerating Requirements Files

After modifying `pyproject.toml`:

```bash
# For Docker (excludes torch/paddle - installed separately in Dockerfile)
uv export --no-group base --no-group dev -o requirements-docker.txt --no-hashes

# Full export (for reference)
uv export -o requirements.txt --no-hashes
```

### Why Separate Files?

Docker images need specific versions of PyTorch and PaddlePaddle:
- **TRE Docker:** CUDA-enabled versions (`torch==2.6.0+cu126`, `paddlepaddle-gpu==3.2.0`)
- **Dev Docker:** CPU-only versions
- **Local Mac:** Standard PyPI versions with MPS support

By excluding the `base` group from Docker requirements, we can install the correct platform-specific versions in each Dockerfile.

---

## Model Setup

The pipeline requires pre-downloaded models. Models are **not** baked into Docker images.

### Download Models (One-time)

```bash
# Activate your local environment first
source .venv/bin/activate

# Download PaddleOCR models
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv5')"

# Download TrOCR models
python -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
"

# Download Qwen models (for entity extraction)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')
"
```

### Copy to Project Models Directory

```bash
mkdir -p models/paddle_models models/paddlex_models models/hf_cache

cp -R ~/.paddleocr/whl/. models/paddle_models/
cp -R ~/.paddlex/official_models/. models/paddlex_models/
cp -R ~/.cache/huggingface/hub/. models/hf_cache/
```

### Model Directory Structure

```text
models/
  paddle_models/     # PaddleOCR detection/recognition
  paddlex_models/    # PaddleX official models
  hf_cache/          # Hugging Face cache (TrOCR, Qwen)
```

---

## Quick Reference

### Local Mac Development

```bash
uv sync --group base --group dev
uv run python -m scripts.run_ui
```

### Docker Dev (Testing)

```bash
docker build -f Dockerfile.dev -t ner-ocr:dev .
docker run -p 7860:7860 -v "$PWD/data":/app/data ner-ocr:dev --mode workbench
```

### Docker TRE (Production)

```bash
docker build -t ner-ocr:latest .
docker run --gpus all -p 7860:7860 ner-ocr:latest --mode workbench
```

---

## Troubleshooting

### TRE: Connection Refused / Cannot Access UI

- Ensure port mapping is set: `docker run -p 7860:7860 ...`
- Use host IP instead of localhost: `http://<host-ip>:7860`
- Check container is running: `docker ps`
- Check logs for errors: `docker logs <container_id>`

### TRE: Browser Hangs on "Connecting to..." External URLs

If the browser shows it's trying to connect to external URLs (e.g., `cdnjs.cloudflare.com`), ensure offline mode is enabled:

```bash
docker run --gpus all -p 7860:7860 \
  -e GRADIO_ANALYTICS_ENABLED=False \
  -e HF_HUB_OFFLINE=1 \
  ner-ocr:latest --mode workbench
```

### TRE: Slow Model Loading

If models take a long time to load in a network-isolated TRE, Hugging Face may be timing out on network requests. Ensure `HF_HUB_OFFLINE=1` is set to skip network checks.

### Out of Memory (Exit Code 137)

- Increase Docker memory (Docker Desktop → Settings → Resources)
- Use smaller models (e.g., `trocr-base-handwritten` instead of `large`)
- Use `Qwen3-1.7B` instead of `Qwen3-8B`

### Slow Entity Extraction

- Entity extraction with Qwen models is CPU-intensive
- **On Mac:** Run locally (not in Docker) to use MPS acceleration
- **In TRE:** Ensure GPU is available and CUDA is working

### Model Not Found Errors

- Ensure models are downloaded and mounted correctly
- Check `config.yaml` paths match your model locations
- Set `local_files_only: false` in config if models need downloading

---

## Project Structure

```text
ner-ocr/
  src/               # Source code
  scripts/           # CLI entrypoints
  data/
    input/           # Input PDFs/images
    output/          # Pipeline output
  models/            # Pre-downloaded models (not in git)
  Dockerfile         # Production (GPU/TRE)
  Dockerfile.dev     # Development (CPU)
  pyproject.toml     # Dependencies
  config.yaml        # Runtime configuration
```

---

## About

### Project Team

Mack Nixon ([mack.nixon@ucl.ac.uk](mailto:mack.nixon@ucl.ac.uk))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))