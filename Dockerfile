# ===============================================
# Production Dockerfile for GPU in TRE (Linux x86_64)
# ===============================================
# This image requires NVIDIA GPU with CUDA support
# For local dev/testing on Mac, use: Dockerfile.dev
#
# Build:  docker build -t ner-ocr:latest .
# Run:    docker run --gpus all -p 7860:7860 ner-ocr:latest --mode workbench
#
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# -------------------------------
# 1) System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg2 \
        build-essential gcc \
        poppler-utils \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 2) Install Python 3.13 
# -------------------------------
# Suppress prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.13 /usr/bin/python3

# get pip for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# -------------------------------
# 3) Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 4) Install Python requirements
# -------------------------------
# Use requirements-docker.txt (excludes torch/paddle - installed separately below)
COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# -------------------------------
# 5) Install GPU-enabled PyTorch
# -------------------------------
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip3 install --no-cache-dir accelerate>=1.12.0

# -------------------------------
# 6) Install GPU-enabled PaddlePaddle
# -------------------------------
RUN pip3 install --no-cache-dir paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# -------------------------------
# 7) Copy app code
# -------------------------------
COPY src/ ./src/
COPY scripts/entrypoint.py ./entrypoint.py
COPY data/ ./data/

ENV PYTHONPATH=/app/src

# -------------------------------
# Disable external network calls for TRE
# -------------------------------
# Disable Gradio analytics and telemetry
ENV GRADIO_ANALYTICS_ENABLED=False
# Prevent Gradio from checking for updates
ENV GRADIO_CHECK_UPDATE=False
# Force Hugging Face to use cached models only (no network calls)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# -------------------------------
# 8) Expose ports
# -------------------------------
# Workbench UI port
EXPOSE 7860

# -------------------------------
# 9) Default entrypoint
# -------------------------------
# Usage examples:
#   Pipeline mode (default):
#     docker run ner-ocr --mode ocr -i /data/input -o /data/output
#   
#   Workbench mode:
#     docker run -p 7860:7860 ner-ocr --mode workbench
#
ENTRYPOINT ["python3", "entrypoint.py"]
# Default to showing help if no args provided
CMD ["--help"]
