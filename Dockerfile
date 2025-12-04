# ===============================================
# Optimized Dockerfile for GPU Python in TRE
# ===============================================
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
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# -------------------------------
# 5) Install GPU-enabled PyTorch
# -------------------------------
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

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
# 8) Entrypoint
# -------------------------------
ENTRYPOINT ["python3", "entrypoint.py"]
