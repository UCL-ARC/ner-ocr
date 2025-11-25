# Match pyproject: requires-python = ">=3.13.5"
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/entrypoint.py ./entrypoint.py
COPY data/ ./data/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "entrypoint.py"]

