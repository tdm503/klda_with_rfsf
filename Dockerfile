# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU by default; set TORCH_INDEX_URL to a CUDA wheel index when building for GPU.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/results /cache/huggingface

ENTRYPOINT ["python", "experiments/orf/evaluate.py"]
CMD ["CLINC", "--mapping", "orf", "--D", "256", "--num_ensembles", "1"]
