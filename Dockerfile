# Qwen3-TTS validation image
# Base: Ubuntu 24.04 (matches DGX Spark host OS from reference article)
# Ref: https://zenn.dev/karaage0703/articles/97f8a01cbb9c49
# GPU: NVIDIA GB10 (sm_121), CUDA 13.0 driver
# Architecture: ARM64 (aarch64)
# Python: 3.12 (Ubuntu 24.04 default)
FROM ubuntu:24.04

WORKDIR /workspace

# System dependencies
# - ffmpeg: audio conversion for voice cloning sample preparation
# - sox + libsox-dev: required by the sox Python package used in qwen-tts pipeline
# - python3-pip: pip for Ubuntu 24.04's Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /workspace/requirements.txt

# Install torch 2.9.1 + torchaudio from cu130 index (same approach as reference article)
# ARM64 cu130 wheels exist and bundle libcudart.so.13, so no separate CUDA toolkit install needed.
# --break-system-packages is required on Ubuntu 24.04 (PEP 668) in Docker context.
RUN pip install --no-cache-dir --break-system-packages \
    "torch==2.9.1" \
    "torchaudio==2.9.1" \
    --index-url https://download.pytorch.org/whl/cu130

# Install remaining dependencies
# Note: qwen-tts==0.1.1 requires transformers==4.57.3 — this is expected.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy scripts and sample audio into image
# (Also available as bind mounts via docker-compose for iterative development)
COPY scripts/ /workspace/scripts/
COPY sample_audio/ /workspace/sample_audio/

# Create output directory
RUN mkdir -p /workspace/output

# Model cache: mount as named volumes so weights are not re-downloaded each run
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Default: interactive bash
# Run tests with: docker compose run qwen-tts python3 scripts/test_basic_tts.py
CMD ["bash"]
