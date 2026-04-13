# Qwen3-TTS validation image
# Base: NVIDIA PyTorch NGC 25.03 (PyTorch 2.7.0, CUDA 12.8 toolkit, Python 3.12)
# GPU: NVIDIA GB10 (sm_121), CUDA 13.0 driver via forward-compatibility
# Architecture: ARM64 (aarch64)
FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /workspace

# System dependencies
# - ffmpeg: audio conversion for voice cloning sample preparation
# - sox + libsox-dev: required by the sox Python package used in qwen-tts pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /workspace/requirements.txt

# Install torchaudio separately with --no-deps
# Reason: NGC base image ships torch==2.7.0a0+7c8ec84dab.nv25.3 (custom build string).
#         torchaudio 2.7.0 requires torch==2.7.0 (exact match), causing a conflict.
#         --no-deps skips the version check; the binaries are functionally compatible.
RUN pip install --no-cache-dir torchaudio==2.7.0 --no-deps

# Install remaining dependencies
# Note: qwen-tts==0.1.1 requires transformers==4.57.3, which downgrades
#       the base image's transformers 5.3.0. This is expected for this image.
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and sample audio into image
# (Also available as bind mounts via docker-compose for iterative development)
COPY scripts/ /workspace/scripts/
COPY sample_audio/ /workspace/sample_audio/

# Create output directory
RUN mkdir -p /workspace/output

# Model cache: mount as named volumes so weights are not re-downloaded each run
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Default: interactive bash
# Run tests with: docker compose run qwen-tts python3 scripts/test_basic_tts.py
CMD ["bash"]
