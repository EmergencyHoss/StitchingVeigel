# Use the official PyTorch image with CUDA 12.1 (works for RTX 50-series GPUs)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system tools and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a workspace
WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip

# Default command
CMD ["/bin/bash"]
