# GTX 980 Ti (Maxwell Architecture) Optimized Dockerfile
# Uses CUDA 11.2.2, cuDNN 8.2.1 for Maxwell compatibility
# Reference: GTX 980 Ti supports compute capability 5.2

# Stage 1: Base image with CUDA 11.2.2 and cuDNN 8.2.1
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables for Maxwell architecture
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set compute capability for GTX 980 Ti (Maxwell SM 5.2)
ENV TORCH_CUDA_ARCH_LIST="5.2"
ENV CUDA_ARCH_FLAGS="-gencode=arch=compute_52,code=sm_52"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Upgrade pip and install base dependencies
RUN pip install --upgrade pip setuptools wheel

# Install OpenCV with CUDA support for GPU-accelerated preprocessing
# We build opencv-contrib-python to get CUDA modules
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    opencv-python-headless==4.7.0.72 \
    opencv-contrib-python-headless==4.7.0.72

# Install PaddlePaddle GPU version compatible with CUDA 11.2
# Using paddlepaddle-gpu 2.6.1 which supports CUDA 11.2 and has better PaddleOCR compatibility
# NOTE: PaddlePaddle 3.x GPU versions require CUDA 11.7+ or 12.x
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==2.6.1.post112 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Copy requirements and install remaining dependencies
COPY ocr-system/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/watch_folder /app/models

# Copy application code
COPY ocr-system/app.py .
COPY ocr-system/gpu_preprocessing.py .
COPY ocr-system/hybrid_ocr.py .
COPY ocr-system/ocr_system.py .
COPY ocr-system/api_server.py .
COPY docker/start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Expose ports
# 8501 - Streamlit Dashboard
# 8000 - REST API for Android clients
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application (both Streamlit and FastAPI)
CMD ["./start.sh"]
