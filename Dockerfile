# Base image with CUDA 11.8 and Python 3.10
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (git, curl, python, ffmpeg for gradio)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir --no-deps git+https://github.com/MIC-DKFZ/HD-BET.git && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy repository scripts and files
COPY . /app/

# Make shell scripts executable
RUN chmod +x run_prediction.sh run_finetuning.sh

# Expose Gradio port
EXPOSE 7860

# Default command launches the Gradio web interface
CMD ["python3", "app.py"]