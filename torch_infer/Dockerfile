FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System libs needed by Pillow and OpenCV (via transformers vision deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch is already in the base image; --no-deps avoids re-downloading it
RUN pip install --no-cache-dir -r requirements.txt

# Flash Attention 2 — pre-built wheel (cu12 + torch2.5 + py3.11, no nvcc needed)
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

COPY server.py .

CMD ["python", "server.py"]
