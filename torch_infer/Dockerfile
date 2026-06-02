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

# Optional: uncomment to enable Flash Attention 2 (faster, needs nvcc at build time)
# RUN pip install flash-attn --no-build-isolation --no-cache-dir

COPY server.py .

CMD ["python", "server.py"]
