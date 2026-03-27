ARG TRITON_IMAGE_TAG
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

# ── Chỉ cần pymupdf cho Triton container (vLLM đã chạy riêng) ────────────────
RUN pip install pymupdf --no-cache-dir

# ── Update Triton vLLM backend model.py lên main branch ──────────────────────
# r26.02 model.py viết cho vLLM 0.15.x. Dù Triton không chạy vLLM trực tiếp
# nữa, model.py vẫn cần mới để engine startup check không bị lỗi API mismatch.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone --depth 1 \
         https://github.com/triton-inference-server/vllm_backend.git \
         /tmp/vllm_backend \
    && cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm/ \
    && rm -rf /tmp/vllm_backend \
    && apt-get purge -y git && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── Patch CUDA_VISIBLE_DEVICES ────────────────────────────────────────────────
# Fix: https://github.com/triton-inference-server/server/issues/6855
RUN python3 - <<'PYEOF'
import sys

path = "/opt/tritonserver/backends/vllm/model.py"
try:
    with open(path) as f:
        src = f.read()
except FileNotFoundError:
    print(f"[patch] {path} not found — skipping", flush=True)
    sys.exit(0)

patch = (
    '    import os\n'
    '    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.get("model_instance_device_id", "0"))\n'
)
marker = "async def initialize(self, args):"
if patch.strip() in src:
    print("[patch] already applied — skipping", flush=True)
    sys.exit(0)
if marker not in src:
    print(f"[patch] marker '{marker}' not found — skipping", flush=True)
    sys.exit(0)

patched = src.replace(marker, marker + "\n" + patch, 1)
with open(path, "w") as f:
    f.write(patched)
print("[patch] CUDA_VISIBLE_DEVICES patch applied to vLLM backend", flush=True)
PYEOF
