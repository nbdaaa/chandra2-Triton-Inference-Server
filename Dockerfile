ARG TRITON_IMAGE_TAG=25.11-vllm-python-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_IMAGE_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Ensure each vLLM Triton instance sees only its assigned GPU.
# Based on the same patching pattern used by the reference repo.
RUN python3 - <<'PY'
import sys
path = "/opt/tritonserver/backends/vllm/model.py"
try:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
except FileNotFoundError:
    print(f"[patch] {path} not found - skipping", flush=True)
    sys.exit(0)
patch = (
    '    import os\n'
    '    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.get("model_instance_device_id", "0"))\n'
)
marker = "async def initialize(self, args):"
if 'CUDA_VISIBLE_DEVICES' in src:
    print("[patch] already applied - skipping", flush=True)
    sys.exit(0)
if marker not in src:
    print(f"[patch] marker {marker!r} not found - skipping", flush=True)
    sys.exit(0)
patched = src.replace(marker, marker + "\n" + patch, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(patched)
print("[patch] CUDA_VISIBLE_DEVICES patch applied to vLLM backend", flush=True)
PY
