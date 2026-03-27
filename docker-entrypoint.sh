#!/bin/bash
set -e

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)

# ── pipeline config.pbtxt ─────────────────────────────────────────────────────
# Kiến trúc 2 container: chỉ load pipeline model, không load chandra_ocr.
# chandra_ocr (vLLM native backend) đã được tách ra container riêng.
# Pipeline chỉ làm HTTP proxy → gọi vLLM container tại VLLM_PORT.
PIPELINE_COUNT=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT:-8010}"

cat > /models/pipeline/config.pbtxt << PBTXT
name: "pipeline"
backend: "python"
max_batch_size: 0

input [
  { name: "PROMPT"     data_type: TYPE_STRING dims: [1] },
  { name: "IMAGE_B64"  data_type: TYPE_STRING dims: [1] },
  { name: "REQUEST_ID" data_type: TYPE_STRING dims: [1] optional: true }
]

output [
  { name: "TEXT" data_type: TYPE_STRING dims: [1] }
]

instance_group [
  { kind: KIND_CPU count: ${PIPELINE_COUNT} }
]

parameters: { key: "max_tokens"      value: { string_value: "12384" } }
parameters: { key: "triton_http_url" value: { string_value: "${VLLM_BASE_URL}" } }
parameters: { key: "vllm_model_name" value: { string_value: "chandra_ocr" } }
PBTXT

echo "[entrypoint] pipeline config.pbtxt updated — vLLM at ${VLLM_BASE_URL}"

# Install redis-py cho cancel support
pip install redis --quiet --no-cache-dir

exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
