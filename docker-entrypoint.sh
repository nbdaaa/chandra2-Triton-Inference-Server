#!/bin/bash
set -e

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

# Build one instance_group block per GPU with explicit GPU assignment.
if [ "$NUM_GPUS" -eq 0 ]; then
    INSTANCE_GROUPS="  {\n    count: 1\n    kind: KIND_CPU\n  }"
else
    INSTANCE_GROUPS=""
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        INSTANCE_GROUPS+="  {\n    count: 1\n    kind: KIND_GPU\n    gpus: [ $i ]\n  }"
        if [ $i -lt $((NUM_GPUS - 1)) ]; then
            INSTANCE_GROUPS+=",\n"
        fi
    done
fi

# ── Chandra 2: tạo config.pbtxt cho chandra_ocr (thay thế dots_ocr) ──────────
printf "backend: \"vllm\"\ninstance_group [\n%b\n]\n" "$INSTANCE_GROUPS" \
    > /models/chandra_ocr/config.pbtxt

echo "[entrypoint] chandra_ocr config.pbtxt updated: $NUM_GPUS instance(s), one per GPU"
cat /models/chandra_ocr/config.pbtxt

# ── Pipeline: scale CPU instances theo số GPU ─────────────────────────────────
PIPELINE_COUNT=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))

# Chú ý: engine_model_name đổi sang "chandra_ocr"
# vllm_model_name phải trùng với "model" field trong model.json của chandra_ocr
# (vLLM dùng tên này để serve qua /v1/chat/completions)
cat > /models/pipeline/config.pbtxt << EOF
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

parameters: { key: "engine_model_name" value: { string_value: "chandra_ocr" } }
parameters: { key: "max_tokens"        value: { string_value: "12384" } }
parameters: { key: "triton_http_url"   value: { string_value: "http://127.0.0.1:8000" } }
parameters: { key: "vllm_model_name"   value: { string_value: "chandra_ocr" } }
EOF

echo "[entrypoint] pipeline config.pbtxt updated: ${PIPELINE_COUNT} CPU instance(s)"

# Install redis-py for pipeline cancel support
pip install redis --quiet --no-cache-dir

# Start Triton
exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}