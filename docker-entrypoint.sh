#!/usr/bin/env bash
set -euo pipefail

num_gpus="$(python3 - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(int(os.environ.get('NUM_GPUS', '0')))
PY
)"

if [ "$num_gpus" -eq 0 ]; then
  engine_instance_groups=' {
  count: 1
  kind: KIND_CPU
 }'
else
  engine_instance_groups=""
  for i in $(seq 0 $((num_gpus - 1))); do
    engine_instance_groups+=" {
  count: 1
  kind: KIND_GPU
  gpus: [ $i ]
 }"
    if [ "$i" -lt $((num_gpus - 1)) ]; then
      engine_instance_groups+=",\n"
    fi
  done
fi

printf 'backend: "vllm"\ninstance_group [\n%b\n]\n' "$engine_instance_groups" > /models/chandra2_engine/config.pbtxt

echo "[entrypoint] chandra2_engine config.pbtxt updated: ${num_gpus} GPU-backed instance(s)"
cat /models/chandra2_engine/config.pbtxt

pipeline_count=$(( num_gpus > 0 ? num_gpus : 1 ))
cat > /models/pipeline/config.pbtxt <<EOC
name: "pipeline"
backend: "python"
max_batch_size: 0
input [
  { name: "PROMPT" data_type: TYPE_STRING dims: [1] },
  { name: "IMAGE_B64" data_type: TYPE_STRING dims: [1] },
  { name: "REQUEST_ID" data_type: TYPE_STRING dims: [1] optional: true }
]
output [
  { name: "TEXT" data_type: TYPE_STRING dims: [1] }
]
instance_group [
  { kind: KIND_CPU count: ${pipeline_count} }
]
parameters: { key: "engine_model_name" value: { string_value: "chandra2_engine" } }
parameters: { key: "max_tokens" value: { string_value: "24000" } }
parameters: { key: "triton_http_url" value: { string_value: "http://127.0.0.1:${TRITON_HTTP_PORT:-8000}" } }
EOC

echo "[entrypoint] pipeline config.pbtxt updated: ${pipeline_count} CPU instance(s)"

pip install --quiet --no-cache-dir redis

exec tritonserver \
  --model-repository=/models \
  --http-port=${TRITON_HTTP_PORT:-8000} \
  --grpc-port=${TRITON_GRPC_PORT:-8001} \
  --metrics-port=${TRITON_METRICS_PORT:-8002}
