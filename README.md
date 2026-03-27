# README

## Monitoring: Prometheus + Grafana

Triton tự động expose metrics Prometheus tại cổng `8002`. Stack monitoring được khởi động cùng Triton bằng một lệnh duy nhất.

### Yêu cầu

* Docker Engine >= 20.10 và Docker Compose v2 (plugin `docker compose`)
* NVIDIA Container Toolkit đã cài và cấu hình

### Bước 1 — Tải dashboard JSON (chỉ làm một lần)

```bash
curl -fsSL "https://grafana.com/api/dashboards/22897/revisions/latest/download" \
  -o monitoring/grafana/dashboards/triton-inference-server.json
```

### Bước 2 — Khởi động toàn bộ stack

Trước khi chạy hệ thống, hãy tạo file `.env` ở thư mục gốc của project. Cấu trúc file `.env`:

```env
# Triton Inference Server
# Triton 26.02 bundle vLLM 0.15.1 — tương thích với Chandra 2 (datalab-to/chandra-ocr-2)
TRITON_IMAGE_TAG=26.02-vllm-python-py3
TRITON_HTTP_PORT=54280
TRITON_GRPC_PORT=8001
TRITON_METRICS_PORT=8002
API_PORT=12345
PDF_INPUT_PATH=/home/user/pdfs
MODEL_REPO_PATH=./workspace/model_repository
HF_CACHE_PATH=~/.cache/huggingface

# Prometheus
PROMETHEUS_IMAGE_TAG=latest
PROMETHEUS_PORT=9090
PROMETHEUS_RETENTION=15d

# Grafana
GRAFANA_IMAGE_TAG=latest
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# Redis
REDIS_IMAGE_TAG=7-alpine
REDIS_PORT=6379

# RedisInsight
REDISINSIGHT_PORT=5540
```

Sau khi tạo file `.env`, chạy docker:

```bash
docker compose up -d
```

Các container sẽ khởi động: **triton**, **prometheus**, **grafana**, **redis**, **redisinsight**, **api**.

### Truy cập

| Dịch vụ      | URL                             | Ghi chú             |
|--------------|---------------------------------|---------------------|
| Triton        | http://localhost:54280          | HTTP inference API  |
| API           | http://localhost:12345          | FastAPI endpoints   |
| Prometheus    | http://localhost:9090           | targets → triton UP |
| Grafana       | http://localhost:3000           | admin / admin       |
| RedisInsight  | http://localhost:5540           | Redis GUI           |

```bash
ssh -p PORT_SSH root@IP_PUBLIC \
  -L 8080:localhost:8080 \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 5540:localhost:5540 \
  -L 54280:localhost:54280
```

Trong Grafana: vào **Dashboards → Triton Inference Server** để xem dashboard NVIDIA Triton.

### Dừng stack

```bash
docker compose down
```

---

## Chạy và test Triton Inference Server cho `chandra_ocr`

Tài liệu này hướng dẫn cách:

1. Khởi chạy Triton Inference Server với model repository đã chuẩn bị sẵn
2. Test OCR bằng `curl`
3. Khắc phục lỗi Docker/NVIDIA runtime thường gặp

---

## 1. Yêu cầu trước khi chạy

* Đã cài **Docker**
* Đã cài **NVIDIA Container Toolkit**
* Host đã nhận GPU (`nvidia-smi` chạy được)
* **NVIDIA Driver version phải >= `570.00`** (yêu cầu của CUDA 13.1 trong Triton 26.02)

* Model repository nằm tại:

```text
workspace/model_repository
```

* Cấu trúc thư mục:

```text
workspace/model_repository/
├── pipeline/          # Python backend — nhận request từ API, gọi chandra_ocr
│   ├── config.pbtxt   # Tự động sinh bởi docker-entrypoint.sh
│   └── 1/
│       └── model.py
└── chandra_ocr/       # vLLM backend — chạy model datalab-to/chandra-ocr-2
    ├── config.pbtxt   # Tự động sinh bởi docker-entrypoint.sh
    └── 1/
        └── model.json
```

> **Lưu ý:** `config.pbtxt` của cả hai model được tự động sinh lại mỗi lần khởi động
> bởi `docker-entrypoint.sh` dựa trên số GPU phát hiện được tại runtime.
> Không cần sửa tay.

---

## 2. Về model Chandra 2

[Chandra 2](https://github.com/datalab-to/chandra) (`datalab-to/chandra-ocr-2`) là OCR model 4B tham số
dựa trên Qwen3-VL, đạt **85.9% trên olmOCR benchmark** (state of the art tính đến 3/2026).

**Khác biệt chính so với dots.ocr (phiên bản trước):**

| | dots.ocr | Chandra 2 |
|---|---|---|
| Model size | ~3B | 4B |
| Architecture | Encoder-decoder (custom) | Decoder-only (Qwen3-VL) |
| Output format | JSON (`bbox`, `category`, `text`) | Markdown (tables, math LaTeX, headings) |
| Chat template | Raw token injection thủ công | OpenAI-compatible messages |
| System prompt | Không có | Có (inject trong `pipeline/model.py`) |
| vLLM API | `/generate_stream` (Triton native) | `/v1/chat/completions` (OpenAI API) |
| Đa ngôn ngữ | Hạn chế | 40+ ngôn ngữ |

Pipeline gọi Chandra 2 qua **OpenAI-compatible `/v1/chat/completions`** mà vLLM expose
tại cùng HTTP port với Triton. vLLM tự apply chat template của Qwen3-VL dựa trên
`tokenizer_config.json` của model — không cần build raw prompt thủ công.

---

## 3. Kiểm tra nhanh driver version

```bash
nvidia-smi
```

hoặc lấy riêng version:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

Triton 26.02 dựa trên CUDA 13.1.1. Tham khảo [NVIDIA CUDA Compatibility Guide][1]
để xác nhận driver version phù hợp.

---

## 4. Khởi chạy Triton Server

Chạy lệnh sau trong terminal thứ nhất:

```bash
HTTP_PORT=54280

docker run --rm --gpus all \
  --network host \
  --ipc=host \
  -v /home/workspace/model_repository:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e TRITON_HTTP_PORT=$HTTP_PORT \
  nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3 \
  tritonserver \
    --model-repository=/models \
    --http-port=$HTTP_PORT \
    --grpc-port=8001 \
    --metrics-port=8002
```

### Giải thích nhanh

* `--gpus all`: cho container dùng GPU
* `--network host`: dùng network của host — cần thiết để pipeline gọi ngược lại
  `/v1/chat/completions` trên cùng host
* `--ipc=host`: chia sẻ IPC (cần thiết cho vLLM shared memory)
* `-v /home/workspace/model_repository:/models`: mount model repository vào container
* `-v ~/.cache/huggingface:/root/.cache/huggingface`: tái sử dụng cache model
* `-e TRITON_HTTP_PORT=$HTTP_PORT`: truyền cổng vào container để `pipeline/model.py`
  tự cấu hình URL gọi `/v1/chat/completions`

> Giữ terminal này mở trong suốt quá trình test.

---

## 5. Kiểm tra server đã sẵn sàng chưa

Mở terminal thứ hai:

```bash
curl http://127.0.0.1:54280/v2/health/ready
```

Kiểm tra model đã được load:

```bash
curl -s http://127.0.0.1:54280/v2/repository/index
```

Bạn nên thấy:

* `pipeline`
* `chandra_ocr`

Kiểm tra vLLM OpenAI endpoint cũng sẵn sàng:

```bash
curl -s http://127.0.0.1:54280/v1/models
```

---

## 6. Test Triton pipeline trực tiếp bằng `curl`

```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.png)

curl -s -X POST http://127.0.0.1:54280/v2/models/pipeline/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": [
      {
        \"name\": \"PROMPT\",
        \"shape\": [1],
        \"datatype\": \"BYTES\",
        \"data\": [\"Convert this document image to markdown.\"]
      },
      {
        \"name\": \"IMAGE_B64\",
        \"shape\": [1],
        \"datatype\": \"BYTES\",
        \"data\": [\"${IMAGE_B64}\"]
      }
    ]
  }"
```

### Giải thích request

Model `pipeline` nhận 2 input bắt buộc và 1 optional:

* `PROMPT`: user prompt cho model. Nếu bỏ trống (`""`), pipeline dùng default prompt
* `IMAGE_B64`: ảnh đã encode base64
* `REQUEST_ID` *(optional)*: UUID để hỗ trợ cancel mid-generation

### Gọi trực tiếp Chandra 2 qua OpenAI API (bỏ qua Triton pipeline)

```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.png)

curl -s -X POST http://127.0.0.1:54280/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"chandra_ocr\",
    \"messages\": [
      {
        \"role\": \"system\",
        \"content\": \"You are an expert document OCR and layout analysis system.\"
      },
      {
        \"role\": \"user\",
        \"content\": [
          {
            \"type\": \"image_url\",
            \"image_url\": { \"url\": \"data:image/png;base64,${IMAGE_B64}\" }
          },
          {
            \"type\": \"text\",
            \"text\": \"Convert this document image to markdown.\"
          }
        ]
      }
    ],
    \"max_tokens\": 12384,
    \"temperature\": 0.0
  }"
```

---

## 7. Kết quả trả về từ Triton

Server sẽ trả JSON dạng:

```json
{
  "model_name": "pipeline",
  "model_version": "1",
  "outputs": [
    {
      "name": "TEXT",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["# Document Title\n\nParagraph text...\n\n| Col1 | Col2 |\n|------|------|\n| ...  | ...  |"]
    }
  ]
}
```

Output là **Markdown** — khác với dots.ocr vốn trả JSON có `bbox` + `category` + `text`.
Chandra 2 preserve structure qua Markdown syntax: headings, tables, LaTeX math (`$...$`),
code blocks, v.v.

---

## 8. API Endpoints (FastAPI)

API chạy tại cổng `API_PORT` (mặc định `12345`), đóng vai trò trung gian giữa client và Triton `pipeline`.

### POST `/infer-image`

Upload một ảnh và nhận kết quả OCR.

**Request** — `multipart/form-data`:

| Field    | Type   | Bắt buộc | Mô tả                                                     |
|----------|--------|-----------|-----------------------------------------------------------|
| `file`   | file   | Có        | File ảnh (PNG, JPG, ...)                                  |
| `prompt` | string | Không     | User prompt. Mặc định: `"Convert this document image to markdown..."` |

```bash
curl -s -X POST http://localhost:12345/infer-image \
  -F "file=@/path/to/image.png" \
  -F "prompt=Convert this document image to markdown."
```

**Response**:

```json
{ "text": "# Title\n\nContent in markdown..." }
```

---

### POST `/infer-pdf`

Upload một file PDF, render từng trang thành ảnh và OCR song song tất cả các trang.

**Request** — `multipart/form-data`:

| Field    | Type   | Bắt buộc | Mô tả             |
|----------|--------|-----------|-------------------|
| `file`   | file   | Có        | File PDF          |
| `prompt` | string | Không     | User prompt       |

```bash
curl -s -X POST http://localhost:12345/infer-pdf \
  -F "file=@/path/to/document.pdf"
```

**Response** — NDJSON stream:

```
{"job_id": "uuid", "status": "processing"}
{"file_path": "document.pdf", "filename": "document", "page_idx": 0, "image_path": "document_page_0000.png", "response": "# Page 1\n\n..."}
{"file_path": "document.pdf", "filename": "document", "page_idx": 2, "image_path": "document_page_0002.png", "response": "..."}
...
{"job_id": "uuid", "status": "completed"}
```

> Các trang không nhất thiết trả về theo thứ tự do được xử lý song song.
> Field `response` chứa Markdown output của từng trang.

---

### POST `/pause-pdf/{job_id}`

Dừng một job đang xử lý. Các trang đã hoàn thành được giữ nguyên trong Redis.
Các trang đang generate token sẽ bị huỷ trong vòng 5 token (cancel key trong Redis).

```bash
curl -s -X POST http://localhost:12345/pause-pdf/<job_id>
```

**Response**:

```json
{
  "job_id": "uuid",
  "status": "paused",
  "pages_saved": 5,
  "pages_remaining": 6
}
```

---

### POST `/resume-pdf/{job_id}`

Tiếp tục một job đang ở trạng thái `paused` hoặc `failed`.

```bash
curl -s -X POST http://localhost:12345/resume-pdf/<job_id>
```

**Response** — NDJSON stream (giống `/infer-pdf`).

Trả `400` nếu job không ở trạng thái `paused` hoặc `failed`.

---

### DELETE `/delete-pdf/{job_id}`

Xoá hoàn toàn một job khỏi Redis và xoá file PDF đã lưu trên disk.
Job phải ở trạng thái `paused`, `completed` hoặc `failed`.

```bash
curl -s -X DELETE http://localhost:12345/delete-pdf/<job_id>
```

**Response**:

```json
{
  "job_id": "uuid",
  "deleted": true
}
```

Trả `400` nếu job đang `processing`. Pause trước, sau đó mới xoá.

---

### GET `/pdf-status`

Liệt kê trạng thái tất cả các job.

```bash
curl -s http://localhost:12345/pdf-status
```

**Response**:

```json
[
  {
    "job_id": "uuid",
    "status": "processing | completed | paused | failed",
    "filename": "document.pdf",
    "ocr_total_pages": 5,
    "ocr_processed_pages": 3,
    "ocr_success_pages": 2,
    "ocr_fail_pages": 1,
    "ocr_remaining_pages": 2
  }
]
```

---

### GET `/pdf-status/{job_id}`

Lấy trạng thái và kết quả của một job cụ thể.

```bash
curl -s http://localhost:12345/pdf-status/<job_id>
```

**Response** (khi `completed`):

```json
{
  "job_id": "uuid",
  "status": "completed",
  "filename": "document.pdf",
  "ocr_total_pages": 5,
  "ocr_processed_pages": 5,
  "ocr_success_pages": 5,
  "ocr_fail_pages": 0,
  "ocr_remaining_pages": 0,
  "result": [
    {
      "file_path": "document.pdf",
      "filename": "document",
      "page_idx": 0,
      "image_path": "document_page_0000.png",
      "response": "# Page 1\n\n..."
    }
  ]
}
```

Trả `404` nếu `job_id` không tồn tại.

---

## 9. Khắc phục lỗi thường gặp

### Docker chưa được cấu hình cho NVIDIA runtime

Nếu khi chạy Triton với `--gpus all` bạn gặp lỗi:

```text
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

Nguyên nhân là Docker daemon chưa được cấu hình để dùng NVIDIA runtime.
Sau khi cài NVIDIA Container Toolkit, chạy `nvidia-ctk runtime configure --runtime=docker`
rồi restart Docker. ([NVIDIA Docs][2])

```bash
# 1) kiểm tra host đã thấy GPU chưa
nvidia-smi

# 2) cài NVIDIA Container Toolkit (nếu chưa cài)
apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# 3) cấu hình Docker dùng NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker

# 4) restart Docker
systemctl restart docker
```

Kiểm tra nhanh:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

### Chandra 2 cần vLLM version cao hơn

Triton 26.02 bundle vLLM 0.15.1. Nếu Chandra 2 yêu cầu version mới hơn,
thêm vào `Dockerfile` trước bước patch:

```dockerfile
RUN pip install "vllm>=0.7.0" --no-cache-dir
```

Sau đó rebuild image:

```bash
docker compose build triton
```

---

### Model chưa được download về cache

Lần đầu khởi động, Triton sẽ tự download `datalab-to/chandra-ocr-2` từ HuggingFace
(~8GB). Nếu server không có internet, download trước trên máy local rồi mount cache:

```bash
# Trên máy có internet
python -c "
from huggingface_hub import snapshot_download
snapshot_download('datalab-to/chandra-ocr-2')
"

# Sau đó copy ~/.cache/huggingface lên server và mount qua HF_CACHE_PATH trong .env
```

Sau khi đã có cache, uncomment trong `.env`:

```env
# HF_HUB_OFFLINE=1
# TRANSFORMERS_OFFLINE=1
```

Và uncomment trong `docker-compose.yml`:

```yaml
# - HF_HUB_OFFLINE=1
# - TRANSFORMERS_OFFLINE=1
```

---

[1]: https://docs.nvidia.com/deploy/cuda-compatibility/index.html "NVIDIA CUDA Compatibility Guide"
[2]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html "Installing the NVIDIA Container Toolkit"