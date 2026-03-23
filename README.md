# Chandra 2 Triton Inference Server

Repo này giữ cấu trúc tương tự repo tham chiếu `dots.ocr-Triton-Inference-Server`, nhưng đã đổi engine sang `datalab-to/chandra-ocr-2`, bỏ Prometheus/Grafana, và cấu hình vLLM theo hướng `bfloat16` + FlashAttention.

## Cấu trúc

```text
.
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── .env.example
├── api/
│   ├── Dockerfile
│   └── main.py
└── workspace/
    └── model_repository/
        ├── chandra2_engine/
        │   ├── config.pbtxt
        │   └── 1/
        │       └── model.json
        └── pipeline/
            ├── config.pbtxt
            └── 1/
                └── model.py
```

## Thành phần chính

- `chandra2_engine`: model Triton backend `vllm`, chạy trực tiếp checkpoint Chandra 2.
- `pipeline`: model Triton backend `python`, nhận `PROMPT` + `IMAGE_B64` và forward sang engine qua endpoint `/generate_stream`.
- `api`: FastAPI nhận ảnh/PDF, base64 hóa ảnh, render PDF thành ảnh, gọi Triton `pipeline`, lưu tiến độ job PDF vào Redis.
- `docker-entrypoint.sh`: tự sinh `config.pbtxt` cho engine/pipeline theo số GPU có sẵn rồi khởi động Triton.

## Cấu hình model

Mặc định repo dùng:
- `model = datalab-to/chandra-ocr-2`
- `dtype = bfloat16`
- `attention_backend = FLASH_ATTN`
- `mm_encoder_attn_backend = FLASH_ATTN`
- `trust_remote_code = true`

## Chạy nhanh

1. Tạo file `.env` từ `.env.example`
2. Chạy:

```bash
docker compose up -d --build
```

3. Kiểm tra:

- Triton health: `http://localhost:8000/v2/health/ready`
- API docs: `http://localhost:18080/docs`
- RedisInsight: `http://localhost:5540`

## API chính

- `POST /infer-image`: OCR một ảnh, trả kết quả ngay.
- `POST /infer-pdf`: tạo job OCR PDF bất đồng bộ.
- `GET /jobs/{job_id}`: lấy trạng thái và kết quả job PDF.
- `POST /jobs/{job_id}/cancel`: yêu cầu hủy job PDF đang chạy.

## Ghi chú triển khai

- Triton image yêu cầu NVIDIA Container Toolkit và GPU tương thích với FlashAttention.
- Với request OCR chuẩn, giữ `limit_mm_per_prompt.image = 1` trong `model.json`.
- Nếu bạn đã cache model đầy đủ, có thể bật `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1` trong service `triton`.
