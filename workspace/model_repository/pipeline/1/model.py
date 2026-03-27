import http.client
import json
import os
from urllib.parse import urlparse

import numpy as np
import redis
import triton_python_backend_utils as pb_utils


def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


# ─── System prompt của Chandra 2 ──────────────────────────────────────────────
# Chandra 2 dựa trên Qwen3-VL và được fine-tune cho OCR layout.
# Model expect một system prompt ngắn, không cần JSON schema phức tạp như dots.ocr
# vì Chandra tự output Markdown/HTML/JSON layout theo chuẩn riêng.
SYSTEM_PROMPT = (
    "You are an expert document OCR and layout analysis system. "
    "Convert the document image to structured output, preserving the original "
    "layout and text content accurately."
)

# User prompt mặc định cho task OCR layout — tương đương prompt_type="ocr_layout"
# trong thư viện chandra-ocr chính thức.
DEFAULT_USER_PROMPT = (
    "Convert this document image to markdown. "
    "Preserve the layout, tables, math equations, and all text content exactly as it appears."
)


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.engine_model_name = params.get("engine_model_name", {}).get(
            "string_value", "chandra_ocr"
        )

        # Chandra dùng OpenAI-compatible /v1/chat/completions endpoint.
        # Pipeline gọi endpoint này thay vì /v2/models/.../generate_stream như dots.ocr.
        triton_http_port = os.environ.get("TRITON_HTTP_PORT")
        if triton_http_port:
            base_url = f"http://127.0.0.1:{triton_http_port}"
        else:
            base_url = params.get("triton_http_url", {}).get(
                "string_value", "http://127.0.0.1:8000"
            )

        # vLLM expose /v1/chat/completions tại cùng port với Triton HTTP
        self.chat_completions_url = f"{base_url}/v1/chat/completions"
        self.model_name = params.get("vllm_model_name", {}).get(
            "string_value", "chandra_ocr"
        )
        self.max_tokens = int(
            params.get("max_tokens", {}).get("string_value", "12384")
        )

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def _call_engine(self, image_b64: str, user_prompt: str, request_id: str = "") -> str:
        """
        Gọi vLLM qua OpenAI-compatible /v1/chat/completions với streaming.

        Chandra 2 (Qwen3-VL base) dùng chat template chuẩn:
          <|im_start|>system\\n{system}<|im_end|>
          <|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>
          <|im_start|>assistant\\n

        vLLM tự apply chat template từ tokenizer_config.json — pipeline
        chỉ cần gửi messages list theo OpenAI format với image_url.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        # vLLM nhận ảnh qua image_url với data URI base64
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,   # Deterministic — quan trọng cho OCR
            "stream": True,       # Streaming để cancel mid-generation hoạt động
        }

        body = json.dumps(payload).encode("utf-8")
        parsed = urlparse(self.chat_completions_url)
        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=1800
        )

        try:
            conn.request(
                "POST",
                parsed.path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()

            if resp.status != 200:
                detail = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Engine HTTP {resp.status}: {detail}")

            # Parse SSE stream: "data: {...}\n\n" hoặc "data: [DONE]\n\n"
            full_text = ""
            buf = b""
            token_count = 0

            while True:
                chunk = resp.read(512)
                if not chunk:
                    break
                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    # SSE format: "data: <json>" hoặc "data: [DONE]"
                    if line.startswith(b"data: "):
                        data = line[6:]
                    else:
                        data = line

                    if data == b"[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # OpenAI streaming delta format
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token_text = delta.get("content", "")
                        if token_text:
                            full_text += token_text
                            token_count += 1

                        # Cancel check mỗi 5 token — giống cơ chế dots.ocr
                        if request_id and token_count % 5 == 0:
                            if self._redis.exists(f"cancel:{request_id}"):
                                self._redis.delete(f"cancel:{request_id}")
                                conn.close()
                                raise RuntimeError("Cancelled")

                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason is not None:
                            break

        finally:
            conn.close()

        if not full_text:
            raise RuntimeError("Engine returned no output")

        return full_text.strip()

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                image_b64_tensor = pb_utils.get_input_tensor_by_name(
                    request, "IMAGE_B64"
                )

                if image_b64_tensor is None:
                    raise ValueError("Missing input tensor: IMAGE_B64")

                image_b64 = _to_str(image_b64_tensor.as_numpy().reshape(-1)[0])
                if not image_b64.strip():
                    raise ValueError("IMAGE_B64 must be provided")

                # Nếu PROMPT được truyền và không rỗng, dùng làm user prompt.
                # Ngược lại dùng DEFAULT_USER_PROMPT.
                user_prompt = DEFAULT_USER_PROMPT
                if prompt_tensor is not None:
                    raw = _to_str(prompt_tensor.as_numpy().reshape(-1)[0]).strip()
                    if raw:
                        user_prompt = raw

                request_id_tensor = pb_utils.get_input_tensor_by_name(
                    request, "REQUEST_ID"
                )
                request_id = (
                    _to_str(request_id_tensor.as_numpy().reshape(-1)[0])
                    if request_id_tensor is not None
                    else ""
                )

                text = self._call_engine(image_b64, user_prompt, request_id)

                out_tensor = pb_utils.Tensor(
                    "TEXT",
                    np.array([text], dtype=object),
                )
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out_tensor])
                )

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )

        return responses