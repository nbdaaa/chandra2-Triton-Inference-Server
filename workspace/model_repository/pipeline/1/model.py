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


# System prompt của Chandra 2 (Qwen3.5-VL base)
SYSTEM_PROMPT = (
    "You are an expert document OCR and layout analysis system. "
    "Convert the document image to structured output, preserving the original "
    "layout and text content accurately."
)

DEFAULT_USER_PROMPT = (
    "Convert this document image to markdown. "
    "Preserve the layout, tables, math equations, and all text content exactly as it appears."
)


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.model_name = params.get("vllm_model_name", {}).get(
            "string_value", "chandra_ocr"
        )
        self.max_tokens = int(
            params.get("max_tokens", {}).get("string_value", "12384")
        )

        # URL của vLLM container — ưu tiên env var VLLM_PORT, fallback sang
        # triton_http_url param trong config.pbtxt (được set bởi entrypoint.sh)
        vllm_port = os.environ.get("VLLM_PORT", "8010")
        param_url = params.get("triton_http_url", {}).get(
            "string_value", f"http://127.0.0.1:{vllm_port}"
        )
        self.chat_completions_url = f"{param_url}/v1/chat/completions"

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def _call_engine(self, image_b64: str, user_prompt: str, request_id: str = "") -> str:
        """
        Gọi vLLM container qua OpenAI /v1/chat/completions với SSE streaming.
        vLLM tự apply Qwen3.5 chat template — không cần build raw prompt thủ công.
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
            "temperature": 0.0,
            "stream": True,
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
                raise RuntimeError(f"vLLM HTTP {resp.status}: {detail}")

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

                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token_text = delta.get("content", "")
                        if token_text:
                            full_text += token_text
                            token_count += 1

                        # Cancel check mỗi 5 token qua Redis
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
            raise RuntimeError("vLLM returned no output")

        return full_text.strip()

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                image_b64_tensor = pb_utils.get_input_tensor_by_name(
                    request, "IMAGE_B64"
                )
                if image_b64_tensor is None:
                    raise ValueError("Missing input tensor: IMAGE_B64")

                image_b64 = _to_str(image_b64_tensor.as_numpy().reshape(-1)[0])
                if not image_b64.strip():
                    raise ValueError("IMAGE_B64 must be provided")

                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
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
