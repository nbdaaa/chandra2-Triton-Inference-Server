"""
torch_infer/server.py
---------------------
Drop-in replacement for the Triton pipeline container.

Exposes the same Triton-compatible HTTP API that api/main.py expects:
  POST /v2/models/pipeline/infer
  GET  /v2/health/ready

Request body (same Triton format the API already sends):
  {
    "inputs": [
      {"name": "PROMPT",     "shape": [1], "datatype": "BYTES", "data": ["..."]},
      {"name": "IMAGE_B64",  "shape": [1], "datatype": "BYTES", "data": ["<base64>"]},
      {"name": "REQUEST_ID", "shape": [1], "datatype": "BYTES", "data": ["<uuid>"]}
    ]
  }

Response body (same Triton format the API already parses):
  {
    "model_name": "pipeline",
    "model_version": "1",
    "outputs": [
      {"name": "TEXT", "datatype": "BYTES", "shape": [1], "data": ["<model output>"]}
    ]
  }

Cancel: Checks Redis key "cancel:<request_id>" every 5 generated tokens,
        matching the behaviour of the old pipeline/model.py.
"""

import asyncio
import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO

import redis as redis_lib
import torch
from fastapi import FastAPI, Request
from fastapi.responses import Response
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME  = os.environ.get("MODEL_NAME", "datalab-to/chandra-ocr-2")
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "12384"))
REDIS_URL   = os.environ.get("REDIS_URL", "redis://localhost:6379")
SERVER_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))

_PROMPTS_BY_MODEL = {
    "datalab-to/chandra-ocr-2": {
        "system": (
            "You are an expert document OCR and layout analysis system. "
            "Convert the document image to structured output, preserving the original "
            "layout and text content accurately."
        ),
        "user": (
            "Convert this document image to markdown. "
            "Preserve the layout, tables, math equations, and all text content exactly as it appears."
        ),
    },
    "datalab-to/surya-ocr-2": {
        "system": (
            "OCR this image to HTML. Each block is a div with data-label and data-bbox "
            "(x0 y0 x1 y1, normalized 0-1000)."
        ),
        "user": "",
    },
}

_prompts        = _PROMPTS_BY_MODEL.get(MODEL_NAME, _PROMPTS_BY_MODEL["datalab-to/chandra-ocr-2"])
SYSTEM_PROMPT       = _prompts["system"]
DEFAULT_USER_PROMPT = _prompts["user"]

_model                         = None
_processor                     = None
_redis_client: redis_lib.Redis = None
_infer_lock: asyncio.Lock      = None
# Single worker so that concurrent PDF-page requests are serialised on the GPU.
# The asyncio lock above prevents more than one request from reaching the executor
# at a time; max_workers=1 is a safety belt.
_executor = ThreadPoolExecutor(max_workers=1)
_ready    = False


# ── Cancel support ─────────────────────────────────────────────────────────────

class _CancelCriteria(StoppingCriteria):
    """Stop generation when a cancel key appears in Redis (checked every 5 tokens)."""

    def __init__(self, request_id: str, r: redis_lib.Redis) -> None:
        self.request_id = request_id
        self._r         = r
        self._n         = 0
        self.cancelled  = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_) -> bool:
        if not self.request_id:
            return False
        self._n += 1
        if self._n % 5 == 0 and self._r.exists(f"cancel:{self.request_id}"):
            self._r.delete(f"cancel:{self.request_id}")
            self.cancelled = True
            return True
        return False


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_blocking() -> None:
    global _model, _processor

    logger.info("Loading processor from %s", MODEL_NAME)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _processor.tokenizer.padding_side = "left"

    logger.info("Loading model from %s", MODEL_NAME)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    _model.eval()
    logger.info("Model ready on %s", next(_model.parameters()).device)


# ── Inference (runs in thread pool) ───────────────────────────────────────────

def _infer_blocking(image_b64: str, user_prompt: str, request_id: str) -> str:
    pil_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    # Step 1: build text with vision-token placeholders (no image processing yet)
    # enable_thinking=False: disable Qwen3 chain-of-thought to avoid <think> tokens in output
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # Step 2: tokenise text + process image together
    inputs = _processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    input_len       = inputs["input_ids"].shape[-1]
    cancel_criteria = _CancelCriteria(request_id, _redis_client)

    with torch.inference_mode():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            eos_token_id=_processor.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([cancel_criteria]),
        )

    if cancel_criteria.cancelled:
        raise RuntimeError("Cancelled")

    return _processor.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client, _infer_lock, _ready
    _infer_lock   = asyncio.Lock()
    _redis_client = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load_model_blocking)
    _ready = True

    yield

    _redis_client.close()
    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


def _resp(content: dict, status_code: int = 200) -> Response:
    return Response(
        content=json.dumps(content, ensure_ascii=False),
        media_type="application/json",
        status_code=status_code,
    )


@app.get("/v2/health/ready")
async def health_ready():
    if not _ready:
        return _resp({"ready": False}, status_code=503)
    return _resp({"ready": True})


@app.post("/v2/models/pipeline/infer")
async def pipeline_infer(request: Request):
    if not _ready:
        return _resp({"error": "Server not ready"}, status_code=503)

    body       = await request.json()
    inputs_map = {inp["name"]: inp["data"][0] for inp in body.get("inputs", [])}

    image_b64   = inputs_map.get("IMAGE_B64", "")
    user_prompt = inputs_map.get("PROMPT", "").strip() or DEFAULT_USER_PROMPT
    request_id  = inputs_map.get("REQUEST_ID", "")

    if not image_b64:
        return _resp({"error": "IMAGE_B64 input is required"}, status_code=400)

    loop = asyncio.get_event_loop()
    try:
        async with _infer_lock:
            text = await loop.run_in_executor(
                _executor, _infer_blocking, image_b64, user_prompt, request_id
            )
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        return _resp({"error": str(exc)}, status_code=500)

    return _resp({
        "model_name":    "pipeline",
        "model_version": "1",
        "outputs": [{
            "name":     "TEXT",
            "datatype": "BYTES",
            "shape":    [1],
            "data":     [text],
        }],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
