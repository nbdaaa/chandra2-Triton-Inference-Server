import asyncio
import base64
import io
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import fitz
import numpy as np
import redis
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000")
API_PORT = int(os.getenv("API_PORT", "18080"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PIPELINE_MODEL_NAME = os.getenv("PIPELINE_MODEL_NAME", "pipeline")

DEFAULT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.

Your response must be a single valid JSON object only. Do not output Markdown. Do not wrap the JSON in code fences. Do not add any explanation, note, prefix, suffix, or extra text before or after the JSON. The output must be parseable by a standard JSON parser."""

app = FastAPI(title="Chandra 2 OCR API", version="1.0.0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
executor = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2)))


class ImageInferResponse(BaseModel):
    text: str


class PdfJobResponse(BaseModel):
    job_id: str
    status: str


def _triton_client() -> InferenceServerClient:
    return InferenceServerClient(url=TRITON_URL.replace("http://", "").replace("https://", ""))


def _as_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _normalize_image(file_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return _as_png_bytes(image)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _infer_image_bytes(image_bytes: bytes, prompt: str, request_id: str = "") -> str:
    client = _triton_client()

    prompt_input = InferInput("PROMPT", [1], "BYTES")
    image_input = InferInput("IMAGE_B64", [1], "BYTES")
    request_input = InferInput("REQUEST_ID", [1], "BYTES")

    prompt_input.set_data_from_numpy(np.array([prompt], dtype=object))
    image_input.set_data_from_numpy(np.array([_b64(image_bytes)], dtype=object))
    request_input.set_data_from_numpy(np.array([request_id], dtype=object))

    result = client.infer(
        model_name=PIPELINE_MODEL_NAME,
        inputs=[prompt_input, image_input, request_input],
        outputs=[InferRequestedOutput("TEXT")],
    )
    out = result.as_numpy("TEXT")
    if out is None or len(out) == 0:
        raise RuntimeError("No TEXT output returned from Triton pipeline")
    value = out.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _render_pdf_pages(pdf_bytes: bytes, dpi: int) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    rendered: List[bytes] = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        rendered.append(pix.tobytes("png"))
    doc.close()
    return rendered


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _set_job(job_id: str, payload: Dict[str, Any]) -> None:
    redis_client.set(_job_key(job_id), json.dumps(payload, ensure_ascii=False))


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    raw = redis_client.get(_job_key(job_id))
    return json.loads(raw) if raw else None


def _cancel_key(job_id: str) -> str:
    return f"cancel:{job_id}"


def _pdf_worker(job_id: str, pdf_bytes: bytes, prompt: str, dpi: int) -> None:
    try:
        pages = _render_pdf_pages(pdf_bytes, dpi=dpi)
        state = {
            "job_id": job_id,
            "status": "running",
            "pages_total": len(pages),
            "pages_done": 0,
            "results": [],
            "error": None,
        }
        _set_job(job_id, state)

        for idx, page_bytes in enumerate(pages, start=1):
            if redis_client.exists(_cancel_key(job_id)):
                redis_client.delete(_cancel_key(job_id))
                state["status"] = "cancelled"
                _set_job(job_id, state)
                return

            text = _infer_image_bytes(page_bytes, prompt=prompt, request_id=job_id)
            state["results"].append({"page": idx, "text": text})
            state["pages_done"] = idx
            _set_job(job_id, state)

        state["status"] = "completed"
        _set_job(job_id, state)
    except Exception as exc:
        state = _get_job(job_id) or {"job_id": job_id, "results": []}
        state["status"] = "failed"
        state["error"] = str(exc)
        _set_job(job_id, state)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/infer-image", response_model=ImageInferResponse)
async def infer_image(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
) -> ImageInferResponse:
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")
    image_bytes = _normalize_image(file_bytes)
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(executor, _infer_image_bytes, image_bytes, prompt, "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ImageInferResponse(text=text)


@app.post("/infer-pdf", response_model=PdfJobResponse)
async def infer_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    dpi: int = Form(200),
) -> PdfJobResponse:
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")
    job_id = str(uuid.uuid4())
    initial = {
        "job_id": job_id,
        "status": "queued",
        "pages_total": 0,
        "pages_done": 0,
        "results": [],
        "error": None,
    }
    _set_job(job_id, initial)
    background_tasks.add_task(_pdf_worker, job_id, pdf_bytes, prompt, dpi)
    return PdfJobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, str]:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    redis_client.set(_cancel_key(job_id), "1", ex=3600)
    return {"job_id": job_id, "status": "cancellation_requested"}
