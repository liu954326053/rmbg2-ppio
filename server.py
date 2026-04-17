"""RMBG-2.0 Serverless inference server for PPIO.

- Loads the `bria-rmbg` ONNX session once on startup (GPU if available).
- Exposes:
    GET  /           -> Gradio web UI (matches the HF Space)
    GET  /health     -> liveness/readiness probe (kept at root for PPIO)
    GET  /info       -> service descriptor (JSON)
    POST /remove     -> REST API: multipart image in, PNG with alpha out
"""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rmbg")

MODEL_NAME = "bria-rmbg"

# Try CUDA first, fall back to CPU so local dev also works.
PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
    "CPUExecutionProvider",
]

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        log.info("Loading %s session...", MODEL_NAME)
        _session = new_session(MODEL_NAME, providers=PROVIDERS)
        try:
            active = _session.inner_session.get_providers()
        except Exception:
            active = "unknown"
        log.info("Session ready. Active providers: %s", active)
    return _session


def _remove_bytes(data: bytes, only_mask: bool = False) -> bytes:
    from rembg import remove

    t0 = time.time()
    out = remove(data, session=_get_session(), only_mask=only_mask)
    log.info("processed %d bytes in %.0f ms", len(data), (time.time() - t0) * 1000)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_session()
    log.info("Warming up with dummy 64x64 image...")
    dummy = Image.new("RGB", (64, 64), (255, 255, 255))
    buf = io.BytesIO()
    dummy.save(buf, format="PNG")
    t0 = time.time()
    _remove_bytes(buf.getvalue())
    log.info("Warmup done in %.0f ms", (time.time() - t0) * 1000)
    yield


app = FastAPI(
    title="RMBG-2.0 Serverless",
    version="1.1.0",
    description="Background removal (BRIA RMBG-2.0 ONNX) on PPIO Serverless GPU.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    try:
        s = _get_session()
        providers = s.inner_session.get_providers()
    except Exception as e:
        raise HTTPException(503, f"session not ready: {e}")
    return {"status": "ok", "model": MODEL_NAME, "providers": providers}


@app.get("/info")
async def info():
    return JSONResponse(
        {
            "service": "RMBG-2.0 background removal",
            "model": MODEL_NAME,
            "endpoints": {
                "GET /": "Gradio web UI",
                "POST /remove": "multipart form: file=<image>; optional only_mask=true",
                "GET /health": "liveness probe",
            },
            "license": "CC BY-NC 4.0 (non-commercial)",
        }
    )


@app.post("/remove")
async def remove_bg(
    file: UploadFile = File(...),
    only_mask: bool = Form(False),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, f"Expected image/*, got {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        out = _remove_bytes(data, only_mask=bool(only_mask))
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(500, f"inference failed: {e}")
    return Response(content=out, media_type="image/png")


# -------- Gradio UI (mirrors the BRIA HF Space layout) --------

def _predict(img: Image.Image | None):
    if img is None:
        return None, None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    out_bytes = _remove_bytes(buf.getvalue())
    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    return out_img, out_bytes


DESCRIPTION = """
# RMBG-2.0 for background removal

Background removal model developed by [BRIA.AI](https://bria.ai/), trained on a carefully
selected dataset, and is available as an open-source model for **non-commercial** use.

For testing upload your image and wait.

[Model card](https://huggingface.co/briaai/RMBG-2.0) · [Blog](https://bria.ai/)

API endpoint: `POST /remove` (multipart `file=<image>`) · served from this same URL.
"""

with gr.Blocks(title="RMBG-2.0", theme=gr.themes.Soft()) as ui:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="input image", sources=["upload", "clipboard"], height=420)
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            out_preview = gr.Image(type="pil", label="RMBG-2.0", image_mode="RGBA", height=420)
            out_file = gr.File(label="output png file")

    submit_btn.click(_predict, inputs=inp, outputs=[out_preview, out_file], api_name=False)
    clear_btn.click(lambda: (None, None, None), outputs=[inp, out_preview, out_file], api_name=False)

ui.queue(default_concurrency_limit=2, max_size=8)

app = gr.mount_gradio_app(app, ui, path="/")
