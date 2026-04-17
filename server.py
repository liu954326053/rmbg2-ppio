"""RMBG-2.0 Serverless on PPIO — Gradio-only deployment.

Layout mirrors the HF Space `briaai/BRIA-RMBG-2.0`:
- GET  /                   -> Gradio web UI (matches the Space visually)
- POST /gradio_api/...     -> auto-generated Gradio API (use `gradio_client`)
- GET  /health             -> small JSON probe for PPIO (NOT exposed in Gradio)

Usage from another project (identical to calling a HF Space):
    from gradio_client import Client, handle_file
    client = Client("https://<your-endpoint>.runsync.serverless.ppinfra.com/")
    image_path, file_path = client.predict(handle_file("photo.jpg"), api_name="/predict")
"""
from __future__ import annotations

import io
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rmbg")

MODEL_NAME = "bria-rmbg"

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


def _remove_bytes(data: bytes) -> bytes:
    from rembg import remove

    t0 = time.time()
    out = remove(data, session=_get_session())
    log.info("processed %d bytes in %.0f ms", len(data), (time.time() - t0) * 1000)
    return out


# -------- Gradio handler --------

def predict(image: Image.Image | None):
    """Remove background. Returns (PIL RGBA preview, PNG filepath)."""
    if image is None:
        raise gr.Error("Please upload an image.")

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    out_bytes = _remove_bytes(buf.getvalue())
    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")

    # Persist a file so gr.File can offer download / gradio_client can pull it.
    tmp = Path(tempfile.mkdtemp(prefix="rmbg_")) / "output.png"
    tmp.write_bytes(out_bytes)
    return out_img, str(tmp)


DESCRIPTION = """
# RMBG-2.0 for background removal

Background removal model developed by [BRIA.AI](https://bria.ai/), trained on a carefully
selected dataset, and is available as an open-source model for **non-commercial** use.

For testing upload your image and wait.

[Model card](https://huggingface.co/briaai/RMBG-2.0) · [Blog](https://bria.ai/) ·
[ComfyUI Node](https://github.com/1038lab/ComfyUI-RMBG) ·
[Purchase weights for commercial use](https://bria.ai/)
"""

with gr.Blocks(title="RMBG-2.0", theme=gr.themes.Soft()) as ui:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(
                type="pil",
                label="input image",
                sources=["upload", "clipboard"],
                height=420,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            out_preview = gr.Image(
                type="pil",
                label="RMBG-2.0",
                image_mode="RGBA",
                height=420,
            )
            out_file = gr.File(label="output png file")

    submit_btn.click(
        fn=predict,
        inputs=inp,
        outputs=[out_preview, out_file],
        api_name="predict",
    )
    clear_btn.click(
        fn=lambda: (None, None, None),
        outputs=[inp, out_preview, out_file],
        api_name=False,
    )

ui.queue(default_concurrency_limit=2, max_size=16)


# -------- FastAPI wrapper so we can expose a /health probe for PPIO --------

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
    version="2.0.0",
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


# Mount Gradio at root — this preserves our FastAPI routes (/health, /docs)
# and serves the UI + Gradio API (/gradio_api/...) everywhere else.
app = gr.mount_gradio_app(app, ui, path="/")
