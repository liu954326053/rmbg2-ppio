"""RMBG-2.0 Serverless inference server for PPIO.

- Loads the `bria-rmbg` ONNX session once on startup (GPU if available).
- Exposes:
    GET  /health      -> liveness/readiness probe
    GET  /            -> service descriptor
    POST /remove      -> multipart image in, PNG with alpha out
"""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rembg import remove

    _get_session()
    # warmup with a tiny dummy image so the first real request has hot kernels
    log.info("Warming up with dummy 64x64 image...")
    dummy = Image.new("RGB", (64, 64), (255, 255, 255))
    buf = io.BytesIO()
    dummy.save(buf, format="PNG")
    t0 = time.time()
    remove(buf.getvalue(), session=_get_session())
    log.info("Warmup done in %.0f ms", (time.time() - t0) * 1000)
    yield


app = FastAPI(
    title="RMBG-2.0 Serverless",
    version="1.0.0",
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


@app.get("/")
async def root():
    return JSONResponse(
        {
            "service": "RMBG-2.0 background removal",
            "model": MODEL_NAME,
            "endpoints": {
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
        from rembg import remove

        t0 = time.time()
        out = remove(data, session=_get_session(), only_mask=bool(only_mask))
        dt = (time.time() - t0) * 1000
        log.info("processed %d bytes in %.0f ms", len(data), dt)
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(500, f"inference failed: {e}")
    return Response(content=out, media_type="image/png")
