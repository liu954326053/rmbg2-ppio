# RMBG-2.0 Serverless (PPIO)

Background-removal inference server powered by **BRIA RMBG-2.0** (ONNX build, redistributed
by the [`rembg`](https://github.com/danielgatis/rembg) project), ready to deploy to
[PPIO Serverless GPU](https://ppio.com/).

## Why this image

- **No HuggingFace token needed** — uses the publicly mirrored ONNX weights.
- **Weights baked in** at build time (`/root/.u2net/bria-rmbg-2.0.onnx`, ~43 MB) —
  no runtime downloads, fast cold start.
- **GPU-accelerated** via `onnxruntime-gpu` + CUDA 12.4 + cuDNN 9.
- **Tiny surface**: FastAPI + uvicorn, `POST /remove`, `GET /health`.

## Build (CI)

GitHub Actions auto-builds on push to `main` and pushes `linux/amd64` to GHCR:

```
ghcr.io/<owner>/<repo>:latest
ghcr.io/<owner>/<repo>:sha-<short-sha>
```

## API

### `GET /health`
```json
{ "status": "ok", "model": "bria-rmbg",
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"] }
```

### `POST /remove`
Multipart form, field `file=<image>`. Optional `only_mask=true` returns a grayscale mask
instead of the alpha-composited PNG.

```bash
curl -X POST "$ENDPOINT/remove" \
  -F "file=@input.jpg" \
  -o output.png
```

## License

RMBG-2.0 weights are released by BRIA AI under **CC BY-NC 4.0** (non-commercial only).
See <https://huggingface.co/briaai/RMBG-2.0>. For commercial use, obtain a commercial
license directly from BRIA.
