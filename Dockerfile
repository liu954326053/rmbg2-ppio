FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/root/.u2net

# Install Python 3.11 via deadsnakes PPA (rembg>=2.0.70 requires py>=3.11)
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common gpg-agent curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
      libgl1 libglib2.0-0 \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && apt-get purge -y --auto-remove software-properties-common gpg-agent \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Pre-download bria-rmbg (RMBG-2.0 ONNX) weights into the image so the
# container has zero external dependencies at runtime / cold start.
RUN python3 -c "from rembg import new_session; new_session('bria-rmbg'); print('model OK')" \
    && du -sh /root/.u2net/ && ls -la /root/.u2net/

COPY server.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
