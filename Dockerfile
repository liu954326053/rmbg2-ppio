FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    U2NET_HOME=/root/.u2net

# Ubuntu 24.04 ships Python 3.12 (satisfies rembg>=2.0.70 which needs py>=3.11).
# Swap to the Azure-hosted apt mirror (runners live on Azure) + add retries to
# avoid sporadic archive.ubuntu.com connection timeouts inside the build.
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
      [ -f "$f" ] && sed -i \
        -e 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' \
        -e 's|http://security.ubuntu.com|http://azure.archive.ubuntu.com|g' "$f" || true; \
    done; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends \
      python3 python3-pip curl ca-certificates \
      libgl1 libglib2.0-0t64; \
    rm -rf /var/lib/apt/lists/*

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
