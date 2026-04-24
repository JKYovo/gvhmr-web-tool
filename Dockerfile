FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runtime.txt requirements-ui.txt setup.py pyproject.toml README.md /app/

RUN python3 -m pip install --retries 20 --timeout 120 --no-build-isolation chumpy && \
    python3 -m pip install --retries 20 --timeout 120 -r requirements-runtime.txt -r requirements-ui.txt

COPY hmr4d /app/hmr4d
COPY tools /app/tools
COPY docs /app/docs

RUN python3 -m pip install --retries 20 --timeout 120 --no-deps .

CMD ["python3", "-m", "hmr4d.service.server", "--host", "0.0.0.0", "--port", "7860"]
