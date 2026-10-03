# Builder: compile omnivoice.cpp (GGUF runtime) with CUDA
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git

WORKDIR /opt/omnivoice.cpp
RUN ./buildcuda.sh && \
    BIN=$(find build -name "omnivoice-tts" -type f | head -1) && \
    echo "Built: $BIN" && \
    cp "$BIN" /opt/omnivoice-tts

# Runtime: Python API server driving the prebuilt binary + Q4 GGUFs by default
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODELS_DIR=/app/models \
    TTS_BINARY=omnivoice-tts \
    MODEL_QUANT=Q4_K_M \
    GGML_BACKEND=CUDA0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/omnivoice-tts /usr/local/bin/omnivoice-tts

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prefetch default quant pair so first request is fast.
# Switch size with --build-arg MODEL_QUANT=Q8_0 (F32|BF16|Q8_0|Q4_K_M);
# runtime override via MODEL_QUANT / MODEL_BASE_FILE / MODEL_TOKENIZER_FILE env.
ARG MODEL_QUANT=Q4_K_M
RUN python -c "import os; from huggingface_hub import hf_hub_download; \
    q=os.environ.get('MODEL_QUANT','Q4_K_M'); \
    [hf_hub_download(repo_id='Serveurperso/OmniVoice-GGUF', \
    filename=f'omnivoice-{p}-{q}.gguf', local_dir='/app/models') \
    for p in ('base','tokenizer')]"

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
