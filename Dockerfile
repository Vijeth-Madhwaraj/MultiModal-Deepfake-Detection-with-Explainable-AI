FROM python:3.10-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY transcript/requirements.txt ./transcript-requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

RUN python -m pip install -r requirements.txt

RUN python -m pip install -r transcript-requirements.txt

COPY . .

RUN python -m pip install -e ./Rule_Based_Fusion_Module

CMD ["python", "analysis/analyze_video.py", "--help"]
