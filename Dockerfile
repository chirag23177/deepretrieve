# Serving image: FastAPI + FAISS + LightGBM. No torch — the request path is
# numpy/FAISS/LightGBM only (user vectors are precomputed at training time
# because the user tower is id-only), so requirements-serve.txt omits it.
# Build AFTER training so artifacts/ and data/processed/ exist:
#   docker build -t deepretrieve .
#   docker run -p 8000:8000 deepretrieve
# See .dockerignore — without it the build context includes .venv and the raw
# MovieLens download (~3 GB) before the first layer runs.

FROM python:3.11-slim

WORKDIR /app

# libgomp is required by LightGBM and FAISS at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY configs/ configs/
COPY src/ src/
# artifacts and the processed split are baked in for a self-contained demo image;
# in production these would be mounted or fetched from object storage
COPY artifacts/ artifacts/
COPY data/processed/ data/processed/

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
