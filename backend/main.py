"""API FastAPI untuk deteksi gambar real/fake memakai model Swin-B."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

try:  # dijalankan sebagai package: uvicorn backend.main:app
    from .predictor import FakeImagePredictor
except ImportError:  # dijalankan dari dalam folder backend: uvicorn main:app
    from predictor import FakeImagePredictor

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
}

# Batas aman agar gambar raksasa tidak menghabiskan memori server.
Image.MAX_IMAGE_PIXELS = 40_000_000


class PredictionResponse(BaseModel):
    label: Literal["real", "fake"]
    confidence: float = Field(
        ..., ge=0, le=100, description="Keyakinan model terhadap label terpilih (persen)"
    )
    prob_fake: float = Field(..., ge=0, le=1)
    prob_real: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., description="Ambang probabilitas fake yang dipakai")
    model_version: str
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model dimuat sekali saat server start, bukan di setiap request.
    app.state.predictor = FakeImagePredictor()
    yield
    app.state.predictor = None


app = FastAPI(
    title="Deteksi Gambar Real/Fake",
    description="API inferensi model Swin-B untuk membedakan gambar asli dan hasil AI.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "name": "Deteksi Gambar Real/Fake API",
        "docs": "/docs",
        "endpoints": ["GET /health", "POST /predict"],
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    return HealthResponse(status="ok", model=predictor.info())


async def _read_upload(file: UploadFile) -> bytes:
    """Baca isi file sambil menegakkan batas ukuran secara streaming."""
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Ukuran gambar melebihi batas {MAX_UPLOAD_MB} MB."
        )

    chunks: list[bytes] = []
    total = 0

    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail=f"Ukuran gambar melebihi batas {MAX_UPLOAD_MB} MB."
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    return b"".join(chunks)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    predictor: FakeImagePredictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Format tidak didukung. Gunakan gambar PNG, JPG, JPEG, WEBP, atau BMP.",
        )

    data = await _read_upload(file)

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=400, detail="File tidak dapat dibaca sebagai gambar yang valid."
        ) from error

    # Inferensi CPU/GPU bersifat blocking, jadi jalankan di thread terpisah
    # agar event loop FastAPI tidak macet.
    result = await asyncio.to_thread(predictor.predict, rgb_image)
    return PredictionResponse(**result)
