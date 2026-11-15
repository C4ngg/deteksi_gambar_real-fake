"""Smoke test cepat untuk endpoint /health dan /predict.

Jalankan dari root repo:
    python backend/smoke_test.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


def make_image(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 240), color).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("health:", health.json())

        response = client.post(
            "/predict",
            files={"file": ("contoh.png", make_image((120, 90, 200)), "image/png")},
        )
        response.raise_for_status()
        print("predict:", response.json())


if __name__ == "__main__":
    main()
