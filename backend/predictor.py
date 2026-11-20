"""Loader checkpoint dan inferensi model Swin-B untuk klasifikasi real/fake."""

from __future__ import annotations

import json
import os
import time
import math

from pathlib import Path
from typing import Any


import timm
import torch
from PIL import Image
import torchvision.transforms as T
from timm.data import resolve_data_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "Swin-B" / "best.pt"
RESULT_JSON_PATH = REPO_ROOT / "Swin-B" / "result.json"

MODEL_NAME = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"

# Urutan kelas mengikuti dataset: indeks 0 = real, indeks 1 = fake.
# Ini konsisten dengan kolom `prob_fake` pada Swin-B/test_predictions.csv.
CLASS_NAMES = ("real", "fake")
FAKE_INDEX = 1


def _read_model_metadata() -> dict[str, Any]:
    """Ambil ringkasan training (epoch terbaik, akurasi) bila result.json tersedia."""
    if not RESULT_JSON_PATH.is_file():
        return {}

    try:
        with RESULT_JSON_PATH.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return {
        "model": data.get("model", "Swin-B"),
        "params_M": data.get("params_M"),
        "best_epoch": data.get("best_epoch"),
        "val_accuracy": (data.get("val") or {}).get("Accuracy"),
        "test_accuracy": (data.get("test") or {}).get("Accuracy"),
        "test_f1": (data.get("test") or {}).get("F1"),
        "test_auc": (data.get("test") or {}).get("AUC"),
    }


class FakeImagePredictor:
    """Wrapper model Swin-B yang sudah dilatih, siap dipakai untuk inferensi."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        fake_threshold: float | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH
        ).expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint model tidak ditemukan di '{self.model_path}'. "
                "Letakkan best.pt di folder Swin-B/ atau atur environment variable MODEL_PATH."
            )


        raw_threshold = (
            fake_threshold
            if fake_threshold is not None
            else os.getenv("FAKE_THRESHOLD", "0.5")
        )

        try:
            self.fake_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            raise ValueError("FAKE_THRESHOLD harus berupa angka.")

        if not math.isfinite(self.fake_threshold) or not (
            0.0 < self.fake_threshold < 1.0
        ):
            raise ValueError("FAKE_THRESHOLD harus berada di antara 0 dan 1.")

        self.device = torch.device(
            device or os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.class_names = CLASS_NAMES
        self.metadata = _read_model_metadata()
        best_epoch = self.metadata.get("best_epoch")
        self.version = f"swin-b-epoch{best_epoch}" if best_epoch else "swin-b"

        self.model = self._load_model()

        # Mengambil mean dan std yang sama dengan konfigurasi training
        dcfg = resolve_data_config({}, model=self.model)

        # Menyamakan preprocessing dengan notebook training
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=dcfg["mean"],
                std=dcfg["std"]
            )
        ])

    def _load_model(self) -> torch.nn.Module:
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=len(self.class_names),
            drop_path_rate=0.1
        )

        # weights_only=True: hanya tensor yang diizinkan, aman dari eksekusi kode
        # saat checkpoint berasal dari sumber tidak tepercaya.
        state_dict = torch.load(
            self.model_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Jalankan model pada satu gambar dan kembalikan hasil terstruktur."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        started = time.perf_counter()
        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
            logits = self.model(tensor)
            probabilities = torch.softmax(logits.float(), dim=1)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        prob_fake = float(probabilities[FAKE_INDEX])
        prob_real = 1.0 - prob_fake
        label = "fake" if prob_fake >= self.fake_threshold else "real"
        confidence = prob_fake if label == "fake" else prob_real

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "prob_fake": round(prob_fake, 6),
            "prob_real": round(prob_real, 6),
            "threshold": self.fake_threshold,
            "model_version": self.version,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def info(self) -> dict[str, Any]:
        return {
            "name": self.metadata.get("model", "Swin-B"),
            "architecture": MODEL_NAME,
            "version": self.version,
            "params_M": self.metadata.get("params_M"),
            "device": str(self.device),
            "checkpoint": str(self.model_path),
            "checkpoint_size_mb": round(self.model_path.stat().st_size / (1024 * 1024), 1),
            "fake_threshold": self.fake_threshold,
            "val_accuracy": self.metadata.get("val_accuracy"),
            "test_accuracy": self.metadata.get("test_accuracy"),
            "test_f1": self.metadata.get("test_f1"),
            "test_auc": self.metadata.get("test_auc"),
        }
