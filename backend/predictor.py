"""Loader checkpoint dan inferensi model Swin-B untuk klasifikasi real/fake."""

from __future__ import annotations

import json
import os
import math
import time

from pathlib import Path
from typing import Any

import timm
import torch
from PIL import Image
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parent.parent
SWIN_B_DIR = REPO_ROOT / "Swin-B"
DEFAULT_MODEL_PATH = SWIN_B_DIR / "swin_base_best_weights.pth"
INFERENCE_CONFIG_PATH = SWIN_B_DIR / "model_inference_config.json"
TEST_METRICS_PATH = SWIN_B_DIR / "test_metrics.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Konfigurasi '{path}' harus berupa objek JSON.")
    return data


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
                f"Bobot model tidak ditemukan di '{self.model_path}'. "
                "Pastikan swin_base_best_weights.pth tersedia di folder Swin-B/ "
                "atau atur environment variable MODEL_PATH."
            )

        self.inference_config = _read_json(INFERENCE_CONFIG_PATH)
        self.model_name = self.inference_config["model_name"]
        self.image_size = self.inference_config["image_size"]
        self.label_to_id = self.inference_config["label_to_id"]
        if (
            not isinstance(self.image_size, int)
            or self.image_size <= 0
            or not isinstance(self.label_to_id, dict)
            or set(self.label_to_id) != {"Real", "AI-generated"}
            or set(self.label_to_id.values()) != {0, 1}
            or set(self.inference_config["class_names"]) != set(self.label_to_id)
        ):
            raise ValueError(
                "model_inference_config.json memiliki konfigurasi kelas/input yang tidak valid."
            )
        self.class_names = tuple(
            name
            for name, _ in sorted(self.label_to_id.items(), key=lambda item: item[1])
        )
        self.real_index = self.label_to_id["Real"]
        self.fake_index = self.label_to_id["AI-generated"]

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

        self.test_metrics = _read_json(TEST_METRICS_PATH)
        self.version = "swin-b"

        self.model = self._load_model()
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=self.inference_config["mean"],
                std=self.inference_config["std"],
            ),
        ])

    def _load_model(self) -> torch.nn.Module:
        model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=len(self.class_names),
        )
        self.checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(self.checkpoint, dict):
            raise ValueError("Checkpoint model harus berisi state_dict.")
        state_dict = (
            self.checkpoint["model_state_dict"]
            if "model_state_dict" in self.checkpoint
            else self.checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Jalankan model pada satu gambar dan kembalikan hasil terstruktur."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        use_amp = self.device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        if use_amp:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = self.model(tensor)
                probabilities = torch.softmax(logits.float(), dim=1)[0]
        if use_amp:
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000

        prob_fake = float(probabilities[self.fake_index])
        prob_real = float(probabilities[self.real_index])
        class_index = (
            self.fake_index if prob_fake >= self.fake_threshold else self.real_index
        )
        label = self.class_names[class_index]
        confidence = float(probabilities[class_index])

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
            "name": "Swin-B",
            "architecture": self.model_name,
            "version": self.version,
            "device": str(self.device),
            "checkpoint": str(self.model_path),
            "checkpoint_size_mb": round(
                self.model_path.stat().st_size / (1024 * 1024), 1
            ),
            "fake_threshold": self.fake_threshold,
            "class_to_idx": {
                name: index for index, name in enumerate(self.class_names)
            },
            "test_loss": self.test_metrics.get("loss"),
            "test_accuracy": self.test_metrics.get("accuracy"),
            "test_precision": self.test_metrics.get("precision"),
            "test_recall": self.test_metrics.get("recall"),
            "test_f1": self.test_metrics.get("f1"),
            "test_auc": self.test_metrics.get("auc"),
        }
