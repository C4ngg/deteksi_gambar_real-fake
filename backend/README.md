# Backend API — Deteksi Gambar Real/AI-generated

Server FastAPI yang memuat `Swin-B/swin_base_best_weights.pth` dan menyediakan endpoint prediksi.

## Ringkasan model
- Arsitektur: `swin_base_patch4_window7_224.ms_in22k_ft_in1k` (timm), 86,7M parameter
- Input: RGB 224×224, normalisasi ImageNet (mean 0.485/0.456/0.406, std 0.229/0.224/0.225)
- Kelas: `0 = Real`, `1 = AI-generated` (mengikuti `model_inference_config.json`)
- Metrik test: akurasi 98,02% · precision 97,84% · recall 98,20% · F1 0,9802 · AUC 0,9949

## Instalasi
```powershell
pip install -r backend/requirements.txt
```

Pastikan file `Swin-B/swin_base_best_weights.pth`, `Swin-B/model_inference_config.json`,
dan `Swin-B/test_metrics.json` tersedia. Bobot besar diabaikan Git. Jika bobot ada di
lokasi berbeda, atur `MODEL_PATH`.

## Menjalankan server
Dari root repo:
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Kalau error, Jalankan ini
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Atau dari folder `backend/`:
```powershell
uvicorn main:app --port 8000
```

Dokumentasi interaktif: <http://127.0.0.1:8000/docs>

## Environment variable
| Nama | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `Swin-B/swin_base_best_weights.pth` | Lokasi bobot model |
| `DEVICE` | `cuda` bila tersedia | `cpu` atau `cuda` |
| `FAKE_THRESHOLD` | `0.5` | Ambang `prob_fake` untuk label fake |
| `MAX_UPLOAD_MB` | `10` | Batas ukuran upload |
| `CORS_ORIGINS` | `localhost:5500, localhost:8000, null` | Origin yang diizinkan (dipisah koma). `null` untuk halaman `file://` |

## Endpoint

### `GET /health`
```json
{
  "status": "ok",
  "model": {
    "name": "Swin-B",
    "architecture": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "version": "swin-b",
    "device": "cpu",
    "checkpoint": "...\\Swin-B\\swin_base_best_weights.pth",
    "fake_threshold": 0.5,
    "class_to_idx": {
      "Real": 0,
      "AI-generated": 1
    },
    "test_loss": 0.1701247949,
    "test_accuracy": 0.9801534356,
    "test_precision": 0.9783909574,
    "test_recall": 0.981981982,
    "test_f1": 0.9801831807,
    "test_auc": 0.9948990337
  }
}
```

### `POST /predict`
Multipart form dengan field `file`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@contoh.jpg"
```

```json
{
  "label": "AI-generated",
  "confidence": 99.99,
  "prob_fake": 0.999912,
  "prob_real": 0.000088,
  "threshold": 0.5,
  "model_version": "swin-b",
  "elapsed_ms": 412.7
}
```

`confidence` adalah keyakinan terhadap label terpilih dalam persen — sesuai kontrak
`showResult()` di `frontend/logika.js`.

### Kode error
| Status | Arti |
|---|---|
| 400 | File kosong / bukan gambar valid |
| 413 | Melebihi `MAX_UPLOAD_MB` |
| 415 | Content-Type tidak didukung |
| 503 | Model belum selesai dimuat |

## Smoke test
```powershell
python backend/smoke_test.py
```

## Catatan
- Model dimuat sekali saat startup; request pertama tidak memperhitungkan waktu load.
- Konfigurasi ukuran input, kelas, dan normalisasi dibaca dari `model_inference_config.json`.
- Metrik pada `/health` dibaca dari `test_metrics.json`.
- Bobot dimuat dengan `weights_only=True`.
- Inferensi berjalan di thread terpisah (`asyncio.to_thread`) agar event loop tidak blocking.
- Untuk deployment produksi, tambahkan autentikasi, rate limit, dan reverse proxy HTTPS.
