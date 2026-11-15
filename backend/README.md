# Backend API — Deteksi Gambar Real/Fake

Server FastAPI yang memuat `Swin-B/best.pt` dan menyediakan endpoint prediksi.

## Ringkasan model
- Arsitektur: `swin_base_patch4_window7_224.ms_in22k_ft_in1k` (timm), 86,7M parameter
- Input: RGB 224×224, normalisasi ImageNet (mean 0.485/0.456/0.406, std 0.229/0.224/0.225)
- Kelas: `0 = real`, `1 = fake`
- Metrik test: akurasi 98,48% · F1 0,9848 · AUC 0,9978

## Instalasi
```powershell
pip install -r backend/requirements.txt
```

Pastikan checkpoint ada di `Swin-B/best.pt` (tidak ikut repo karena 347 MB). Jika lokasinya
berbeda, atur `MODEL_PATH`.

## Menjalankan server
Dari root repo:
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Atau dari folder `backend/`:
```powershell
uvicorn main:app --port 8000
```

Dokumentasi interaktif: <http://127.0.0.1:8000/docs>

## Environment variable
| Nama | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `Swin-B/best.pt` | Lokasi checkpoint |
| `DEVICE` | `cuda` bila tersedia | `cpu` atau `cuda` |
| `FAKE_THRESHOLD` | `0.5` | Ambang `prob_fake` untuk label fake |
| `MAX_UPLOAD_MB` | `10` | Batas ukuran upload |
| `CORS_ORIGINS` | `localhost:5500, localhost:8000` | Origin yang diizinkan (dipisah koma) |

## Endpoint

### `GET /health`
```json
{
  "status": "ok",
  "model": {
    "name": "Swin-B",
    "architecture": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "version": "swin-b-epoch5",
    "params_M": 86.745274,
    "device": "cpu",
    "checkpoint": "...\\Swin-B\\best.pt",
    "checkpoint_size_mb": 347.1,
    "fake_threshold": 0.5,
    "val_accuracy": 0.9883,
    "test_accuracy": 0.9848,
    "test_f1": 0.9848,
    "test_auc": 0.9978
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
  "label": "fake",
  "confidence": 99.99,
  "prob_fake": 0.999912,
  "prob_real": 0.000088,
  "threshold": 0.5,
  "model_version": "swin-b-epoch5",
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
- `torch.load(..., weights_only=True)` dipakai agar checkpoint tidak bisa mengeksekusi kode.
- Inferensi berjalan di thread terpisah (`asyncio.to_thread`) agar event loop tidak blocking.
- Untuk deployment produksi, tambahkan autentikasi, rate limit, dan reverse proxy HTTPS.
