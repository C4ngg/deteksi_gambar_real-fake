// 1. Pengaturan dasar
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/png", "image/jpg", "image/jpeg", "image/webp", "image/bmp"];

// Alamat backend FastAPI. Bisa dioverride lewat window.API_BASE_URL
// sebelum logika.js dimuat, misalnya saat deploy ke domain lain.
const API_BASE_URL = (window.API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const PREDICT_TIMEOUT_MS = 30000;
const HEALTH_TIMEOUT_MS = 4000;

// 2. Elemen halaman
const elements = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  dropContent: document.getElementById("dropContent"),
  preview: document.getElementById("preview"),
  fileInfo: document.getElementById("fileInfo"),
  error: document.getElementById("error"),
  serverStatus: document.getElementById("serverStatus"),
  serverStatusText: document.getElementById("serverStatusText"),
  analyzeButton: document.getElementById("analyzeBtn"),
  resetButton: document.getElementById("resetBtn"),
  loading: document.getElementById("loading"),
  result: document.getElementById("result"),
  resultBadge: document.getElementById("resultBadge"),
  resultText: document.getElementById("resultText"),
  resultBar: document.getElementById("resultBar"),
  resultNote: document.getElementById("resultNote")
};

const missingElements = Object.entries(elements)
  .filter(([, element]) => !element)
  .map(([name]) => name);

if (missingElements.length > 0) {
  throw new Error(`Elemen HTML tidak ditemukan: ${missingElements.join(", ")}`);
}

// 3. Data yang sedang digunakan
let selectedFile = null;
let previewUrl = null;
let fileRequestId = 0;
let isAnalyzing = false;

// 4. Bantuan untuk menampilkan dan menyembunyikan elemen
function show(element) {
  element.classList.remove("hidden");
}

function hide(element) {
  element.classList.add("hidden");
}

// Status koneksi ke backend, ditampilkan di pojok kartu.
function setServerStatus(state, text) {
  elements.serverStatus.className = `ready-state is-${state}`;
  elements.serverStatusText.textContent = text;
  elements.serverStatus.title = `Server: ${API_BASE_URL}`;
}

async function checkServer() {
  setServerStatus("checking", "MEMERIKSA");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      signal: controller.signal,
      headers: { Accept: "application/json" }
    });

    if (!response.ok) {
      throw new Error(`Server merespons dengan status ${response.status}.`);
    }

    const data = await response.json();

    if (!data || data.status !== "ok") {
      throw new Error("Server belum siap.");
    }

    setServerStatus("online", "SIAP");
    return true;
  } catch (error) {
    setServerStatus("offline", "OFFLINE");
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function showError(message) {
  elements.error.textContent = message;
  show(elements.error);
}

function clearError() {
  elements.error.textContent = "";
  hide(elements.error);
}

function formatFileSize(sizeInBytes) {
  const sizeInMegabytes = sizeInBytes / (1024 * 1024);

  if (sizeInMegabytes >= 1) {
    return `${sizeInMegabytes.toFixed(1)} MB`;
  }

  return `${Math.max(1, Math.round(sizeInBytes / 1024))} KB`;
}

// 5. Reset pilihan gambar dan hasil sebelumnya
function resetPage() {
  fileRequestId += 1;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  selectedFile = null;
  isAnalyzing = false;
  elements.fileInput.value = "";
  elements.fileInput.disabled = false;
  elements.preview.removeAttribute("src");
  elements.fileInfo.textContent = "";
  elements.resultText.textContent = "";
  elements.resultNote.textContent = "";
  elements.resultBadge.className = "badge";
  elements.resultBar.className = "bar-fill";
  elements.resultBar.style.width = "0%";
  elements.analyzeButton.disabled = true;
  elements.resetButton.disabled = true;

  show(elements.dropContent);
  hide(elements.preview);
  hide(elements.fileInfo);
  hide(elements.loading);
  hide(elements.result);
  clearError();
}

// 6. Periksa file lalu tampilkan pratinjau
function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    throw new Error("Format tidak didukung. Pilih gambar PNG, JPG, JPEG, WEBP, atau BMP.");
  }

  if (file.size > MAX_FILE_SIZE) {
    throw new Error("Ukuran gambar melebihi batas 10 MB.");
  }
}

function selectFile(file) {
  if (!file || isAnalyzing) {
    return;
  }

  resetPage();

  try {
    validateFile(file);
  } catch (error) {
    elements.fileInput.value = "";
    showError(error.message);
    return;
  }

  const requestId = fileRequestId;
  const newPreviewUrl = URL.createObjectURL(file);
  const image = new Image();

  image.onload = function () {
    if (requestId !== fileRequestId) {
      URL.revokeObjectURL(newPreviewUrl);
      return;
    }

    previewUrl = newPreviewUrl;
    selectedFile = file;
    elements.preview.src = previewUrl;
    elements.fileInfo.textContent = `${file.name} • ${formatFileSize(file.size)}`;
    elements.analyzeButton.disabled = false;
    elements.resetButton.disabled = false;

    hide(elements.dropContent);
    show(elements.preview);
    show(elements.fileInfo);
  };

  image.onerror = function () {
    URL.revokeObjectURL(newPreviewUrl);

    if (requestId === fileRequestId) {
      showError("Gambar tidak dapat dibaca. Coba pilih file gambar lain.");
    }
  };

  image.src = newPreviewUrl;
}

// 7. Bagian deteksi
// Backend FastAPI mengembalikan { label, confidence, prob_fake, prob_real, ... }.
async function readErrorDetail(response) {
  try {
    const data = await response.json();

    if (typeof data.detail === "string") {
      return data.detail;
    }

    if (Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) {
      return `Permintaan ditolak: ${data.detail[0].msg}`;
    }
  } catch (error) {
    // Respons bukan JSON, pakai pesan umum di bawah.
  }

  return `Server mengembalikan kesalahan (${response.status}).`;
}

async function detectImage(file) {
  const formData = new FormData();
  formData.append("file", file, file.name);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PREDICT_TIMEOUT_MS);

  let response;

  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal
    });
  } catch (error) {
    setServerStatus("offline", "OFFLINE");

    if (error.name === "AbortError") {
      throw new Error("Analisis melebihi batas waktu. Coba lagi dengan gambar yang lebih kecil.");
    }

    throw new Error(
      `Tidak dapat terhubung ke server deteksi di ${API_BASE_URL}. Pastikan backend sudah berjalan.`
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    if (response.status === 503) {
      setServerStatus("offline", "OFFLINE");
    } else {
      setServerStatus("online", "SIAP");
    }

    throw new Error(await readErrorDetail(response));
  }

  const prediction = await response.json().catch(function () {
    return null;
  });

  if (
    !prediction ||
    (prediction.label !== "real" && prediction.label !== "fake") ||
    !Number.isFinite(prediction.confidence)
  ) {
    throw new Error("Format hasil deteksi dari server tidak valid.");
  }

  setServerStatus("online", "SIAP");

  // Hanya kontrak yang dipakai showResult yang diteruskan.
  return { label: prediction.label, confidence: prediction.confidence };
}

function showResult(prediction) {
  const validLabels = ["real", "fake"];
  
  if (
    !prediction ||
    !validLabels.includes(prediction.label) ||
    !Number.isFinite(prediction.confidence)
  ) {
    throw new Error("Format hasil deteksi tidak valid.");
  }
  
  // Fungsi perlu mengembalikan: { label: "real" atau "fake", confidence: angka 0-99 }.
  const confidence = Math.min(99, Math.max(0, Math.round(prediction.confidence)));
  const isReal = prediction.label === "real";

  elements.resultBadge.className = `badge ${prediction.label}`;
  elements.resultBadge.textContent = isReal ? "REAL" : "FAKE";
  elements.resultText.textContent = isReal
    ? "Gambar terindikasi asli."
    : "Gambar terindikasi palsu.";
  elements.resultBar.className = `bar-fill ${prediction.label}`;
  elements.resultBar.style.width = `${confidence}%`;
  elements.resultNote.textContent = `Tingkat keyakinan model: ${confidence}%.`;

  show(elements.result);
}

async function analyzeSelectedImage() {
  if (!selectedFile || isAnalyzing) {
    return;
  }

  clearError();
  hide(elements.result);
  show(elements.loading);
  isAnalyzing = true;
  elements.analyzeButton.disabled = true;
  elements.resetButton.disabled = true;
  elements.fileInput.disabled = true;

  try {
    const prediction = await detectImage(selectedFile);
    showResult(prediction);
  } catch (error) {
    showError(error.message || "Terjadi kesalahan saat menganalisis gambar.");
  } finally {
    isAnalyzing = false;
    hide(elements.loading);
    elements.fileInput.disabled = false;
    elements.analyzeButton.disabled = !selectedFile;
    elements.resetButton.disabled = !selectedFile;
  }
}

// 8. Hubungkan aksi pengguna
elements.fileInput.addEventListener("change", function () {
  const file = elements.fileInput.files[0];
  selectFile(file);
});

elements.dropzone.addEventListener("click", function (event) {
  if (event.target !== elements.fileInput && !isAnalyzing) {
    elements.fileInput.click();
  }
});

elements.dropzone.addEventListener("keydown", function (event) {
  if ((event.key === "Enter" || event.key === " ") && !isAnalyzing) {
    event.preventDefault();
    elements.fileInput.click();
  }
});

elements.dropzone.addEventListener("dragover", function (event) {
  event.preventDefault();

  if (!isAnalyzing) {
    elements.dropzone.classList.add("dragover");
  }
});

elements.dropzone.addEventListener("dragleave", function () {
  elements.dropzone.classList.remove("dragover");
});

elements.dropzone.addEventListener("drop", function (event) {
  event.preventDefault();
  elements.dropzone.classList.remove("dragover");

  if (!isAnalyzing && event.dataTransfer.files.length > 0) {
    selectFile(event.dataTransfer.files[0]);
  }
});

elements.analyzeButton.addEventListener("click", analyzeSelectedImage);
elements.resetButton.addEventListener("click", resetPage);

// 9. Cek ketersediaan backend saat halaman dibuka
setServerStatus("checking", "MEMERIKSA");
checkServer();