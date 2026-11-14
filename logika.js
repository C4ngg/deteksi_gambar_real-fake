// 1. Pengaturan dasar
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/png", "image/jpg", "image/jpeg", "image/webp", "image/bmp"];

// 2. Elemen halaman
const elements = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  dropContent: document.getElementById("dropContent"),
  preview: document.getElementById("preview"),
  fileInfo: document.getElementById("fileInfo"),
  error: document.getElementById("error"),
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
// Tempatnya model/API
async function detectImage(file) {
  throw new Error(
    "Model deteksinya belum terhubung di 'detectImage()'🗿."
  );
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
  
  // Fungsi perlu mengembalikan: { label: "real" atau "fake", confidence: angka 0-100 }.
  const confidence = Math.min(100, Math.max(0, Math.round(prediction.confidence)));
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