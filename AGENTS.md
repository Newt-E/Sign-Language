# AGENTS.md — Deteksi Bahasa Isyarat BISINDO

Panduan ini untuk AI coding agent (misal Claude Code) yang membantu implementasi project ini.

## Tentang Project
Project portofolio: sistem deteksi gesture tangan BISINDO (alfabet A-Z & angka 0-9) berbasis landmark tangan, dengan web demo real-time berbasis Streamlit (Python, server-side).

Lihat `PRD-sign-language-bisindo.md` dan `architecture-sign-language-bisindo.md` untuk detail requirement dan desain sistem.

## Prinsip Implementasi
- **Mulai sederhana.** Ini project portofolio dengan scope terbatas (static gesture saja) — hindari over-engineering (misal jangan tambah backend, jangan tambah CNN branch, jangan tambah fitur di luar scope tanpa diminta).
- **Minimal dependency.** Pakai library yang benar-benar dibutuhkan saja.
- **Kode harus mudah dibaca**, karena ini juga jadi bahan portofolio yang mungkin dilihat orang lain (reviewer/rekruter).

## Struktur Project (disarankan)
```
sign-language-bisindo/
├── data/                     # dataset mentah (gitignore kalau besar)
├── notebooks/                # eksplorasi & eksperimen (Jupyter Notebook)
│   ├── extract_landmarks.ipynb  # ekstraksi landmark dari dataset gambar
│   └── train_model.ipynb        # training MLP + evaluasi (accuracy, confusion matrix)
├── training/                 # versi final, hasil rapikan dari notebook (script .py)
│   ├── extract_landmarks.py
│   ├── train_model.py
│   └── evaluate.py
├── web/                      # web demo (Streamlit, server-side)
│   ├── app.py                 # entry point Streamlit: webrtc, MediaPipe Python, load model, UI
│   ├── requirements.txt       # streamlit, streamlit-webrtc, mediapipe, opencv-python, tensorflow, dll
│   └── model/                 # model hasil training (.h5 / SavedModel), tanpa convert
├── PRD-sign-language-bisindo.md
├── architecture-sign-language-bisindo.md
└── README.md
```

## Tahap Kerja (urutan wajib diikuti)
1. Ekstraksi landmark dari dataset — mulai di `notebooks/extract_landmarks.ipynb` untuk eksplorasi (cek jumlah sample per kelas, cek beberapa hasil landmark). Jangan lanjut ke training sebelum tahap ini diverifikasi hasilnya.
2. Training model MLP — mulai di `notebooks/train_model.ipynb`. Arsitektur mengikuti `architecture-sign-language-bisindo.md`, jangan ganti ke CNN atau arsitektur lain tanpa didiskusikan dulu.
3. Evaluasi — wajib tampilkan accuracy dan confusion matrix di notebook sebelum lanjut convert.
4. Setelah pipeline notebook fix dan hasilnya oke, rapikan logic-nya jadi script `.py` di folder `training/` (`extract_landmarks.py`, `train_model.py`, `evaluate.py`) — ini versi final yang reusable, bukan pengganti notebook (notebook tetap disimpan sebagai catatan eksplorasi).
5. Simpan model hasil training (`.h5` atau SavedModel) di `web/model/` — tidak ada langkah konversi format.
6. Bangun web demo — gunakan Streamlit + `streamlit-webrtc` untuk akses kamera, MediaPipe Hands versi Python untuk ekstraksi landmark, model Keras di-load langsung untuk inference. Semua berjalan di server (bukan client-side browser).

## Batasan Teknis (jangan dilanggar)
- **Web demo berbasis Streamlit (Python, server-side).** Inference dan preprocessing (MediaPipe) berjalan di server, bukan di browser. Jangan tambah konversi ke TF.js atau bikin frontend JS terpisah.
- **Model harus MLP berbasis landmark**, bukan CNN berbasis gambar mentah.
- **Tidak menyimpan/merekam data webcam pengguna secara permanen** — frame diproses per request untuk inference saja, tidak disimpan ke disk atau database.

## Konvensi Kode
- Python: ikuti PEP8, gunakan type hints kalau memungkinkan.
- Stack web sepenuhnya Python (Streamlit) — tidak perlu HTML/CSS/JS terpisah kecuali untuk penyesuaian kecil lewat komponen Streamlit yang mendukungnya.
- Commit message singkat dan jelas per tahap kerja (misal: "add landmark extraction script", bukan "update files").
