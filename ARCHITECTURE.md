# Architecture — Deteksi Bahasa Isyarat BISINDO

## 1. Overview Arsitektur

Sistem terbagi menjadi dua fase: **training pipeline** (dikerjakan offline di lokal/Colab) dan **inference pipeline** (berjalan di server Python, dibungkus Streamlit).

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (offline)            │
│                                                             │
│  Dataset gambar BISINDO                                    │
│         │                                                   │
│         ▼                                                   │
│  MediaPipe Hands (Python) ──► ekstrak 21 landmark (x,y,z)  │
│         │                                                   │
│         ▼                                                   │
│  Normalisasi koordinat (relatif ke wrist)                  │
│         │                                                   │
│         ▼                                                   │
│  Model MLP (Keras)                                          │
│    Dense(128) → Dropout → Dense(64) → Dropout → Dense(36)  │
│         │                                                   │
│         ▼                                                   │
│  Evaluasi (accuracy, confusion matrix)                      │
│         │                                                   │
│         ▼                                                   │
│  Simpan model (.h5 / SavedModel) — tidak perlu convert      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ (model.h5 / SavedModel)
┌─────────────────────────────────────────────────────────┐
│              INFERENCE PIPELINE (server-side, Streamlit)   │
│                                                             │
│  streamlit-webrtc menangkap stream webcam pengguna          │
│  (browser → server via WebRTC)                              │
│         │                                                   │
│         ▼                                                   │
│  MediaPipe Hands (Python) ──► ekstrak landmark per frame    │
│         │                                                   │
│         ▼                                                   │
│  Normalisasi koordinat (sama seperti training)               │
│         │                                                   │
│         ▼                                                   │
│  Model Keras (.h5/SavedModel) ──► prediksi kelas             │
│         │                                                   │
│         ▼                                                   │
│  Render hasil (huruf/angka + confidence) di UI Streamlit     │
└─────────────────────────────────────────────────────────┘
```

## 2. Komponen

### 2.1 Data & Preprocessing (offline)
- **Input**: dataset gambar BISINDO (alfabet dari sumber publik, angka publik atau self-collected).
- **Tools**: MediaPipe Hands (Python) untuk ekstraksi landmark.
- **Output**: dataset tabular — 63 fitur (21 titik × x,y,z) per sample, dengan label kelas.

### 2.2 Model (offline, training)
- **Framework**: TensorFlow/Keras.
- **Arsitektur**: MLP — Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(36, softmax).
- **Alasan pemilihan**: input berupa landmark koordinat (bukan gambar mentah), sehingga tidak memerlukan CNN untuk ekstraksi fitur visual — MediaPipe sudah menangani deteksi tangan di luar model yang dilatih.

### 2.3 Model Loading
- Model hasil training (`.h5` atau format SavedModel) dimuat langsung oleh aplikasi Streamlit menggunakan Keras/TensorFlow — tidak ada langkah konversi format.

### 2.4 Web Demo (Streamlit, server-side)
- **Stack**: Python + Streamlit, tanpa frontend terpisah (UI didefinisikan lewat komponen Streamlit).
- **Akses kamera**: `streamlit-webrtc` — menangani streaming video dari browser pengguna ke server lewat WebRTC.
- **Ekstraksi landmark real-time**: MediaPipe Hands versi Python, dijalankan di server per frame yang diterima dari `streamlit-webrtc`.
- **Overlay landmark**: titik-titik landmark digambar di atas frame video (pakai OpenCV) sebelum ditampilkan kembali ke pengguna, sebagai visual feedback bahwa tangan terdeteksi.
- **Inference**: model Keras memuat langsung dan menjalankan prediksi di server, hasilnya dikirim balik ke UI Streamlit.
- **UI**: menampilkan video kamera + overlay landmark + hasil prediksi (kelas + confidence score), update tiap frame.
- **Di luar scope saat ini**: fitur rangkai huruf jadi kata (history deteksi) — ditunda sampai versi dasar stabil.

## 3. Alasan Desain Kunci

| Keputusan | Alasan |
|---|---|
| Landmark-based (bukan raw image ke CNN) | Model lebih ringan, robust terhadap background/pencahayaan, cepat dilatih |
| MLP (bukan CNN) | Input sudah berupa vektor koordinat sederhana, bukan gambar — CNN tidak diperlukan |
| Server-side inference (Streamlit, Python) | Model & MediaPipe langsung dipakai dalam format aslinya (tanpa convert), development lebih cepat, cocok untuk demo portofolio |
| MediaPipe untuk deteksi tangan | Pre-trained, akurat; karena training dan inference sama-sama di Python, tidak perlu jaga konsistensi lintas bahasa (Python vs JS) |

## 4. Deployment
- **Hosting**: Streamlit Community Cloud (deploy langsung dari repo GitHub).
- **Backend**: aplikasi Streamlit itu sendiri berfungsi sebagai server — tidak ada layanan backend terpisah yang perlu di-maintain.
- **Catatan latency**: karena webcam diproses lewat WebRTC ke server, respons real-time bergantung pada koneksi dan performa server hosting (beda karakteristik dari inference client-side).
