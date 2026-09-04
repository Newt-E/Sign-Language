# Implementation Plan — Deteksi Bahasa Isyarat BISINDO

## Tahap 1: Persiapan Dataset
- [ ] Download dataset alfabet BISINDO (Kaggle: `achmadnoer/alfabet-bisindo` atau `agungmrf/indonesian-sign-language-bisindo`, atau Roboflow).
- [ ] Cek ketersediaan dataset angka BISINDO publik; jika tidak ada, rencanakan pengumpulan mandiri (10 kelas, ambil beberapa variasi background/pencahayaan per kelas).
- [ ] Susun struktur folder dataset per kelas (misal `data/A/`, `data/B/`, ..., `data/0/`, `data/1/`, ...).

## Tahap 2: Ekstraksi Landmark (Python)
- [ ] Install `mediapipe`, `opencv-python`.
- [ ] Loop semua gambar di dataset, jalankan MediaPipe Hands, ambil 21 landmark (x, y, z) → 63 fitur per sample.
- [ ] Normalisasi koordinat (misal kurangi posisi wrist sebagai origin, lalu scale).
- [ ] Simpan hasil sebagai file tabular (CSV/NumPy array) dengan kolom fitur + label kelas.
- [ ] Buang sample yang gagal terdeteksi tangannya (log jumlahnya untuk cek kualitas dataset).

## Tahap 3: Training Model (Keras)
- [ ] Split data: train/validation/test (perhatikan tidak ada kebocoran antar split kalau dataset dari beberapa sumber/orang yang sama).
- [ ] Bangun arsitektur MLP:
  ```
  Input(63) → Dense(128, relu) → Dropout(0.3)
            → Dense(64, relu) → Dropout(0.3)
            → Dense(36, softmax)
  ```
- [ ] Compile dengan optimizer Adam, loss `categorical_crossentropy`.
- [ ] Training dengan callback EarlyStopping (monitor val_loss) biar tidak overfit.
- [ ] Simpan model terbaik (`model.h5` atau format SavedModel).

## Tahap 4: Evaluasi
- [ ] Hitung accuracy pada test set.
- [ ] Buat confusion matrix, identifikasi kelas yang sering tertukar.
- [ ] Kalau accuracy kurang memuaskan: cek kualitas data dulu (jumlah sample per kelas, variasi), baru pertimbangkan tuning arsitektur.

## Tahap 5: Siapkan Model untuk Serving
- [ ] Pastikan model tersimpan dalam format `.h5` atau SavedModel dari Tahap 3 (tidak perlu konversi ke format lain).
- [ ] Taruh file model di folder `web/model/` (atau folder serupa yang diakses langsung oleh app Streamlit).
- [ ] Tulis fungsi loader model (load sekali, cache pakai `st.cache_resource` biar tidak reload tiap frame).

## Tahap 6: Web Demo (Streamlit)
- [ ] Install `streamlit`, `streamlit-webrtc`, `mediapipe`, `opencv-python`.
- [ ] Setup halaman Streamlit dasar (`app.py`) dengan komponen `webrtc_streamer` dari `streamlit-webrtc` untuk stream kamera.
- [ ] Buat video frame callback: terima frame dari `streamlit-webrtc` → jalankan MediaPipe Hands (Python) → ekstrak landmark.
- [ ] Gambar overlay landmark ke frame (pakai OpenCV) sebelum frame dikembalikan ke UI.
- [ ] Normalisasi landmark (sama persis seperti saat training) → jalankan prediksi pakai model yang sudah di-load.
- [ ] Tampilkan hasil prediksi (kelas + confidence) di UI Streamlit, update tiap frame.
- [ ] **Tidak termasuk tahap ini**: fitur rangkai huruf jadi kata (history) — ditunda ke tahap terpisah setelah versi dasar di atas berjalan stabil.

## Tahap 7: Deploy
- [ ] Push project ke GitHub, sertakan `requirements.txt` (streamlit, streamlit-webrtc, mediapipe, opencv-python, tensorflow, dll).
- [ ] Deploy ke Streamlit Community Cloud, connect langsung ke repo GitHub.
- [ ] Test akses dari device lain untuk pastikan demo berjalan dan permission kamera bekerja (HTTPS otomatis ditangani Streamlit Community Cloud).

## Tahap 8: Dokumentasi
- [ ] Tulis README dengan ringkasan project, cara pakai demo, dan link live demo.
- [ ] Sertakan screenshot/GIF demo real-time untuk portofolio.

## Catatan Penting
- Normalisasi landmark di app Streamlit **harus identik** dengan normalisasi saat training — meskipun sama-sama Python, tetap pastikan fungsi preprocessing dipakai ulang (bukan ditulis dua kali) supaya konsisten.
- Latency: karena webrtc + inference berjalan di server, uji performa di kondisi hosting nyata (bukan cuma lokal) sebelum menganggap real-time-nya cukup responsif.
- Kalau nanti mau extend ke dynamic gesture (kata/kalimat), ini di luar scope tahap pertama — dicatat sebagai pengembangan lanjutan, bukan bagian dari rencana saat ini.
