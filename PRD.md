# PRD — Deteksi Bahasa Isyarat BISINDO (Alfabet & Angka)

## 1. Latar Belakang
Project computer vision untuk mengenali bahasa isyarat BISINDO (Bahasa Isyarat Indonesia), sebagai portofolio pribadi — bukan bagian dari skripsi.

## 2. Tujuan
Membangun sistem yang dapat mendeteksi dan mengklasifikasikan gesture tangan statis (alfabet A-Z dan angka 0-9) dalam bahasa isyarat BISINDO secara real-time, lalu didemonstrasikan melalui web demo (Streamlit) yang bisa diakses dan dicoba langsung oleh siapa saja.

## 3. Scope

### In-scope (tahap pertama)
- Static gesture recognition: alfabet A-Z (26 kelas) dan angka 0-9 (10 kelas) — total 36 kelas.
- Bahasa isyarat: BISINDO.
- Web demo berbasis Streamlit dengan input kamera real-time (bukan upload gambar), menggunakan `streamlit-webrtc`.
- Model dan preprocessing (MediaPipe) berjalan di server Python (backend Streamlit), bukan di browser.

### Out-of-scope (untuk saat ini)
- Fitur rangkai huruf jadi kata (history deteksi) — ditunda, dikerjakan setelah versi dasar (deteksi per-huruf/angka) jalan stabil.
- Dynamic gesture (kata/kalimat bergerak) — bisa jadi pengembangan lanjutan.
- Bahasa isyarat selain BISINDO (misal SIBI, ASL).
- Deployment mobile/aplikasi native.

## 4. Target Pengguna
- Perekrut/reviewer portofolio yang mengevaluasi kemampuan teknis di bidang computer vision.
- Siapa pun yang ingin mencoba demo secara langsung dari browser.

## 5. Kebutuhan Fungsional
| ID | Kebutuhan |
|----|-----------|
| F1 | Sistem dapat mengakses kamera pengguna melalui browser |
| F2 | Sistem dapat mendeteksi tangan dan mengekstrak landmark dari video secara real-time |
| F3 | Sistem dapat mengklasifikasikan gesture menjadi salah satu dari 36 kelas (26 huruf + 10 angka) |
| F4 | Sistem menampilkan hasil prediksi beserta confidence score secara real-time di layar |
| F5 | Sistem menampilkan overlay titik-titik landmark tangan di atas video kamera |

## 6. Kebutuhan Non-Fungsional
| ID | Kebutuhan |
|----|-----------|
| NF1 | Inference berjalan real-time tanpa lag signifikan (mempertimbangkan latency server) |
| NF2 | Backend Python (Streamlit) menangani preprocessing (MediaPipe) dan inference model |
| NF3 | Dapat di-hosting gratis di Streamlit Community Cloud |

## 7. Sumber Data
- Dataset alfabet BISINDO dari sumber publik (Kaggle/Roboflow).
- Dataset angka: dicek ketersediaannya di sumber publik; jika tidak tersedia, dikumpulkan secara mandiri (10 kelas).

## 8. Metrik Keberhasilan
- Akurasi klasifikasi model pada test set.
- Confusion matrix untuk mengidentifikasi kelas yang sering tertukar (terutama gesture yang mirip secara visual).
- Web demo dapat diakses publik dan berfungsi secara real-time.

## 9. Risiko & Batasan
- Dataset publik BISINDO untuk angka mungkin terbatas/tidak tersedia.
- Akurasi model bergantung pada kualitas dan variasi dataset (pencahayaan, sudut, latar belakang).
- Gesture BISINDO yang mirip secara visual berpotensi menyebabkan kesalahan klasifikasi.
