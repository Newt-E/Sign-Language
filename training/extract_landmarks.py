"""
Ekstraksi Landmark Tangan BISINDO (Multi-process Parallel)
Script ini mengekstrak 21 titik koordinat landmark (x, y, z) dari dataset gambar
menggunakan MediaPipe HandLandmarker, melakukan normalisasi (relatif terhadap wrist & scale),
dan menyimpannya sebagai file tabular (CSV) untuk training model MLP.
"""

import os
import urllib.request
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Global detector instance for worker process
_detector = None


def ensure_model_file(model_path: str = MODEL_PATH):
    """Pastikan file model hand_landmarker.task tersedia, download jika belum ada."""
    if not os.path.exists(model_path):
        print(f"Downloading model asset to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded successfully.")


def _init_worker(model_path: str):
    """Inisialisasi detector MediaPipe untuk setiap worker process."""
    global _detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    _detector = vision.HandLandmarker.create_from_options(options)


def extract_and_normalize_landmarks(hand_landmarks) -> list[float]:
    """
    Ekstrak 21 titik landmark tangan dan lakukan normalisasi:
    1. Koordinat relatif terhadap wrist (landmark index 0).
    2. Scale normalization dibagi nilai absolut maksimum (invariant terhadap jarak/posisi).
    
    Returns:
        list[float]: 63 nilai float ternormalisasi (21 titik * 3 koordinat).
    """
    coords = []
    for lm in hand_landmarks:
        coords.append([lm.x, lm.y, lm.z])
    coords = np.array(coords)  # Shape: (21, 3)

    # 1. Translasi relatif ke wrist (titik 0)
    wrist = coords[0]
    coords_relative = coords - wrist

    # 2. Scale normalization
    max_val = np.max(np.abs(coords_relative))
    if max_val > 0:
        coords_normalized = coords_relative / max_val
    else:
        coords_normalized = coords_relative

    return coords_normalized.flatten().tolist()


def _process_single_image(item: tuple[str, str]) -> tuple[bool, str, list[float] | None]:
    """Fungsi helper yang dipanggil di worker process untuk satu gambar."""
    img_path, cls = item
    global _detector
    try:
        mp_image = mp.Image.create_from_file(img_path)
        results = _detector.detect(mp_image)
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            features = extract_and_normalize_landmarks(results.hand_landmarks[0])
            return True, cls, features
    except Exception:
        pass
    return False, cls, None


def run_extraction(data_dir: str, output_csv: str, model_path: str = MODEL_PATH, workers: int = 8):
    """
    Menjalankan proses ekstraksi landmark secara parallel untuk seluruh gambar di direktori dataset.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {data_dir}")

    ensure_model_file(model_path)

    # Kumpulkan seluruh gambar
    all_items = []
    class_counts_raw = {}
    for root, _, files in os.walk(data_dir):
        valid_files = [
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if valid_files:
            class_name = os.path.basename(root)
            class_counts_raw[class_name] = len(valid_files)
            for fpath in valid_files:
                all_items.append((fpath, class_name))

    total_images = len(all_items)
    print(f"Ditemukan {len(class_counts_raw)} kelas dengan total {total_images} gambar di '{data_dir}'.")
    print(f"Memulai ekstraksi parallel menggunakan {workers} worker processes...\n")

    data_rows = []
    stats_success = {cls: 0 for cls in class_counts_raw}
    stats_failed = {cls: 0 for cls in class_counts_raw}

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(model_path,)) as executor:
        for success, cls, feat in tqdm(
            executor.map(_process_single_image, all_items, chunksize=30),
            total=total_images,
            desc="Ekstraksi Landmark"
        ):
            if success and feat is not None:
                data_rows.append(feat + [cls])
                stats_success[cls] += 1
            else:
                stats_failed[cls] += 1

    # Buat DataFrame
    feature_cols = []
    for i in range(21):
        feature_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    columns = feature_cols + ["label"]

    df = pd.DataFrame(data_rows, columns=columns)
    df["label"] = df["label"].astype(str)

    # Simpan ke file output
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Ringkasan Ekstraksi
    total_success = sum(stats_success.values())
    total_failed = sum(stats_failed.values())
    detection_rate = (total_success / total_images * 100) if total_images > 0 else 0.0

    print("\n" + "=" * 60)
    print("RINGKASAN EKSTRAKSI LANDMARK (TAHAP 2)")
    print("=" * 60)
    print(f"Total Gambar Input       : {total_images}")
    print(f"Total Berhasil Diekstrak   : {total_success} ({detection_rate:.2f}%)")
    print(f"Total Gagal / Dilewati     : {total_failed}")
    print(f"Hasil Disimpan Ke          : {output_csv}")
    print(f"Dimensi Dataset Tabular    : {df.shape} (baris, kolom)")
    print("=" * 60)

    print("\nDetail Per Kelas:")
    print(f"{'Kelas':<8} {'Total':<8} {'Berhasil':<10} {'Gagal':<8} {'Akurasi Deteksi':<15}")
    print("-" * 55)
    for cls in sorted(class_counts_raw.keys()):
        tot = class_counts_raw[cls]
        succ = stats_success[cls]
        fail = stats_failed[cls]
        rate = (succ / tot * 100) if tot > 0 else 0.0
        print(f"{cls:<8} {tot:<8} {succ:<10} {fail:<8} {rate:>6.1f}%")
    print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi Landmark Tangan BISINDO")
    parser.add_argument("--data_dir", type=str, default="Data", help="Path direktori dataset mentah")
    parser.add_argument("--output", type=str, default="data/landmarks_bisindo.csv", help="Path output file CSV")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path ke file hand_landmarker.task")
    parser.add_argument("--workers", type=int, default=8, help="Jumlah parallel worker processes")
    args = parser.parse_args()

    run_extraction(args.data_dir, args.output, args.model_path, args.workers)
