"""
Evaluasi Model Deteksi Gesture BISINDO
Script ini memuat model yang sudah dilatih dan data test set,
menghitung akurasi, classification report per kelas, dan menyimpan grafik Confusion Matrix.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras


def stratified_split_dataset(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42
):
    rng = np.random.RandomState(random_state)
    train_idx, val_idx, test_idx = [], [], []

    classes = np.unique(y)
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n = len(cls_indices)

        if n >= 6:
            n_test = max(1, int(np.round(n * test_size)))
            n_val = max(1, int(np.round(n * val_size)))
            test_idx.extend(cls_indices[:n_test])
            val_idx.extend(cls_indices[n_test : n_test + n_val])
            train_idx.extend(cls_indices[n_test + n_val :])
        elif n >= 3:
            test_idx.extend(cls_indices[:1])
            val_idx.extend(cls_indices[1:2])
            train_idx.extend(cls_indices[2:])
        else:
            train_idx.extend(cls_indices)

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )


def evaluate(
    model_path: str = "training/model_bisindo.keras",
    classes_path: str = "training/classes.json",
    data_csv_path: str = "data/landmarks_bisindo.csv",
    output_cm_path: str = "training/confusion_matrix.png"
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di '{model_path}'. Jalankan training terlebih dahulu.")

    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"File classes.json tidak ditemukan di '{classes_path}'.")

    print("=" * 60)
    print("TAHAP 4: EVALUASI MODEL MLP BISINDO")
    print("=" * 60)

    # 1. Load Classes
    with open(classes_path, "r") as f:
        classes = json.load(f)
    print(f"Total kelas ({len(classes)}): {classes}")

    # 2. Load Test Set
    if os.path.exists("training/data_split/X_test.npy") and os.path.exists("training/data_split/y_test.npy"):
        print("Memuat test set dari cache 'training/data_split/'...")
        X_test = np.load("training/data_split/X_test.npy")
        y_test = np.load("training/data_split/y_test.npy")
    else:
        print(f"Membaca dataset dari {data_csv_path} untuk test split...")
        df = pd.read_csv(data_csv_path, dtype={"label": str})
        df["label"] = df["label"].astype(str)
        X = df.drop(columns=["label"]).values.astype(np.float32)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(classes)
        y_encoded = label_encoder.transform(df["label"].values)

        _, _, X_test, _, _, y_test = stratified_split_dataset(
            X, y_encoded, test_size=0.15, val_size=0.15, random_state=42
        )

    print(f"Test set size: {len(X_test)} sampel.")

    # 3. Load Model
    print(f"Memuat model dari {model_path}...")
    model = keras.models.load_model(model_path)

    # 4. Predict
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Metrics
    acc = accuracy_score(y_test, y_pred)
    loss, keras_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n" + "=" * 60)
    print("RINGKASAN METRIK EVALUASI")
    print("=" * 60)
    print(f"Test Loss      : {loss:.4f}")
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print("=" * 60)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0)
    print(report)

    # Simpan classification report ke text file
    os.makedirs("training", exist_ok=True)
    with open("training/classification_report.txt", "w") as f:
        f.write(f"Test Loss: {loss:.4f}\nTest Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # 6. Confusion Matrix & Plot
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title(f"Confusion Matrix BISINDO (Test Accuracy: {acc * 100:.2f}%)", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_cm_path) or ".", exist_ok=True)
    plt.savefig(output_cm_path, dpi=150)
    plt.close()
    print(f"\nGambar Confusion Matrix disimpan ke: {output_cm_path}")

    # 7. Identifikasi Pasangan yang Sering Tertukar (Misclassifications)
    misclassified = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                misclassified.append((classes[i], classes[j], int(cm[i][j])))

    misclassified.sort(key=lambda x: x[2], reverse=True)
    if misclassified:
        print("\nTop 5 Gesture yang Paling Sering Tertukar:")
        for true_cls, pred_cls, count in misclassified[:5]:
            print(f"  - True '{true_cls}' diprediksi sebagai '{pred_cls}': {count} kali")
    else:
        print("\nTidak ada kesalahan prediksi pada test set!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Model BISINDO")
    parser.add_argument("--model", type=str, default="training/model_bisindo.keras", help="Path file model")
    parser.add_argument("--classes", type=str, default="training/classes.json", help="Path classes JSON")
    parser.add_argument("--output_cm", type=str, default="training/confusion_matrix.png", help="Path output confusion matrix")
    args = parser.parse_args()

    evaluate(model_path=args.model, classes_path=args.classes, output_cm_path=args.output_cm)
