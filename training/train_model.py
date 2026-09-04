"""
Training Model MLP Deteksi Gesture BISINDO
Script ini memuat dataset landmark (63 fitur), melakukan stratified train/val/test split,
membangun dan melatih model MLP berbasis Keras, serta menyimpan model terbaik.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def stratified_split_dataset(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42
):
    """
    Melakukan pembagian stratified train/val/test split yang robust bahkan jika ada kelas dengan jumlah sampel sedikit.
    """
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


def build_mlp_model(input_dim: int = 63, num_classes: int = 35) -> keras.Model:
    """
    Membangun arsitektur MLP sesuai architecture-sign-language-bisindo.md:
    Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="landmark_input"),
        layers.Dense(128, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(64, activation="relu", name="dense_2"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(num_classes, activation="softmax", name="classification_output"),
    ], name="BISINDO_MLP")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train(
    data_path: str = "data/landmarks_bisindo.csv",
    output_model_path: str = "training/model_bisindo.keras",
    classes_json_path: str = "training/classes.json",
    epochs: int = 100,
    batch_size: int = 64
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di '{data_path}'. Jalankan tahap ekstraksi landmark terlebih dahulu.")

    print("=" * 60)
    print("TAHAP 3: TRAINING MODEL MLP BISINDO")
    print("=" * 60)

    # 1. Load data
    print(f"Memuat dataset dari {data_path}...")
    df = pd.read_csv(data_path, dtype={"label": str})
    df["label"] = df["label"].astype(str)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    # 2. Encode label
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    classes = [str(c) for c in label_encoder.classes_]
    num_classes = len(classes)
    print(f"Dataset berhasil dimuat: {len(X)} sampel, {num_classes} kelas terdeteksi.")

    # Simpan metadata kelas untuk training dan web demo
    os.makedirs(os.path.dirname(classes_json_path) or ".", exist_ok=True)
    with open(classes_json_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Daftar kelas disimpan ke {classes_json_path}")

    # Simpan juga ke web/model/ jika folder web ada
    web_classes_path = "web/model/classes.json"
    os.makedirs(os.path.dirname(web_classes_path) or ".", exist_ok=True)
    with open(web_classes_path, "w") as f:
        json.dump(classes, f, indent=2)

    # 3. Stratified Split (70% Train, 15% Validation, 15% Test)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_dataset(
        X, y_encoded, test_size=0.15, val_size=0.15, random_state=42
    )

    print(f"Split data:")
    print(f"  - Train      : {len(X_train)} sampel ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Validation : {len(X_val)} sampel ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  - Test       : {len(X_test)} sampel ({len(X_test)/len(X)*100:.1f}%)")

    # Simpan test set untuk script evaluate.py
    os.makedirs("training/data_split", exist_ok=True)
    np.save("training/data_split/X_test.npy", X_test)
    np.save("training/data_split/y_test.npy", y_test)

    # 4. Build Model
    model = build_mlp_model(input_dim=63, num_classes=num_classes)
    model.summary()

    # 5. Callbacks
    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 6. Fit Model
    print(f"\nMemulai training selama maksimal {epochs} epoch (batch size: {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=1
    )

    # Simpan juga format .h5 untuk kompatibilitas converter
    h5_path = os.path.splitext(output_model_path)[0] + ".h5"
    try:
        model.save(h5_path)
        print(f"Model .h5 juga disimpan ke {h5_path}")
    except Exception as e:
        print(f"Notice saat simpan .h5: {e}")

    # 7. Evaluasi pada Test Set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "=" * 60)
    print("HASIL TRAINING & EVALUASI SEMENTARA:")
    print("=" * 60)
    print(f"Best Validation Loss : {min(history.history['val_loss']):.4f}")
    print(f"Best Validation Acc  : {max(history.history['val_accuracy']) * 100:.2f}%")
    print(f"Test Set Loss        : {test_loss:.4f}")
    print(f"Test Set Accuracy    : {test_acc * 100:.2f}%")
    print(f"Model disimpan di    : {output_model_path}")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training MLP Deteksi Gesture BISINDO")
    parser.add_argument("--data", type=str, default="data/landmarks_bisindo.csv", help="Path CSV landmark")
    parser.add_argument("--output", type=str, default="training/model_bisindo.keras", help="Path model output")
    parser.add_argument("--classes", type=str, default="training/classes.json", help="Path file classes json")
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_model_path=args.output,
        classes_json_path=args.classes,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
