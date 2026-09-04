"""
Web Demo — Deteksi Bahasa Isyarat BISINDO
Streamlit app dengan akses kamera real-time via streamlit-webrtc.
MediaPipe HandLandmarker (Python, server-side) digunakan untuk ekstraksi landmark,
dan model MLP Keras untuk inference klasifikasi gesture.
"""

import json
import os
import queue
from pathlib import Path
from typing import NamedTuple

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow import keras

# ---------------------------------------------------------------------------
# Konstanta & Path
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_bisindo.keras"
CLASSES_PATH = BASE_DIR / "model" / "classes.json"

# File hand_landmarker.task ada di root project (satu level di atas web/)
HAND_LANDMARKER_PATH = BASE_DIR.parent / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Load Resources (di-cache agar tidak reload tiap frame)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_keras_model() -> keras.Model:
    """Load model MLP Keras dari disk, di-cache oleh Streamlit."""
    return keras.models.load_model(str(MODEL_PATH))


@st.cache_resource
def load_classes() -> list[str]:
    """Load daftar nama kelas dari classes.json."""
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def load_hand_landmarker() -> mp_vision.HandLandmarker:
    """Buat instance HandLandmarker MediaPipe (LIVE_STREAM mode untuk async callback)."""
    base_options = mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Preprocessing (identik dengan extract_landmarks.py)
# ---------------------------------------------------------------------------

def normalize_landmarks(hand_landmarks) -> np.ndarray:
    """
    Normalisasi 21 landmark tangan:
    1. Koordinat relatif terhadap wrist (landmark index 0).
    2. Scale normalization dibagi nilai absolut maksimum.

    Returns:
        np.ndarray: array 1D dengan 63 nilai float ternormalisasi.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])  # (21, 3)
    coords_relative = coords - coords[0]  # relatif ke wrist
    max_val = np.max(np.abs(coords_relative))
    if max_val > 0:
        coords_normalized = coords_relative / max_val
    else:
        coords_normalized = coords_relative
    return coords_normalized.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Hasil prediksi (dikirim dari VideoProcessor ke UI via queue)
# ---------------------------------------------------------------------------

class PredictionResult(NamedTuple):
    label: str
    confidence: float
    hand_detected: bool


# ---------------------------------------------------------------------------
# Video Processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Callback yang dipanggil oleh streamlit-webrtc untuk setiap frame video.
    Menjalankan MediaPipe + MLP inference di server, mengembalikan frame
    dengan overlay landmark dan label prediksi.
    """

    def __init__(self):
        self.model = load_keras_model()
        self.classes = load_classes()
        self.landmarker = load_hand_landmarker()
        self.result_queue: queue.Queue[PredictionResult] = queue.Queue(maxsize=1)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # MediaPipe membutuhkan RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = self.landmarker.detect(mp_image)

        hand_detected = bool(result.hand_landmarks)
        label = ""
        confidence = 0.0

        if hand_detected:
            landmarks = result.hand_landmarks[0]

            # Gambar landmark & koneksi ke frame (BGR)
            self._draw_landmarks(img, landmarks)

            # Normalisasi & prediksi
            features = normalize_landmarks(landmarks).reshape(1, -1)
            predictions = self.model.predict(features, verbose=0)[0]
            class_idx = int(np.argmax(predictions))
            label = self.classes[class_idx]
            confidence = float(predictions[class_idx])

            # Tampilkan label + confidence di frame
            self._draw_prediction(img, label, confidence)
        else:
            # Tangan tidak terdeteksi
            cv2.putText(
                img, "Tidak ada tangan terdeteksi",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        # Kirim hasil ke UI (non-blocking, drop frame lama kalau penuh)
        result_payload = PredictionResult(label=label, confidence=confidence, hand_detected=hand_detected)
        try:
            self.result_queue.put_nowait(result_payload)
        except queue.Full:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            self.result_queue.put_nowait(result_payload)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # -----------------------------------------------------------------------
    # Helper drawing methods
    # -----------------------------------------------------------------------

    def _draw_landmarks(self, img: np.ndarray, landmarks) -> None:
        """Gambar titik landmark dan koneksi tulang tangan ke frame."""
        h, w = img.shape[:2]

        # Koneksi tulang tangan (MediaPipe hand topology)
        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # Ibu jari
            (0, 5), (5, 6), (6, 7), (7, 8),           # Telunjuk
            (0, 9), (9, 10), (10, 11), (11, 12),      # Jari tengah
            (0, 13), (13, 14), (14, 15), (15, 16),    # Jari manis
            (0, 17), (17, 18), (18, 19), (19, 20),    # Kelingking
            (5, 9), (9, 13), (13, 17),                 # Telapak tangan
        ]

        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for start, end in CONNECTIONS:
            cv2.line(img, pts[start], pts[end], (0, 200, 100), 2)

        for x, y in pts:
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
            cv2.circle(img, (x, y), 5, (0, 150, 255), 1)

    def _draw_prediction(self, img: np.ndarray, label: str, confidence: float) -> None:
        """Tampilkan label prediksi dan confidence score di frame."""
        text = f"{label}  ({confidence * 100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        # Background hitam semi-transparan untuk keterbacaan
        cv2.rectangle(img, (8, 8), (tw + 16, th + 20), (0, 0, 0), -1)
        cv2.putText(img, text, (12, th + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 120), 3)


# ---------------------------------------------------------------------------
# UI Streamlit
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="BISINDO Sign Language Detection",
        page_icon="🤟",
        layout="wide",
    )

    st.title("🤟 Deteksi Bahasa Isyarat BISINDO")
    st.markdown(
        "Sistem deteksi gesture tangan **BISINDO** (Bahasa Isyarat Indonesia) secara real-time. "
        "Arahkan tangan ke kamera dan tunjukkan salah satu dari **35 gesture** (huruf A–Y & angka 0–9)."
    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Tentang Demo")
        st.markdown(
            "**Model**: MLP berbasis koordinat landmark tangan (63 fitur).\n\n"
            "**Preprocessing**: MediaPipe HandLandmarker (Python, server-side).\n\n"
            "**Framework**: Streamlit + streamlit-webrtc."
        )
        st.divider()
        confidence_threshold = st.slider(
            "Threshold Confidence (%)", min_value=10, max_value=95, value=60, step=5
        )
        st.divider()
        classes = load_classes()
        st.markdown(f"**Kelas yang didukung ({len(classes)} total):**")
        st.markdown(", ".join(classes))

    # Layout kolom
    col_video, col_result = st.columns([2, 1])

    with col_video:
        st.subheader("📷 Kamera Real-time")
        ctx = webrtc_streamer(
            key="bisindo-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_result:
        st.subheader("🔍 Hasil Prediksi")

        result_placeholder = st.empty()
        confidence_placeholder = st.empty()
        info_placeholder = st.empty()

        if ctx.state.playing and ctx.video_processor:
            while True:
                try:
                    pred: PredictionResult = ctx.video_processor.result_queue.get(timeout=3.0)
                except queue.Empty:
                    result_placeholder.info("⏳ Menunggu frame...")
                    continue

                if pred.hand_detected:
                    confidence_pct = pred.confidence * 100
                    if confidence_pct >= confidence_threshold:
                        result_placeholder.markdown(
                            f"<h1 style='text-align:center; font-size:96px;'>{pred.label}</h1>",
                            unsafe_allow_html=True,
                        )
                        confidence_placeholder.progress(
                            int(confidence_pct),
                            text=f"Confidence: {confidence_pct:.1f}%"
                        )
                        info_placeholder.success(f"Gesture terdeteksi: **{pred.label}**")
                    else:
                        result_placeholder.markdown(
                            f"<h1 style='text-align:center; font-size:96px; color:gray;'>{pred.label}</h1>",
                            unsafe_allow_html=True,
                        )
                        confidence_placeholder.progress(
                            int(confidence_pct),
                            text=f"Confidence: {confidence_pct:.1f}% (di bawah threshold)"
                        )
                        info_placeholder.warning("Confidence rendah — coba posisikan tangan lebih jelas.")
                else:
                    result_placeholder.markdown(
                        "<h1 style='text-align:center; font-size:96px;'>—</h1>",
                        unsafe_allow_html=True,
                    )
                    confidence_placeholder.empty()
                    info_placeholder.info("Arahkan tangan ke kamera.")
        else:
            result_placeholder.info("▶️ Klik **START** di panel kamera untuk memulai deteksi.")


if __name__ == "__main__":
    main()
