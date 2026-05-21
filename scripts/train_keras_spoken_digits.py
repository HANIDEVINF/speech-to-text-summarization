from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "free-spoken-digit-dataset"
RECORDINGS = DATA_DIR / "recordings"
MODEL_DIR = ROOT / "public" / "model"
SAMPLES_DIR = ROOT / "public" / "samples"
REPORT_DIR = ROOT / "reports"
REPO_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
INPUT_LENGTH = 8000
FRAME_COUNT = 16
BINS_PER_FRAME = 32
FEATURE_SIZE = FRAME_COUNT * BINS_PER_FRAME
SEED = 42


def ensure_dataset() -> None:
    DATA_DIR.parent.mkdir(exist_ok=True)
    if not RECORDINGS.exists():
        print("Cloning Free Spoken Digit Dataset...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)], check=True)


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    target = resample(audio.astype(np.float32), INPUT_LENGTH)
    max_abs = np.max(np.abs(target)) or 1.0
    return (target / max_abs).astype(np.float32)


def extract_features(audio: np.ndarray) -> np.ndarray:
    frames = np.array_split(audio, FRAME_COUNT)
    features: list[float] = []
    for frame in frames:
        windowed = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(windowed))
        bands = np.array_split(spectrum[: len(spectrum) // 2], BINS_PER_FRAME)
        features.extend(float(np.log1p(band.mean())) for band in bands)
    array = np.array(features, dtype=np.float32)
    mean = array.mean()
    std = array.std() or 1.0
    return (array - mean) / std


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[Path]]:
    ensure_dataset()
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[Path] = []
    for path in sorted(RECORDINGS.glob("*.wav")):
        match = re.match(r"(\d)_", path.name)
        if not match:
            continue
        xs.append(extract_features(load_audio(path)))
        ys.append(int(match.group(1)))
        paths.append(path)
    return np.stack(xs), np.array(ys, dtype=np.int64), paths


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    x, y, paths = load_dataset()
    x_train, x_test, y_train, y_test, train_paths, test_paths = train_test_split(
        x, y, paths, test_size=0.2, random_state=SEED, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_SIZE,), name="spectral_features"),
            tf.keras.layers.Dense(192, activation="relu", name="dense_spectral_patterns"),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(96, activation="relu", name="dense_digit_patterns"),
            tf.keras.layers.Dense(48, activation="relu", name="dense_audio_features"),
            tf.keras.layers.Dense(10, activation="softmax", name="digit_probabilities"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=25,
        batch_size=32,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)],
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "dataset": "Free Spoken Digit Dataset",
        "dataset_url": REPO_URL,
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "input_length": INPUT_LENGTH,
        "feature_size": FEATURE_SIZE,
        "frame_count": FRAME_COUNT,
        "bins_per_frame": BINS_PER_FRAME,
        "classes": list(range(10)),
        "architecture": [
            "Log FFT band-energy features",
            "Dense(192, relu)",
            "Dropout(0.3)",
            "Dense(96, relu)",
            "Dense(48, relu)",
            "Dense(10, softmax)",
        ],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "epochs_ran": len(history.history["loss"]),
    }

    weights = model.get_weights()
    export = {
        "model_type": "keras_conv1d_spoken_digit_classifier",
        "input_length": INPUT_LENGTH,
        "feature_size": FEATURE_SIZE,
        "frame_count": FRAME_COUNT,
        "bins_per_frame": BINS_PER_FRAME,
        "classes": list(range(10)),
        "metrics": metrics,
        "weights": {
            "dense1_kernel": weights[0].tolist(),
            "dense1_bias": weights[1].tolist(),
            "dense2_kernel": weights[2].tolist(),
            "dense2_bias": weights[3].tolist(),
            "dense3_kernel": weights[4].tolist(),
            "dense3_bias": weights[5].tolist(),
            "out_kernel": weights[6].tolist(),
            "out_bias": weights[7].tolist(),
        },
    }

    model.save(MODEL_DIR / "keras_spoken_digit.keras")
    (MODEL_DIR / "spoken_digit_model.json").write_text(json.dumps(export), encoding="utf-8")
    (REPORT_DIR / "keras_spoken_digit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "spoken_digit_classification_report.txt").write_text(
        classification_report(y_test, predictions),
        encoding="utf-8",
    )

    copied: set[int] = set()
    for path in test_paths:
        digit = int(path.name[0])
        if digit not in copied:
            shutil.copy(path, SAMPLES_DIR / f"digit_{digit}.wav")
            copied.add(digit)
        if len(copied) == 10:
            break

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
