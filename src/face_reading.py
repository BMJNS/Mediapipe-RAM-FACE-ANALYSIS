import cv2
from pathlib import Path

import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------- Paths / Config -----------------
BASE_DIR = "/Users/bmjonas04/Desktop/ACABI/hand_test"

# Photo 1 = NEUTRAL, Photo 2 = EMOTION / EXPRESSION
NEUTRAL_IMAGE_PATH = f"{BASE_DIR}/photo neutral3.jpg"
EMOTION_IMAGE_PATH = f"{BASE_DIR}/photo stroke.jpg"

MODEL_PATH = Path("/Users/bmjonas04/Desktop/ACABI/face_landmarker.task")
MAX_FACES = 1

# ----------------- Checks -----------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not Path(NEUTRAL_IMAGE_PATH).exists():
    raise FileNotFoundError(f"Neutral image not found: {NEUTRAL_IMAGE_PATH}")
if not Path(EMOTION_IMAGE_PATH).exists():
    raise FileNotFoundError(f"Emotion image not found: {EMOTION_IMAGE_PATH}")

# ----------------- MediaPipe FaceLandmarker (IMAGE mode) -----------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    num_faces=MAX_FACES,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    running_mode=VisionRunningMode.IMAGE,
)
landmarker = FaceLandmarker.create_from_options(options)

# ----------------- Helpers -----------------
def _get_categories(face_blendshapes_entry):
    """Return a list of Category objects regardless of MediaPipe version."""
    if hasattr(face_blendshapes_entry, "categories"):
        return face_blendshapes_entry.categories
    return face_blendshapes_entry


def detect_blendshapes(image_path: str) -> dict:
    """Return dict[name]=score for first face in image."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_blendshapes or len(result.face_blendshapes) == 0:
        raise RuntimeError(f"No face blendshapes detected in: {image_path}")

    cats = _get_categories(result.face_blendshapes[0])
    if not cats:
        raise RuntimeError(f"Face blendshapes list was empty in: {image_path}")

    return {c.category_name: float(c.score) for c in cats}


def print_scores(title: str, scores_dict: dict):
    """Print ALL scores sorted high→low."""
    print(f"\n--- {title} ---")
    items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    for name, score in items:
        print(f"{name:30s} {score:.4f}")


def print_signed_delta(title: str, delta_dict: dict):
    """Print ALL signed deltas sorted by |Δ|."""
    print(f"\n--- {title} ---")
    items = sorted(delta_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, d in items:
        print(f"{name:30s} {d:+.4f}")


def compute_delta(neutral: dict, emotion: dict) -> dict:
    """delta = emotion - neutral for union of keys"""
    keys = set(neutral.keys()) | set(emotion.keys())
    return {k: float(emotion.get(k, 0.0)) - float(neutral.get(k, 0.0)) for k in keys}


def plot_raw_scores(scores_dict: dict, title: str):
    """Graph ALL raw scores (0–1)."""
    items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()


def plot_delta_signed(delta_dict: dict, title: str):
    """Graph ALL signed deltas (-1 to 1)."""
    items = sorted(delta_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("Δ (emotion − neutral)")
    plt.title(title)
    plt.ylim(-1.0, 1.0)
    plt.axhline(0.0, linewidth=1)  # zero line for easier reading
    plt.tight_layout()


# ----------------- Main -----------------
def main():
    # 1) Extract blendshapes for BOTH photos
    neutral_scores = detect_blendshapes(NEUTRAL_IMAGE_PATH)
    emotion_scores = detect_blendshapes(EMOTION_IMAGE_PATH)

    # 2) Print ALL raw scores to terminal
    print_scores(f"PHOTO 1 (NEUTRAL) ALL BLENDSHAPE SCORES: {NEUTRAL_IMAGE_PATH}", neutral_scores)
    print_scores(f"PHOTO 2 (EMOTION) ALL BLENDSHAPE SCORES: {EMOTION_IMAGE_PATH}", emotion_scores)

    # 3) Delta = emotion - neutral (print ALL signed deltas)
    delta = compute_delta(neutral_scores, emotion_scores)
    print_signed_delta("DELTA (PHOTO 2 - PHOTO 1) ALL BLENDSHAPES [SIGNED, sorted by |Δ|]", delta)

    # ----------------- PLOTTING (ONLY 3 GRAPHS TOTAL) -----------------
    # Graph 1: Neutral raw scores (ALL)
    plot_raw_scores(neutral_scores, "Neutral Facial Blendshape Scores (Baseline) (ALL, 0–1)")

    # Graph 2: Emotion raw scores (ALL)
    plot_raw_scores(emotion_scores, "Facial Blendshape Scores During Simulated Facial Asymmetry (ALL, 0–1)")

    # Graph 3: Signed delta (ALL, -1 to 1)
    plot_delta_signed(delta, "Blendshape Deltas for Simulated Facial Asymmetry Relative to Neutral (ALL, −1 to 1)")

    print("\nClose the 3 plot windows to end.")
    plt.show()

# final comment

if __name__ == "__main__":
    main()
