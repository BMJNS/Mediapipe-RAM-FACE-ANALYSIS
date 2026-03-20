import cv2
from pathlib import Path

import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("MediaPipe version:", mp.__version__)

# ----------------- Paths / Config -----------------
BASE_DIR = "/Users/bmjonas04/Desktop/ACABI/hand_test"

# Photo 1 = NEUTRAL, Photo 2 = EMOTION / EXPRESSION
NEUTRAL_IMAGE_PATH = f"{BASE_DIR}/photo neutral4.jpg"
EMOTION_IMAGE_PATH = f"{BASE_DIR}/photo cheek.jpg"

MODEL_PATH = Path("/Users/bmjonas04/Desktop/ACABI/face_landmarker.task")
MAX_FACES = 1

# Asymmetry decision threshold (hard line)
ASYM_THRESHOLD = 0.25

# Minimum activation gate: if the expression isn't "present", don't classify symmetry/asymmetry
MIN_ACTIVATION = 0.05

# Remove eye-gaze-ish L/R bases from symmetry/asymmetry logic (keep raw scores though)
EXCLUDED_LR_BASES = {
    "eyeLookIn",
    "eyeLookOut",
    "eyeLookUp",
    "eyeLookDown",
}

EPS = 1e-6

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
        print(f"{name:45s} {score:.4f}")


def print_signed_delta(title: str, delta_dict: dict):
    """Print ALL signed deltas sorted by |Δ|."""
    print(f"\n--- {title} ---")
    items = sorted(delta_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, d in items:
        print(f"{name:45s} {d:+.4f}")


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
    plt.axhline(0.0, linewidth=1)
    plt.tight_layout()


# ---------- Left/Right pairing + derived metrics ----------
def _lr_pair_key(name: str):
    """
    Detect left/right pairs based on MediaPipe naming:
      'mouthSmileLeft' <-> 'mouthSmileRight'
    Returns (base, side) where side in {'Left','Right'} or (None,None)
    """
    if name.endswith("Left"):
        return name[:-4], "Left"
    if name.endswith("Right"):
        return name[:-5], "Right"
    return None, None


def add_symmetry_asymmetry_strength(
    scores: dict,
    asym_threshold: float = ASYM_THRESHOLD,
    min_activation: float = MIN_ACTIVATION,
) -> dict:
    """
    For each L/R pair, add:
      - <base>_strength
      - <base>_asymmetry
      - <base>_symmetry

    Skips bases in EXCLUDED_LR_BASES (eye look features), but keeps raw L/R scores.

    Uses a magnitude gate:
      If max(L,R) < min_activation:
        treat as "absent" -> don't classify symmetry/asymmetry (kept at 0).

    Strength uses hard threshold (when present):
      if asymmetry <= T: avg(L,R)
      else: max(L,R)

    Leaves non-paired keys unchanged.
    """
    out = dict(scores)  # keep originals as-is

    # group into pairs
    pairs = {}
    for k, v in scores.items():
        base, side = _lr_pair_key(k)
        if base is None:
            continue
        pairs.setdefault(base, {})[side] = float(v)

    # compute derived metrics for complete pairs
    for base, d in pairs.items():
        if base in EXCLUDED_LR_BASES:
            continue
        if "Left" not in d or "Right" not in d:
            continue

        L = d["Left"]
        R = d["Right"]
        M = max(L, R)

        if M < min_activation:
            # absent -> no laterality classification
            S = (L + R) / 2.0
            A = 0.0
            Sym = 0.0
        else:
            A = abs(L - R) / (L + R + EPS)   # 0..1
            Sym = 1.0 - A                    # 0..1

            if A <= asym_threshold:
                S = (L + R) / 2.0
            else:
                S = max(L, R)

        out[f"{base}_strength"] = float(S)
        out[f"{base}_asymmetry"] = float(A)
        out[f"{base}_symmetry"] = float(Sym)

    return out


# ---------- NEW: delta filters for plotting ----------
def filter_delta_strength_only(delta_dict: dict) -> dict:
    """
    Final 'strength' delta plot should include:
      1) any derived <base>_strength
      2) ALSO raw magnitudes for non-L/R features (i.e., keys that are not derived metrics and
         not Left/Right sided and not in excluded eyeLook bases)
    """
    out = {}
    for k, v in delta_dict.items():
        # include all derived strengths
        if k.endswith("_strength"):
            out[k] = v
            continue

        # skip derived laterality metrics
        if k.endswith(("_symmetry", "_asymmetry")):
            continue

        # skip raw Left/Right components (we represent those via _strength)
        if k.endswith("Left") or k.endswith("Right"):
            continue

        # skip MediaPipe neutral
        if k == "_neutral":
            continue

        # skip eye-look families entirely from this "strength" plot
        if any(k.startswith(base) for base in EXCLUDED_LR_BASES):
            continue

        # everything else is a non-L/R raw magnitude -> include it
        out[k] = v

    return out


def filter_delta_laterality_only(delta_dict: dict) -> dict:
    """
    Keep only symmetry + asymmetry deltas:
      - <base>_symmetry
      - <base>_asymmetry
    """
    keep_suffixes = ("_symmetry", "_asymmetry")
    out = {k: v for k, v in delta_dict.items() if k.endswith(keep_suffixes)}

    # Also remove any eye-look-derived metrics if they ever slip in
    for base in EXCLUDED_LR_BASES:
        for key in list(out.keys()):
            if key.startswith(base + "_"):
                out.pop(key, None)

    return out


# ----------------- Main -----------------
def main():
    print(
        f"\nAsymmetry threshold A > {ASYM_THRESHOLD:.2f} => unilateral (strength=max), else bilateral (strength=avg)."
        f"\nMinimum activation gate: max(L,R) < {MIN_ACTIVATION:.2f} => treat as absent (no symmetry/asymmetry classification)."
        f"\nExcluded L/R bases from symmetry/asymmetry logic: {sorted(EXCLUDED_LR_BASES)}"
    )

    # 1) Extract blendshapes for BOTH photos
    neutral_scores_raw = detect_blendshapes(NEUTRAL_IMAGE_PATH)
    emotion_scores_raw = detect_blendshapes(EMOTION_IMAGE_PATH)

    # 2) Add derived metrics for L/R pairs (strength/asym/sym) while keeping originals
    neutral_scores = add_symmetry_asymmetry_strength(neutral_scores_raw, ASYM_THRESHOLD, MIN_ACTIVATION)
    emotion_scores = add_symmetry_asymmetry_strength(emotion_scores_raw, ASYM_THRESHOLD, MIN_ACTIVATION)

    # 3) Print ALL scores to terminal (includes derived metrics, EXCLUDES is_unilateral)
    print_scores(f"PHOTO 1 (NEUTRAL) ALL SCORES (BLENDSHAPES + DERIVED): {NEUTRAL_IMAGE_PATH}", neutral_scores)
    print_scores(f"PHOTO 2 (EMOTION) ALL SCORES (BLENDSHAPES + DERIVED): {EMOTION_IMAGE_PATH}", emotion_scores)

    # 4) Delta = emotion - neutral (print ALL signed deltas)
    delta_all = compute_delta(neutral_scores, emotion_scores)
    print_signed_delta("DELTA (PHOTO 2 - PHOTO 1) ALL SCORES [SIGNED, sorted by |Δ|]", delta_all)

    # 5) Split delta into two separate plots:
    #    - Strength plot: derived strengths + non-L/R raw magnitudes
    #    - Laterality plot: symmetry + asymmetry only
    delta_strength = filter_delta_strength_only(delta_all)
    delta_laterality = filter_delta_laterality_only(delta_all)

    # ----------------- PLOTTING (NOW 4 GRAPHS TOTAL) -----------------
    # Graph 1: Neutral (raw + derived strength/asym/sym; includes non-L/R raw magnitudes)
    plot_raw_scores(neutral_scores, "GRAPH 1 — NEUTRAL SCORES (RAW + strength/asym/sym)")

    # Graph 2: Emotion (raw + derived strength/asym/sym; includes non-L/R raw magnitudes)
    plot_raw_scores(emotion_scores, "GRAPH 2 — EMOTION SCORES (RAW + strength/asym/sym)")

    # Graph 3: Delta (strength + non-L/R magnitudes)
    plot_delta_signed(delta_strength, "GRAPH 3 — DELTA (EMOTION − NEUTRAL) — STRENGTH + NON-L/R MAGNITUDES")

    # Graph 4: Delta (symmetry + asymmetry only)
    plot_delta_signed(delta_laterality, "GRAPH 4 — DELTA (EMOTION − NEUTRAL) — SYMMETRY + ASYMMETRY ONLY")

    print("\nClose the plot windows to end.")
    plt.show()


if __name__ == "__main__":
    main()
