import cv2
import time
from collections import deque
from pathlib import Path

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- Config ----------
MODEL_PATH = Path("/Users/bmjonas04/Desktop/ACABI/face_landmarker.task")  # your file
MAX_FACES = 1
TOP_K = 6
SMOOTH_WIN = 10

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# ---------- Setup drawing ----------
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION

# ---------- Build FaceLandmarker ----------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    num_faces=MAX_FACES,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    running_mode=VisionRunningMode.VIDEO,
)
landmarker = FaceLandmarker.create_from_options(options)

# ---------- Helpers ----------
def draw_landmarks(image, normalized_landmarks):
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    landmark_list.landmark.extend(
        [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in normalized_landmarks]
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmark_list,
        connections=FACE_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
    )

def _get_categories(face_blendshapes_entry):
    """Return a list of Category objects regardless of MP version."""
    # Newer builds: entry is a Classifications with .categories
    if hasattr(face_blendshapes_entry, "categories"):
        return face_blendshapes_entry.categories
    # Some builds: it’s already a list[Category]
    return face_blendshapes_entry

def top_blendshapes(cats, k=TOP_K):
    cats_sorted = sorted(cats, key=lambda c: c.score, reverse=True)
    return cats_sorted[:k]

def get_score(cats, name, default=0.0):
    for c in cats:
        if c.category_name == name:
            return c.score
    return default

# Rolling buffers for smoothing
smile_left_buf = deque(maxlen=SMOOTH_WIN)
smile_right_buf = deque(maxlen=SMOOTH_WIN)
blink_left_buf = deque(maxlen=SMOOTH_WIN)
blink_right_buf = deque(maxlen=SMOOTH_WIN)

# ---------- Webcam loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_idx = 0
fps_t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    if result.face_landmarks:
        # draw landmarks for first face
        draw_landmarks(frame, result.face_landmarks[0])

        # --- Blendshapes (defensive to version differences)
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            cats = _get_categories(result.face_blendshapes[0])
            if cats:  # might be empty on the very first frames
                # Smile asymmetry (L-R) and Blink diff (L-R)
                sL = get_score(cats, "mouthSmileLeft")
                sR = get_score(cats, "mouthSmileRight")
                smile_left_buf.append(sL)
                smile_right_buf.append(sR)
                sL_s = sum(smile_left_buf) / len(smile_left_buf)
                sR_s = sum(smile_right_buf) / len(smile_right_buf)
                smile_symmetry = sL_s - sR_s

                bL = get_score(cats, "eyeBlinkLeft")
                bR = get_score(cats, "eyeBlinkRight")
                blink_left_buf.append(bL)
                blink_right_buf.append(bR)
                bL_s = sum(blink_left_buf) / len(blink_left_buf)
                bR_s = sum(blink_right_buf) / len(blink_right_buf)
                blink_diff = bL_s - bR_s

                # Show top K blendshapes
                tops = top_blendshapes(cats, k=TOP_K)
                y = 22
                for c in tops:
                    cv2.putText(
                        frame,
                        f"{c.category_name}: {c.score:.2f}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 18

                cv2.putText(
                    frame,
                    f"Smile asym (L-R): {smile_symmetry:+.2f}",
                    (10, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Blink diff (L-R): {blink_diff:+.2f}",
                    (10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    # FPS overlay (update every 30 frames)
    if frame_idx % 30 == 0:
        t1 = time.time()
        fps = 30.0 / (t1 - fps_t0)
        fps_t0 = t1
        cv2.putText(
            frame,
            f"FPS ~ {fps:.1f}",
            (frame.shape[1] - 140, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("FaceLandmarker + Blendshapes", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
