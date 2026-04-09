import os
import cv2
import pandas as pd
from collections import defaultdict
import mediapipe as mp
from mediapipe.tasks import python as mpython
from mediapipe.tasks.python import vision
from typing import List,Dict
import src.lib.file_utils as fu
from src.lib.globals import *
import src.lib.metadata as metadata


def get_face_image_landmarker(MODEL_PATH: str) -> any:
    assert os.path.exists(MODEL_PATH)
    _base_options = mpython.BaseOptions
    _face_landmarker = vision.FaceLandmarker
    face_landmarker_options = vision.FaceLandmarkerOptions
    _vision_running_mode = vision.RUnningMode

    MAX_FACES = 1

    options = face_landmarker_options(
        base_options = _base_options(model_asset_path = MODEL_PATH),
        num_faces = MAX_FACES,
        output_face_blendshapes = True,
        output_facial_transformation_matrixes = False,
        running_mode = _vision_running_mode.IMAGE
    )

    landmarker = _face_landmarker.create_from_options(options)
    return landmarker


def detect_blendshapes(landmarker: any, image_path: str) -> dict:
    bgr = cv2.imread(image_path)
    assert bgr is not None

    RGB = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB)
    result = landmarker.detect(mediapipe_image)

    assert result.face_blendshapes
    assert len(result.face_blendshapes) != 0
    
    blendshapes = {
        c.category_name: float(c.score)
        for c in result.face_blendshapes[0].categories
    }
    return blendshapes


def get_trial_groupings() -> Dict[List[str]]:
    all_found_files = fu.search_files("protocol1", directory=RAW_DIR)

    groupings = defaultdict(list)
    for file in all_found_files:
        subject = metadata.get_subject(file)
        groupings[subject].append(file)

    return dict(groupings)


def get_outpath(subject: str) -> str:
    filename = f"protocol1_{subject}.csv"
    filepath = os.path.join(PROCESSED_DIR, filename)
    return filepath


def main():
    MODEL_PATH = ""
    landmarker = get_face_image_landmarker(MODEL_PATH)
    for subject, trial_files in get_trial_groupings().items():
        outpath = get_outpath(subject)

        rows = []
        indices = []

        for image_file in trial_files:
            blendshapes = detect_blendshapes(landmarker, image_file)
            trial = metadata.get_trial(image_file)

            rows.append(blendshapes)
            indices.append(trial)

        df = pd.DataFrame(rows, index=indices)
        df.to_csv(outpath)

    return


if __name__ == "__main__":
    main()
