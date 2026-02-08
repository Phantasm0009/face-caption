"""Download MediaPipe Face Landmarker model for face mesh + emotion. Run once."""
import os
import sys
import urllib.request

# Official model (check https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker for latest)
# This URL may need updating; see MediaPipe releases or model zoo.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
# Some distributions use face_landmarker_v2.task; try both filenames
OUTPUT_NAMES = ["face_landmarker.task", "face_landmarker_v2.task"]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in OUTPUT_NAMES:
        path = os.path.join(MODELS_DIR, name)
        if os.path.isfile(path):
            print("Model already at:", path)
            return
    print("Downloading MediaPipe Face Landmarker model...")
    path = os.path.join(MODELS_DIR, "face_landmarker.task")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Done. Model at:", path)
    except Exception as e:
        print("Download failed:", e)
        print("Download manually from:")
        print("  https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker")
        print("Save the .task file to:", path)


if __name__ == "__main__":
    main()
