"""Download MediaPipe Hand Landmarker model for pinch-to-scale. Run once."""
import os
import sys
import urllib.request

# Official model (Hand Landmarker Tasks API)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_NAME = "hand_landmarker.task"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, OUTPUT_NAME)
    if os.path.isfile(path):
        print("Model already at:", path)
        return
    print("Downloading MediaPipe Hand Landmarker model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Done. Model at:", path)
    except Exception as e:
        print("Download failed:", e)
        print("Download manually from:")
        print("  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker")
        print("Save the .task file to:", path)


if __name__ == "__main__":
    main()
