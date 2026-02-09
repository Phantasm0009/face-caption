"""
MediaPipe Hand Landmarker (Tasks API) for pinch-to-scale.
Uses the same Tasks API as face_mesh; requires hand_landmarker.task model.
Run download_hand_landmarker_model.py once to fetch the model.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(SCRIPT_DIR, "models"),
    os.path.join(SCRIPT_DIR),
    os.path.join(os.getcwd(), "models"),
    os.getcwd(),
]
MODEL_NAMES = ["hand_landmarker.task"]

HAS_HAND_TASKS = False
HandLandmarker = None
HandLandmarkerOptions = None
BaseOptions = None
RunningMode = None
Image = None
ImageFormat = None

try:
    from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_lib
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode

    HandLandmarker = hand_landmarker_lib.HandLandmarker
    HandLandmarkerOptions = hand_landmarker_lib.HandLandmarkerOptions
    BaseOptions = base_options.BaseOptions
    RunningMode = vision_task_running_mode.VisionTaskRunningMode
    Image = image_lib.Image
    ImageFormat = image_lib.ImageFormat
    HAS_HAND_TASKS = True
except Exception:
    pass


def _find_model_path():
    for d in MODEL_DIRS:
        for name in MODEL_NAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


def create_hand_landmarker():
    """Create HandLandmarker if MediaPipe Tasks and model are available. Else return None."""
    if not HAS_HAND_TASKS:
        import sys
        print("Hand: MediaPipe Tasks API not available (install mediapipe).", file=sys.stderr)
        return None
    path = _find_model_path()
    if not path:
        import sys
        print("Hand: model not found. Run: python download_hand_landmarker_model.py", file=sys.stderr)
        print("  Looked in:", MODEL_DIRS, file=sys.stderr)
        return None
    try:
        # Use CPU only: packaged MediaPipe often has GPU disabled (ImageCloneCalculator fails).
        try:
            base_opts = BaseOptions(
                model_asset_path=path,
                delegate=BaseOptions.Delegate.CPU,
            )
        except Exception:
            base_opts = BaseOptions(model_asset_path=path)
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return HandLandmarker.create_from_options(options)
    except Exception as e:
        import sys
        print("Hand: failed to load model:", path, file=sys.stderr)
        print("  ", type(e).__name__, str(e), file=sys.stderr)
        return None


def detect_hands(landmarker, frame_rgb, timestamp_ms):
    """
    Run hand landmarker on RGB frame (numpy, HWC, uint8).
    Returns list of hand landmarks (each hand = list of 21 NormalizedLandmark with .x, .y)
    or None. Landmark index 4 = thumb tip, 8 = index tip.
    """
    if landmarker is None or frame_rgb is None:
        return None
    try:
        if not frame_rgb.flags.c_contiguous:
            frame_rgb = frame_rgb.copy()
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks
    except Exception as e:
        import sys
        print("Hand detect error:", type(e).__name__, e, file=sys.stderr)
        return None
