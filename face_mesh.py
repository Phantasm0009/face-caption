"""
MediaPipe Face Landmarker (face mesh) for robust tracking and emotion from blendshapes.
Fallback: use OpenCV Haar in main app if model is missing or MediaPipe unavailable.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(SCRIPT_DIR, "models"),
    os.path.join(SCRIPT_DIR),
    os.path.join(os.getcwd(), "models"),
    os.getcwd(),
]
MODEL_NAMES = ["face_landmarker_v2.task", "face_landmarker.task"]

# Blendshape names used for emotion (MediaPipe Face Landmarker output)
BLENDSHAPE_SMILE_LEFT = "mouthSmileLeft"
BLENDSHAPE_SMILE_RIGHT = "mouthSmileRight"
BLENDSHAPE_JAW_OPEN = "jawOpen"
BLENDSHAPE_BROW_DOWN_LEFT = "browDownLeft"
BLENDSHAPE_BROW_DOWN_RIGHT = "browDownRight"
BLENDSHAPE_BROW_INNER_UP = "browInnerUp"
BLENDSHAPE_MOUTH_FROWN_LEFT = "mouthFrownLeft"
BLENDSHAPE_MOUTH_FROWN_RIGHT = "mouthFrownRight"

HAS_MEDIAPIPE = False
FaceLandmarker = None
FaceLandmarkerOptions = None
BaseOptions = None
RunningMode = None
Image = None
ImageFormat = None

try:
    from mediapipe.tasks.python.vision import face_landmarker
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode

    FaceLandmarker = face_landmarker.FaceLandmarker
    FaceLandmarkerOptions = face_landmarker.FaceLandmarkerOptions
    BaseOptions = base_options.BaseOptions
    RunningMode = vision_task_running_mode.VisionTaskRunningMode
    Image = image_lib.Image
    ImageFormat = image_lib.ImageFormat
    HAS_MEDIAPIPE = True
except Exception:
    pass


def _find_model_path():
    for d in MODEL_DIRS:
        for name in MODEL_NAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


def create_face_landmarker():
    """Create FaceLandmarker if MediaPipe and model are available. Else return None."""
    if not HAS_MEDIAPIPE:
        return None
    path = _find_model_path()
    if not path:
        import sys
        print("Face Landmarker: no model file found. Looked in:", file=sys.stderr)
        for d in MODEL_DIRS:
            for name in MODEL_NAMES:
                print("  ", os.path.join(d, name), file=sys.stderr)
        return None
    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception as e:
        import sys
        print("Face Landmarker: failed to load model:", path, file=sys.stderr)
        print("  ", e, file=sys.stderr)
        return None


def detect_face(landmarker, frame_rgb, timestamp_ms):
    """
    Run face landmarker on RGB frame (numpy, HWC, uint8).
    Returns (face_bbox, landmarks, blendshapes) or (None, None, None).
    face_bbox = (x, y, w, h) in pixel coords; landmarks = list of NormalizedLandmark; blendshapes = list of Category.
    """
    if landmarker is None or frame_rgb is None:
        return None, None, None
    try:
        h, w = frame_rgb.shape[:2]
        if not frame_rgb.flags.c_contiguous:
            frame_rgb = frame_rgb.copy()
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None, None, None
        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min = int(min(xs) * w)
        x_max = int(max(xs) * w)
        y_min = int(min(ys) * h)
        y_max = int(max(ys) * h)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
        return bbox, landmarks, blendshapes
    except Exception:
        return None, None, None


def _score(blendshapes, name, idx):
    """Get blendshape score by name (or by index if name not found)."""
    name_lower = name.lower()
    for c in blendshapes:
        cn = (c.category_name or "").strip()
        if c.score is not None and (cn == name or cn.lower() == name_lower):
            return float(c.score)
    if 0 <= idx < len(blendshapes) and blendshapes[idx].score is not None:
        return float(blendshapes[idx].score)
    return 0.0


def emotion_from_blendshapes(blendshapes, smooth_prev=None):
    """
    Infer emotion from MediaPipe face blendshape scores.
    Returns one of: happy, sad, surprised, angry, neutral.
    """
    if not blendshapes:
        return smooth_prev if smooth_prev else "neutral"
    n = len(blendshapes)
    smile = (_score(blendshapes, BLENDSHAPE_SMILE_LEFT, 44) + _score(blendshapes, BLENDSHAPE_SMILE_RIGHT, 45)) / 2
    jaw_open = _score(blendshapes, BLENDSHAPE_JAW_OPEN, 25)
    brow_down = (_score(blendshapes, BLENDSHAPE_BROW_DOWN_LEFT, 1) + _score(blendshapes, BLENDSHAPE_BROW_DOWN_RIGHT, 2)) / 2
    brow_up = _score(blendshapes, BLENDSHAPE_BROW_INNER_UP, 3)
    frown = (_score(blendshapes, BLENDSHAPE_MOUTH_FROWN_LEFT, 30) + _score(blendshapes, BLENDSHAPE_MOUTH_FROWN_RIGHT, 31)) / 2

    if smile > 0.4:
        emotion = "happy"
    elif frown > 0.35:
        emotion = "sad"
    elif brow_down > 0.4:
        emotion = "angry"
    elif jaw_open > 0.35 or brow_up > 0.3:
        emotion = "surprised"
    else:
        emotion = "neutral"

    if smooth_prev and smooth_prev != emotion:
        # Optional: require 2 consecutive frames to switch (reduce jitter)
        pass  # we could add a small state machine here
    return emotion
