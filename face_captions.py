"""
Face-following captions: subtitles anchored near your face that follow you.
- Real-time webcam + face detection
- Speech-to-text captions in Minecraft-style font
- Emotion-based emoji and styling
"""

import cv2
import numpy as np
import re
import threading
import queue
import time
import os
import sys

# Load .env so DEEPGRAM_API_KEY (and others) are set before STT starts
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# Real-time speech: use project library (streaming when possible)
try:
    from realtime_stt import StreamingSTT, RESULT_FINAL, RESULT_PARTIAL, get_caption_mode
    HAS_STT = True
except ImportError:
    HAS_STT = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# MediaPipe Face Mesh (optional): better tracking + real emotion from blendshapes
try:
    from face_mesh import (
        create_face_landmarker,
        detect_face,
        emotion_from_blendshapes,
        HAS_MEDIAPIPE as HAS_FACE_MESH_MODULE,
    )
    HAS_FACE_MESH = HAS_FACE_MESH_MODULE
except ImportError:
    HAS_FACE_MESH = False
    create_face_landmarker = None
    detect_face = None
    emotion_from_blendshapes = None

# MediaPipe Hands: prefer Tasks API (hand_landmarker.task), fallback to legacy solutions
try:
    from hand_landmarker import (
        create_hand_landmarker,
        detect_hands as detect_hands_tasks,
        HAS_HAND_TASKS,
    )
except ImportError:
    create_hand_landmarker = None
    detect_hands_tasks = None
    HAS_HAND_TASKS = False

try:
    import mediapipe as mp
    _solutions = getattr(mp, "solutions", None)
    _mp_hands_module = getattr(_solutions, "hands", None) if _solutions else None
    HAS_HANDS_LEGACY = _mp_hands_module is not None
except (ImportError, AttributeError):
    _mp_hands_module = None
    HAS_HANDS_LEGACY = False

# Optional translation (pip install googletrans==4.0.0-rc1; may conflict with deepgram's httpcore)
try:
    from googletrans import Translator
    _translator = Translator()
    HAS_TRANSLATE = True
except (ImportError, AttributeError):
    _translator = None
    HAS_TRANSLATE = False

# --- Config ---
CAMERA_INDEX = 1
# Bigger, readable captions (large so "I oh" and short phrases are easy to read)
CAPTION_FONT_SIZE = 52
CAPTION_MAX_WIDTH = 620
# Gap between bottom of caption box and top of head (larger = caption sits well above head)
CAPTION_OFFSET_ABOVE_HEAD = 55
MAX_CAPTION_LEN = 220
# Captions disappear after this many seconds (longer = see messages a bit longer)
CAPTION_TIMEOUT_SEC = 4.5
EMOTION_SMOOTH = 0.3
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# Prefer 720p capture = smooth stream (no heavy 1080p resize every frame)
CAMERA_RESOLUTIONS = [(1280, 720), (1920, 1080), (960, 540), (0, 0)]
# Max size we process/show (if camera is larger, we resize to this = less lag)
DISPLAY_SIZE = (1280, 720)
# Face: smaller scale = faster detection; every Nth frame = smoother video
FACE_DETECT_SCALE = 0.35
FACE_DETECT_EVERY_N = 3
FACE_SMOOTH = 0.28
MAX_CAPTION_LINES = 2
# Instant captions: show text as soon as it’s said (999 = no typewriter delay)
REVEAL_CHARS_PER_FRAME = 999
# Minecraft-style chat background (default when emotion color not used)
CAPTION_BG_COLOR = (45, 42, 34, 230)  # dark wood/chest tone for Minecraft look
CAPTION_BG_PADDING = 12
# Pixel border for Minecraft GUI look (highlight top/left, shadow bottom/right)
CAPTION_BORDER_LIGHT = (90, 85, 72, 255)   # bevel highlight
CAPTION_BORDER_DARK = (20, 18, 15, 255)    # bevel shadow
CAPTION_BORDER_PX = 2                        # border width in pixels
# Pinch-to-scale caption box (hand gesture: pinch in = shrink, pinch out = grow)
CAPTION_SCALE_MIN = 0.55
CAPTION_SCALE_MAX = 1.85
PINCH_SENSITIVITY = 0.012   # lower = smaller pinch gesture triggers resize
# Emotion -> (emoji, color name). Color used to tint the caption box.
EMOTIONS = {
    "happy": ("😊", "yellow"),
    "sad": ("😢", "blue"),
    "surprised": ("😲", "orange"),
    "angry": ("😠", "red"),
    "neutral": ("😐", "white"),
}
# Emotion-based box tint (RGBA). Makes captions pop by mood.
EMOTION_COLORS = {
    "yellow": (45, 45, 15, 220),   # happy
    "blue": (15, 25, 45, 220),     # sad
    "orange": (50, 35, 15, 220),  # surprised
    "red": (50, 20, 20, 220),     # angry
    "white": (35, 35, 35, 200),   # neutral
}
# UI toggles (can also be toggled live: T=tail, H=history, F=fade, C=color filter, M=translate)
SHOW_FACE_BOX = False              # Set True to draw green face outline
SPEECH_BUBBLE_TAIL = True         # Draw a tail on the caption box pointing to head
CAPTION_HISTORY_LINES = 2         # Exactly 2 lines: top = oldest, bottom = newest (real-time)
FADE_IN_FRAMES = 2                # Short fade for smoother pop-in (0 = instant)
# Emotion: require this many consecutive frames before switching (reduces jitter)
EMOTION_HOLD_FRAMES = 3
# Translation (optional): pip install googletrans==4.0.0-rc1
TRANSLATION_DEST = "es"           # Target language code when translation is enabled


def get_minecraft_font(size: int):
    """Load Minecraft-style font; fallback to default if missing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [script_dir, FONT_DIR]
    names = ["Minecraft.ttf", "Minecraftia.ttf", "MinecraftRegular.ttf"]
    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int) -> list:
    """Simple word wrap; returns list of lines."""
    words = text.split()
    if not words:
        return [text] if text else [""]
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        try:
            bbox = font.getbbox(candidate)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word if font.getbbox(word)[2] - font.getbbox(word)[0] <= max_width else candidate
        except Exception:
            lines.append(candidate)
            current = ""
    if current:
        lines.append(current)
    return lines or [text]


def render_caption_pil(
    text: str,
    font_size: int,
    bg_color: tuple = None,
    speech_bubble: bool = True,
    max_width: int = None,
    padding: int = None,
) -> np.ndarray:
    """Render text in 2 lines; emotion-tinted box; optional speech-bubble tail. max_width/padding for pinch scale."""
    if not text or not HAS_PIL:
        return None
    font = get_minecraft_font(font_size)
    use_max_width = max_width if max_width is not None else CAPTION_MAX_WIDTH
    pad = padding if padding is not None else CAPTION_BG_PADDING
    # Two lines only: oldest on top, newest on bottom
    if "\n" in text:
        all_lines = []
        for phrase in text.split("\n"):
            phrase = (phrase or "").strip()
            if phrase:
                all_lines.extend(_wrap_text(phrase, font, use_max_width))
        max_total = 2
        lines = all_lines[-max_total:]
    else:
        lines = _wrap_text(text, font, use_max_width)
        if len(lines) > MAX_CAPTION_LINES:
            lines = lines[-MAX_CAPTION_LINES:]
    line_height = font_size + 4
    try:
        max_w = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
    except Exception:
        max_w = use_max_width
    total_w = max_w + 2 * pad
    box_h = line_height * len(lines) + 2 * pad
    tail_h = max(6, 10 * font_size // 52) if speech_bubble else 0
    total_h = box_h + tail_h
    fill_color = bg_color if bg_color is not None else CAPTION_BG_COLOR
    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (total_w - 1, box_h - 1)], fill=fill_color, outline=None)
    b = CAPTION_BORDER_PX
    dark = (0, 0, 0)
    for i in range(b):
        draw.line([(i, i), (total_w - 1 - i, i)], fill=(255, 255, 255))
        draw.line([(i, i), (i, box_h - 1 - i)], fill=(255, 255, 255))
        draw.line([(i, box_h - 1 - i), (total_w - 1 - i, box_h - 1 - i)], fill=dark)
        draw.line([(total_w - 1 - i, i), (total_w - 1 - i, box_h - 1 - i)], fill=dark)
    fc = fill_color if isinstance(fill_color, (tuple, list)) and len(fill_color) >= 3 else (45, 42, 34)
    inner_shadow_color = (
        max(0, fc[0] - 15),
        max(0, fc[1] - 15),
        max(0, fc[2] - 15),
    )
    for i in range(1):
        offset = b + i
        draw.line([(offset, offset), (total_w - 1 - offset, offset)], fill=inner_shadow_color)
        draw.line([(offset, offset), (offset, box_h - 1 - offset)], fill=inner_shadow_color)
    if speech_bubble and tail_h:
        cx = total_w // 2
        tail_w = 12
        draw.polygon(
            [(cx - tail_w, box_h - 1), (cx + tail_w, box_h - 1), (cx, total_h - 1)],
            fill=fill_color,
            outline=None,
        )
        draw.line([(cx - tail_w, total_h - 1), (cx + tail_w, total_h - 1)], fill=dark)
    shadow_offsets = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
    for i, line in enumerate(lines):
        y = pad + i * line_height
        for dx, dy in shadow_offsets:
            draw.text((pad + dx, y + dy), line, font=font, fill=(0, 0, 0, 200))
        draw.text((pad + 1, y + 1), line, font=font, fill=(0, 0, 0, 255))
        draw.text((pad, y), line, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def overlay_caption_on_frame(
    frame: np.ndarray,
    caption_rgba: np.ndarray,
    x: int,
    y: int,
    alpha_mult: float = 1.0,
) -> np.ndarray:
    """Blend caption onto frame at (x, y). alpha_mult for fade-in (0..1)."""
    if caption_rgba is None or caption_rgba.size == 0:
        return frame
    h, w = caption_rgba.shape[:2]
    x = max(0, min(x, frame.shape[1] - w))
    y = max(0, min(y, frame.shape[0] - h))
    roi = frame[y : y + h, x : x + w]
    if roi.shape[:2] != (h, w):
        return frame
    alpha = (caption_rgba[:, :, 3:4] / 255.0) * alpha_mult
    rgb = caption_rgba[:, :, :3]
    roi[:] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


def _pinch_distance(hand_landmarks) -> float:
    """Distance between thumb tip (4) and index tip (8) in image coords. Returns float or None."""
    if not hand_landmarks or len(hand_landmarks.landmark) < 9:
        return None
    t = hand_landmarks.landmark[4]
    i = hand_landmarks.landmark[8]
    return ((t.x - i.x) ** 2 + (t.y - i.y) ** 2) ** 0.5


def _pinch_center_px(hand_landmarks, frame_w: int, frame_h: int):
    """Legacy API: midpoint of thumb (4) and index (8) in pixel coords. Returns (x, y) or None."""
    if not hand_landmarks or len(hand_landmarks.landmark) < 9:
        return None
    t = hand_landmarks.landmark[4]
    i = hand_landmarks.landmark[8]
    mid_x = (t.x + i.x) / 2
    mid_y = (t.y + i.y) / 2
    return (int(mid_x * frame_w), int(mid_y * frame_h))


def _pinch_from_tasks_landmarks(landmarks, frame_w: int, frame_h: int):
    """Tasks API: from list of 21 NormalizedLandmark return (pinch_dist, (center_x, center_y)) or (None, None)."""
    if not landmarks or len(landmarks) < 9:
        return None, None
    t = landmarks[4]
    i = landmarks[8]
    tx, ty = getattr(t, "x", 0), getattr(t, "y", 0)
    ix, iy = getattr(i, "x", 0), getattr(i, "y", 0)
    dist = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
    cx = int((tx + ix) / 2 * frame_w)
    cy = int((ty + iy) / 2 * frame_h)
    return dist, (cx, cy)


# Handle zone: right-bottom corner of caption box (fraction of width/height)
CAPTION_HANDLE_RIGHT_FRAC = 0.45   # use right 55% of box (bigger = easier to hit)
CAPTION_HANDLE_BOTTOM_FRAC = 0.40  # use bottom 60% of box


class FaceKalmanTracker:
    """Smooth face position with a Kalman filter (position + velocity model)."""
    def __init__(self):
        # 4 states (x, y, dx, dy), 2 measurements (x, y)
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0]
        ], dtype=np.float32)
        # Tuned: lower process noise = smoother, trust predictions more
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        # Tuned: higher measurement noise = smoother (was 0.1)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        # Tuned: better initial uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        self.initialized = False
        self.frames_lost = 0

    def update(self, x=None, y=None):
        """Update with measurement (x, y). If x is None, only predict (no face)."""
        if x is None or y is None:
            if not self.initialized:
                return None, None
            self.frames_lost += 1
            if self.frames_lost > 30:
                self.kf.errorCovPost *= 1.05
            pred = self.kf.predict()
            return float(pred[0, 0]), float(pred[1, 0])
        self.frames_lost = 0
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
            return float(x), float(y)
        self.kf.predict()
        estimated = self.kf.correct(measurement)
        if self.kf.errorCovPost[0, 0] > 1.0:
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        return float(estimated[0, 0]), float(estimated[1, 0])


class EmotionEstimator:
    """Emotion placeholder when using OpenCV face detection (no landmarks)."""

    def __init__(self):
        self.smoothed = "neutral"

    def update(self, _landmarks=None) -> str:
        return self.smoothed


class FrameReader:
    """Reads latest frame in a thread so main loop always gets freshest frame (smoother video)."""
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.latest = frame.copy()

    def start(self):
        self._thread.start()

    def read(self):
        with self.lock:
            if self.latest is not None:
                return self.latest.copy()
        return None

    def stop(self):
        self.running = False
        self._thread.join(timeout=1.0)


def main():
    if not HAS_PIL:
        print("Install Pillow: pip install Pillow")
        sys.exit(1)

    # OpenCV built-in face detector (no extra model download)
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Could not load face cascade.")
        sys.exit(1)

    # Best camera quality: DirectShow on Windows, try highest resolutions first
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Check camera index.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for rw, rh in CAMERA_RESOLUTIONS:
        if rw and rh:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w >= 320 and actual_h >= 240:
            print(f"Camera: {actual_w}x{actual_h}")
            break
    frame_reader = FrameReader(cap)
    frame_reader.start()
    time.sleep(0.3)

    text_queue = queue.Queue()
    stt = StreamingSTT(text_queue) if HAS_STT else None
    if HAS_STT and stt.start():
        mode = get_caption_mode()
        engine = getattr(stt, "_engine", "batch")
        if mode == "streaming":
            print("✓ Caption mode: streaming (real-time) — using", engine)
        else:
            print("✓ Caption mode: fast batch (install faster-whisper or vosk for streaming)")
        latency_info = {
            "deepgram": "100-200ms",
            "faster-whisper": "200-300ms",
            "vosk": "150-250ms",
            "batch": "600-800ms",
        }
        print("  Expected latency:", latency_info.get(engine, "varies"))
    else:
        print("Speech: install speech_recognition and PyAudio (and optionally vosk for real-time)")

    # Caption: 2 lines only (oldest top, newest bottom); entries expire after CAPTION_TIMEOUT_SEC
    live_partial = ""
    last_final = ""
    caption_history = []  # list of (text, timestamp)
    current_caption = ""
    reveal_len = 0
    emotion_estimator = EmotionEstimator()
    emotion = "neutral"
    last_caption_render = None
    last_caption_text = ""
    frames_since_caption_change = 999
    show_face_box = SHOW_FACE_BOX
    show_speech_tail = SPEECH_BUBBLE_TAIL
    caption_history_lines = CAPTION_HISTORY_LINES
    fade_in_frames = FADE_IN_FRAMES
    apply_color_filter = False
    translate_enabled = False
    last_translated = ""
    last_final_translated = ""  # which last_final we translated
    emotion_hold_prev = "neutral"
    emotion_hold_frames = 0
    window_name = "Face captions (Q=quit B=box T=tail H=history F=fade C=color M=translate P=pinch +=/-=size)"
    caption_cache = {}
    MAX_CAPTION_CACHE_SIZE = 10
    last_caption_scale = 1.0
    last_caption_x, last_caption_y, last_caption_w, last_caption_h = None, None, None, None
    fps_history = []
    last_fps_time = time.time()

    face_center_x, face_top_y = 0.0, 0.0
    last_face_bbox = None
    frame_count = 0
    emotion_smooth_prev = "neutral"
    face_tracker = FaceKalmanTracker()
    caption_scale = 1.0
    last_pinch_dist = None
    pinch_enabled = True

    # Hand landmarker: prefer MediaPipe Tasks API (same as face), fallback to legacy solutions
    hand_landmarker_tasks = None
    if create_hand_landmarker:
        hand_landmarker_tasks = create_hand_landmarker()
    hand_landmarker = None
    if HAS_HANDS_LEGACY and _mp_hands_module:
        try:
            hand_landmarker = _mp_hands_module.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )
        except Exception:
            hand_landmarker = None
    HAS_HANDS = hand_landmarker_tasks is not None or hand_landmarker is not None
    if HAS_HANDS:
        print("Caption resize: pinch at box corner (P=toggle) or use + / - keys")
    else:
        print("Caption resize: use + / - keys to change size (run download_hand_landmarker_model.py for pinch)")

    # Prefer MediaPipe Face Mesh if model is present (smoother tracking + real emotion)
    face_landmarker_mp = None
    if HAS_FACE_MESH and create_face_landmarker:
        face_landmarker_mp = create_face_landmarker()
        if face_landmarker_mp:
            print("Face: MediaPipe Face Mesh (emotion from expression)")
        else:
            print("Face: MediaPipe model not found; run download_face_landmarker_model.py")
    if not face_landmarker_mp:
        print("Face: OpenCV Haar cascade")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    disp_w, disp_h = DISPLAY_SIZE
    while True:
        frame = frame_reader.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if w > disp_w or h > disp_h:
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            h, w = disp_h, disp_w
        if face_center_x == 0 and face_top_y == 0:
            face_center_x, face_top_y = w / 2, h / 3

        frame_count += 1
        timestamp_ms = int(frame_count * 1000 / 30)

        detect_interval = 2 if last_face_bbox is None else 4
        detect_scale = 0.50 if last_face_bbox is None else 0.30

        if face_landmarker_mp and (frame_count % detect_interval == 0):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_mp, landmarks_mp, blendshapes_mp = detect_face(face_landmarker_mp, rgb, timestamp_ms)
            if bbox_mp is not None:
                x_f, y_f, bw, bh = bbox_mp
                last_face_bbox = (x_f, y_f, bw, bh)
                new_cx = x_f + bw / 2
                new_ty = float(y_f)
                cx, ty = face_tracker.update(new_cx, new_ty)
                if cx is not None:
                    face_center_x, face_top_y = cx, ty
                emotion = emotion_from_blendshapes(blendshapes_mp, emotion_smooth_prev)
                emotion_smooth_prev = emotion
            else:
                last_face_bbox = None
                cx, ty = face_tracker.update(None, None)
                if cx is not None:
                    face_center_x, face_top_y = cx, ty
                emotion = emotion_smooth_prev
        # Pinch-to-scale only when hand is in the caption box's right-bottom "handle" zone
        # Use last frame's actual caption position so the zone matches the drawn corner
        if pinch_enabled and (hand_landmarker_tasks or hand_landmarker):
            handle_x1, handle_y1, handle_x2, handle_y2 = None, None, None, None
            if last_caption_x is not None and last_caption_w is not None and last_caption_h is not None:
                handle_x1 = last_caption_x + int(last_caption_w * CAPTION_HANDLE_RIGHT_FRAC)
                handle_y1 = last_caption_y + int(last_caption_h * CAPTION_HANDLE_BOTTOM_FRAC)
                handle_x2 = last_caption_x + last_caption_w
                handle_y2 = last_caption_y + last_caption_h
            rgb_hands = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pinch_dist = None
            center_px = None
            if hand_landmarker_tasks and detect_hands_tasks:
                hands_result = detect_hands_tasks(hand_landmarker_tasks, rgb_hands, timestamp_ms)
                if hands_result and len(hands_result) > 0:
                    pinch_dist, center_px = _pinch_from_tasks_landmarks(hands_result[0], w, h)
            elif hand_landmarker:
                hand_results = hand_landmarker.process(rgb_hands)
                if hand_results.multi_hand_landmarks:
                    hl = hand_results.multi_hand_landmarks[0]
                    center_px = _pinch_center_px(hl, w, h)
                    pinch_dist = _pinch_distance(hl)

            if handle_x1 is not None and center_px is not None:
                in_handle = (
                    handle_x1 <= center_px[0] <= handle_x2
                    and handle_y1 <= center_px[1] <= handle_y2
                )
                if in_handle and pinch_dist is not None:
                    if last_pinch_dist is not None:
                        delta = pinch_dist - last_pinch_dist
                        if delta < -PINCH_SENSITIVITY:
                            caption_scale = max(CAPTION_SCALE_MIN, caption_scale * 0.96)
                        elif delta > PINCH_SENSITIVITY:
                            caption_scale = min(CAPTION_SCALE_MAX, caption_scale * 1.04)
                        caption_scale = max(CAPTION_SCALE_MIN, min(CAPTION_SCALE_MAX, caption_scale))
                    last_pinch_dist = pinch_dist
                else:
                    last_pinch_dist = None
            else:
                last_pinch_dist = None
        else:
            last_pinch_dist = None
        if frame_count % detect_interval == 0:
            small_w = max(80, int(w * detect_scale))
            small_h = max(60, int(h * detect_scale))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            faces = face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.12,
                minNeighbors=5,
                minSize=(24, 24),
            )
            if len(faces) > 0:
                scale_x, scale_y = w / small_w, h / small_h
                best = 0
                best_dist = float("inf")
                for i, (fx, fy, fw, fh) in enumerate(faces):
                    cx_scaled = fx + fw / 2
                    cy_scaled = fy
                    dx = (cx_scaled * scale_x) - face_center_x
                    dy = (cy_scaled * scale_y) - face_top_y
                    d = dx * dx + dy * dy
                    if d < best_dist:
                        best_dist = d
                        best = i
                x_f, y_f, bw, bh = faces[best]
                x_f = int(x_f * scale_x)
                y_f = int(y_f * scale_y)
                bw = int(bw * scale_x)
                bh = int(bh * scale_y)
                last_face_bbox = (x_f, y_f, bw, bh)
                new_cx = x_f + bw / 2
                new_ty = float(y_f)
                cx, ty = face_tracker.update(new_cx, new_ty)
                if cx is not None:
                    face_center_x, face_top_y = cx, ty
            else:
                last_face_bbox = None
                cx, ty = face_tracker.update(None, None)
                if cx is not None:
                    face_center_x, face_top_y = cx, ty
            emotion = emotion_estimator.update(None)

        if emotion != emotion_hold_prev:
            emotion_hold_frames += 1
            if emotion_hold_frames >= EMOTION_HOLD_FRAMES:
                emotion_hold_prev = emotion
                emotion_hold_frames = 0
            else:
                emotion = emotion_hold_prev
        else:
            emotion_hold_frames = 0

        face_bbox = last_face_bbox

        try:
            while True:
                item = text_queue.get_nowait()
                text, is_final = item if isinstance(item, tuple) and len(item) == 2 else (item, True)
                text = (text or "").strip()
                if not text:
                    continue
                if is_final:
                    last_final = text
                    live_partial = ""
                    if caption_history_lines > 0 and text:
                        now = time.time()
                        caption_history.append((text, now))
                        if len(caption_history) > caption_history_lines:
                            caption_history.pop(0)
                else:
                    live_partial = text
        except queue.Empty:
            pass
        current_caption = live_partial if live_partial else last_final
        # Remove placeholders and square/box chars (recognizer artifacts or missing glyphs)
        current_caption = re.sub(r"\s*\[\s*\]\s*", " ", current_caption)
        current_caption = re.sub(r"[\u25A0-\u25AB\u25FB\u25FC\uFFFD]", "", current_caption)  # □ ■ ▢ etc.
        current_caption = re.sub(r"\s+", " ", current_caption).strip()
        if len(current_caption) > MAX_CAPTION_LEN:
            current_caption = current_caption[-MAX_CAPTION_LEN:].strip()

        # Smooth reveal: when caption shrinks (new phrase), reset; else reveal more chars each frame
        target_len = len(current_caption)
        if target_len < reveal_len:
            reveal_len = target_len
        else:
            reveal_len = min(reveal_len + REVEAL_CHARS_PER_FRAME, target_len)
        displayed_caption = current_caption[:reveal_len]

        # Translation: when enabled, translate last_final once and show with original
        if translate_enabled and HAS_TRANSLATE and _translator and last_final and last_final != last_final_translated:
            try:
                last_translated = _translator.translate(last_final, dest=TRANSLATION_DEST).text or ""
                last_final_translated = last_final
            except Exception:
                last_translated = ""
                last_final_translated = last_final
        if not translate_enabled:
            last_translated = ""
            last_final_translated = ""

        # Exactly 2 lines: top = oldest, bottom = newest (real-time); expire after CAPTION_TIMEOUT_SEC
        now = time.time()
        if caption_history_lines > 0 and caption_history:
            # Drop expired entries (text goes away after timeout)
            caption_history[:] = [(t, ts) for t, ts in caption_history if now - ts < CAPTION_TIMEOUT_SEC]
            MAX_HISTORY_ENTRIES = 10
            if len(caption_history) > MAX_HISTORY_ENTRIES:
                caption_history[:] = caption_history[-MAX_HISTORY_ENTRIES:]
            # At most 2 lines: oldest, newest (newest can be live partial)
            valid = caption_history[-2:]
            line1 = valid[0][0] if len(valid) >= 1 else ""
            line2 = displayed_caption if displayed_caption else (valid[1][0] if len(valid) >= 2 else "")
            if line1 and line2:
                if line1.strip() == line2.strip():
                    display_text = line2  # avoid duplicate line
                else:
                    display_text = line1 + "\n" + line2
            else:
                display_text = line2 or line1 or ""
        else:
            display_text = displayed_caption if displayed_caption else ""
        if translate_enabled and last_translated and display_text.strip() and display_text != "...":
            display_text = (display_text.split("\n")[-1] if "\n" in display_text else display_text) + "\n(" + last_translated + ")"
        if not display_text.strip():
            display_text = "..."

        scaled_font = max(12, int(CAPTION_FONT_SIZE * caption_scale))
        scaled_max_width = max(200, int(CAPTION_MAX_WIDTH * caption_scale))
        scaled_padding = max(4, int(CAPTION_BG_PADDING * caption_scale))
        scale_key = round(caption_scale, 2)
        if display_text != last_caption_text or last_caption_render is None or scale_key != last_caption_scale:
            frames_since_caption_change = 0
            last_caption_text = display_text
            last_caption_scale = scale_key
            cache_key = (emotion, show_speech_tail, len(display_text) < 20, scale_key)
            if cache_key in caption_cache and caption_cache[cache_key][1] == display_text:
                last_caption_render = caption_cache[cache_key][0]
            else:
                color_name = EMOTIONS.get(emotion, EMOTIONS["neutral"])[1]
                bg_color = EMOTION_COLORS.get(color_name, CAPTION_BG_COLOR)
                last_caption_render = render_caption_pil(
                    display_text,
                    scaled_font,
                    bg_color=bg_color,
                    speech_bubble=show_speech_tail,
                    max_width=scaled_max_width,
                    padding=scaled_padding,
                )
                if last_caption_render is not None:
                    if len(caption_cache) >= MAX_CAPTION_CACHE_SIZE:
                        caption_cache.pop(next(iter(caption_cache)))
                    caption_cache[cache_key] = (last_caption_render, display_text)
        else:
            frames_since_caption_change += 1

        alpha_mult = 1.0
        if fade_in_frames > 0 and frames_since_caption_change < fade_in_frames:
            alpha_mult = 0.5 + 0.5 * (frames_since_caption_change / fade_in_frames)

        if apply_color_filter:
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

        lookahead_frames = 3
        if face_tracker.initialized and hasattr(face_tracker.kf, "statePost"):
            vx = float(face_tracker.kf.statePost[2, 0])
            vy = float(face_tracker.kf.statePost[3, 0])
            cx = int(face_center_x + vx * lookahead_frames)
            ty = int(face_top_y + vy * lookahead_frames)
        else:
            cx = int(face_center_x)
            ty = int(face_top_y)
        caption_x = cx
        caption_y = ty - CAPTION_OFFSET_ABOVE_HEAD
        if last_caption_render is not None:
            cw = last_caption_render.shape[1]
            ch = last_caption_render.shape[0]
            dynamic_offset = CAPTION_OFFSET_ABOVE_HEAD
            if face_bbox is not None:
                _, _, _, fh = face_bbox
                dynamic_offset = max(CAPTION_OFFSET_ABOVE_HEAD, int(fh * 0.3))
            caption_y = ty - ch - dynamic_offset
            caption_y = max(0, caption_y)
            caption_x = max(0, min(cx - cw // 2, w - cw))
            frame = overlay_caption_on_frame(
                frame, last_caption_render, caption_x, caption_y, alpha_mult=alpha_mult
            )
            last_caption_x, last_caption_y, last_caption_w, last_caption_h = caption_x, caption_y, cw, ch
            # Resize handle hint: always show corner when pinch is on (so user sees where to aim)
            if pinch_enabled:
                grip_size = max(14, min(cw, ch) // 3)
                x1, y1 = caption_x + cw - grip_size, caption_y + ch - grip_size
                x2, y2 = caption_x + cw, caption_y + ch
                color = (0, 255, 255) if HAS_HANDS else (120, 120, 120)  # bright when active, dim when keys-only
                cv2.line(frame, (x1, y2), (x2, y2), color, 2)
                cv2.line(frame, (x2, y1), (x2, y2), color, 2)
        else:
            last_caption_x, last_caption_y, last_caption_w, last_caption_h = None, None, None, None

        if show_face_box and face_bbox:
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

        cv2.imshow(window_name, frame)
        current_time = time.time()
        fps_history.append(current_time)
        fps_history = [t for t in fps_history if current_time - t < 1.0]
        fps = len(fps_history)
        if current_time - last_fps_time > 0.5:
            cv2.setWindowTitle(window_name, "Face Captions — {0} FPS (Q=quit +=/-=size P=pinch)".format(fps))
            last_fps_time = current_time
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            show_face_box = not show_face_box
        elif key == ord("t"):
            show_speech_tail = not show_speech_tail
        elif key == ord("h"):
            caption_history_lines = 0 if caption_history_lines > 0 else 2
        elif key == ord("f"):
            fade_in_frames = 0 if fade_in_frames > 0 else 4
        elif key == ord("c"):
            apply_color_filter = not apply_color_filter
        elif key == ord("m"):
            translate_enabled = not translate_enabled
            if not translate_enabled:
                last_translated = ""
                last_final_translated = ""
        elif key == ord("p"):
            pinch_enabled = not pinch_enabled
            print("Pinch resize:", "ON (pinch at caption corner)" if pinch_enabled else "OFF")
        elif key in (ord("+"), ord("=")):
            caption_scale = min(CAPTION_SCALE_MAX, caption_scale * 1.08)
            caption_scale = round(caption_scale, 2)
            print("Caption size: larger ({:.0%})".format(caption_scale))
        elif key == ord("-"):
            caption_scale = max(CAPTION_SCALE_MIN, caption_scale * 0.92)
            caption_scale = round(caption_scale, 2)
            print("Caption size: smaller ({:.0%})".format(caption_scale))

    if stt:
        stt.stop()
    frame_reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
