"""
Face-following captions: subtitles anchored near your face that follow you.
- Real-time webcam + face detection
- Speech-to-text captions in Minecraft-style font
- Emotion-based emoji and styling
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import sys

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

# --- Config ---
CAMERA_INDEX = 1
# Bigger, readable captions (large so "I oh" and short phrases are easy to read)
CAPTION_FONT_SIZE = 52
CAPTION_MAX_WIDTH = 580
CAPTION_OFFSET_ABOVE_HEAD = 22
MAX_CAPTION_LEN = 100
EMOTION_SMOOTH = 0.3
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# Prefer 720p capture = smooth stream (no heavy 1080p resize every frame)
CAMERA_RESOLUTIONS = [(1280, 720), (1920, 1080), (960, 540), (0, 0)]
# Max size we process/show (if camera is larger, we resize to this = less lag)
DISPLAY_SIZE = (1280, 720)
# Face: every 2nd frame + fast resize to keep stream smooth
FACE_DETECT_SCALE = 0.4
FACE_DETECT_EVERY_N = 1
FACE_SMOOTH = 0.55
# Minecraft-style chat background
CAPTION_BG_COLOR = (30, 30, 30, 200)
CAPTION_BG_PADDING = 12  # Slightly more padding for larger text

# Emotion -> (emoji, optional color tint name)
EMOTIONS = {
    "happy": ("😊", "yellow"),
    "sad": ("😢", "blue"),
    "surprised": ("😲", "orange"),
    "angry": ("😠", "red"),
    "neutral": ("😐", "white"),
}


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


def render_caption_pil(text: str, font_size: int, font_path: str = None) -> np.ndarray:
    """Render text with Minecraft font and semi-transparent dark background (BGRA)."""
    if not text or not HAS_PIL:
        return None
    font = get_minecraft_font(font_size)
    lines = _wrap_text(text, font, CAPTION_MAX_WIDTH)
    line_height = font_size + 4
    pad = CAPTION_BG_PADDING
    # Measure actual text width (max line width)
    try:
        max_w = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
    except Exception:
        max_w = CAPTION_MAX_WIDTH
    total_w = max_w + 2 * pad
    total_h = line_height * len(lines) + 2 * pad
    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Minecraft-style light black semi-transparent background
    draw.rectangle([(0, 0), (total_w - 1, total_h - 1)], fill=CAPTION_BG_COLOR, outline=None)
    for i, line in enumerate(lines):
        y = pad + i * line_height
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text((pad + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((pad, y), line, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def overlay_caption_on_frame(frame: np.ndarray, caption_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Blend caption image onto frame at (x, y); frame is BGR."""
    if caption_rgba is None or caption_rgba.size == 0:
        return frame
    h, w = caption_rgba.shape[:2]
    # Keep within frame
    x = max(0, min(x, frame.shape[1] - w))
    y = max(0, min(y, frame.shape[0] - h))
    roi = frame[y : y + h, x : x + w]
    if roi.shape[:2] != (h, w):
        return frame
    alpha = caption_rgba[:, :, 3:4] / 255.0
    rgb = caption_rgba[:, :, :3]
    roi[:] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    return frame


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
        if mode == "streaming":
            print("Caption mode: streaming (real-time)")
        else:
            print("Caption mode: fast batch (install vosk + model for streaming)")
    else:
        print("Speech: install speech_recognition and PyAudio (and optionally vosk for real-time)")

    # Caption: show only current content (no old + new). Partial while speaking, single last final when done.
    live_partial = ""
    last_final = ""  # single most recent phrase only (replaced on new final)
    current_caption = ""
    emotion_estimator = EmotionEstimator()
    emotion = "neutral"
    last_caption_render = None
    last_caption_text = ""
    window_name = "Face-following captions (Q to quit)"

    # Smoothed face position so caption doesn't jump
    face_center_x, face_top_y = 0.0, 0.0
    last_face_bbox = None
    frame_count = 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    disp_w, disp_h = DISPLAY_SIZE
    while True:
        frame = frame_reader.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        # Only resize if camera gave us something bigger (keeps 720p path zero-cost)
        if w > disp_w or h > disp_h:
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            h, w = disp_h, disp_w
        if face_center_x == 0 and face_top_y == 0:
            face_center_x, face_top_y = w / 2, h / 3

        # Run face detection on a small image and only every N frames (much faster)
        frame_count += 1
        if frame_count % FACE_DETECT_EVERY_N == 0:
            small_w = max(80, int(w * FACE_DETECT_SCALE))
            small_h = max(60, int(h * FACE_DETECT_SCALE))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            faces = face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(20, 20),
            )
            if len(faces) > 0:
                x_f, y_f, bw, bh = faces[0]
                # Scale bbox back to full frame
                scale_x, scale_y = w / small_w, h / small_h
                x_f = int(x_f * scale_x)
                y_f = int(y_f * scale_y)
                bw = int(bw * scale_x)
                bh = int(bh * scale_y)
                last_face_bbox = (x_f, y_f, bw, bh)
                new_cx = x_f + bw / 2
                new_ty = float(y_f)
                face_center_x = face_center_x + FACE_SMOOTH * (new_cx - face_center_x)
                face_top_y = face_top_y + FACE_SMOOTH * (new_ty - face_top_y)
            else:
                last_face_bbox = None

        face_bbox = last_face_bbox
        emotion = emotion_estimator.update(None)

        # Drain STT queue: partial = live text as you speak; final = replace (only that phrase, no old text)
        try:
            while True:
                item = text_queue.get_nowait()
                text, is_final = item if isinstance(item, tuple) and len(item) == 2 else (item, True)
                text = (text or "").strip()
                if not text:
                    continue
                if is_final:
                    last_final = text   # replace with new phrase only (no concatenation)
                    live_partial = ""
                else:
                    live_partial = text
        except queue.Empty:
            pass
        # Show only current: live words while speaking, or the single phrase you just said
        current_caption = live_partial if live_partial else last_final
        if len(current_caption) > MAX_CAPTION_LEN:
            current_caption = current_caption[-MAX_CAPTION_LEN:].strip()

        # Build display line: emoji + caption
        emoji, _ = EMOTIONS.get(emotion, EMOTIONS["neutral"])
        display_text = f"{emoji} {current_caption}" if current_caption else emoji
        if not display_text.strip():
            display_text = emoji

        if display_text != last_caption_text or last_caption_render is None:
            last_caption_render = render_caption_pil(display_text, CAPTION_FONT_SIZE)
            last_caption_text = display_text

        # Position caption above face (use smoothed position)
        cx, ty = int(face_center_x), int(face_top_y)
        caption_x = cx
        caption_y = ty - CAPTION_OFFSET_ABOVE_HEAD
        if last_caption_render is not None:
            cw = last_caption_render.shape[1]
            caption_x = cx - cw // 2
            caption_y = ty - last_caption_render.shape[0] - CAPTION_OFFSET_ABOVE_HEAD
            frame = overlay_caption_on_frame(frame, last_caption_render, caption_x, caption_y)

        # Optional: draw face box
        if face_bbox:
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if stt:
        stt.stop()
    frame_reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
