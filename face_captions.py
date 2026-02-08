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
# Face: stick to your head (smoother = less "running away")
FACE_DETECT_SCALE = 0.45
FACE_DETECT_EVERY_N = 1
FACE_SMOOTH = 0.28
MAX_CAPTION_LINES = 2
# Real-time: show full text immediately (no reveal delay). Set to 3–5 for smooth type-in effect.
REVEAL_CHARS_PER_FRAME = 999  # 999 = no delay; lower = smoother type-in, more delay
# Minecraft-style chat background (default when emotion color not used)
CAPTION_BG_COLOR = (30, 30, 30, 200)
CAPTION_BG_PADDING = 12
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
# UI toggles
SHOW_FACE_BOX = False              # Set True to draw green face outline
SPEECH_BUBBLE_TAIL = True         # Draw a tail on the caption box pointing to head
CAPTION_HISTORY_LINES = 2         # 0 = current only; 2 = show last 2 phrases + current (chat-style)
FADE_IN_FRAMES = 4                # Fade-in animation when caption updates (0 = no fade)


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
) -> np.ndarray:
    """Render text in 2 lines; emotion-tinted box; optional speech-bubble tail."""
    if not text or not HAS_PIL:
        return None
    font = get_minecraft_font(font_size)
    # Multi-phrase (chat history): split by newline, wrap each, show last 6 lines so history is visible
    if "\n" in text:
        all_lines = []
        for phrase in text.split("\n"):
            phrase = (phrase or "").strip()
            if phrase:
                all_lines.extend(_wrap_text(phrase, font, CAPTION_MAX_WIDTH))
        max_total = 6 if "\n" in text else MAX_CAPTION_LINES
        lines = all_lines[-max_total:]
    else:
        lines = _wrap_text(text, font, CAPTION_MAX_WIDTH)
        if len(lines) > MAX_CAPTION_LINES:
            lines = lines[-MAX_CAPTION_LINES:]
    line_height = font_size + 4
    pad = CAPTION_BG_PADDING
    try:
        max_w = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
    except Exception:
        max_w = CAPTION_MAX_WIDTH
    total_w = max_w + 2 * pad
    box_h = line_height * len(lines) + 2 * pad
    tail_h = 10 if speech_bubble else 0
    total_h = box_h + tail_h
    fill_color = bg_color if bg_color is not None else CAPTION_BG_COLOR
    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (total_w - 1, box_h - 1)], fill=fill_color, outline=None)
    if speech_bubble and tail_h:
        cx = total_w // 2
        tail_w = 12
        draw.polygon(
            [(cx - tail_w, box_h - 1), (cx + tail_w, box_h - 1), (cx, total_h - 1)],
            fill=fill_color,
            outline=None,
        )
    for i, line in enumerate(lines):
        y = pad + i * line_height
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text((pad + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
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

    # Caption: show only current content; smooth reveal so text flows in
    live_partial = ""
    last_final = ""
    caption_history = []
    current_caption = ""
    reveal_len = 0
    emotion_estimator = EmotionEstimator()
    emotion = "neutral"
    last_caption_render = None
    last_caption_text = ""
    frames_since_caption_change = 999
    show_face_box = SHOW_FACE_BOX
    window_name = "Face-following captions (Q=quit B=face box)"

    face_center_x, face_top_y = 0.0, 0.0
    last_face_bbox = None
    frame_count = 0
    emotion_smooth_prev = "neutral"

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

        if face_landmarker_mp and (frame_count % FACE_DETECT_EVERY_N == 0):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_mp, landmarks_mp, blendshapes_mp = detect_face(face_landmarker_mp, rgb, timestamp_ms)
            if bbox_mp is not None:
                x_f, y_f, bw, bh = bbox_mp
                last_face_bbox = (x_f, y_f, bw, bh)
                new_cx = x_f + bw / 2
                new_ty = float(y_f)
                face_center_x = face_center_x + FACE_SMOOTH * (new_cx - face_center_x)
                face_top_y = face_top_y + FACE_SMOOTH * (new_ty - face_top_y)
                emotion = emotion_from_blendshapes(blendshapes_mp, emotion_smooth_prev)
                emotion_smooth_prev = emotion
            else:
                last_face_bbox = None
                emotion = emotion_smooth_prev
        elif frame_count % FACE_DETECT_EVERY_N == 0:
            small_w = max(80, int(w * FACE_DETECT_SCALE))
            small_h = max(60, int(h * FACE_DETECT_SCALE))
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
                face_center_x = face_center_x + FACE_SMOOTH * (new_cx - face_center_x)
                face_top_y = face_top_y + FACE_SMOOTH * (new_ty - face_top_y)
            else:
                last_face_bbox = None
            emotion = emotion_estimator.update(None)

        face_bbox = last_face_bbox

        # Drain STT queue: partial = live text as you speak; final = replace (only that phrase, no old text)
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
                    if CAPTION_HISTORY_LINES > 0 and text:
                        caption_history.append(text)
                        if len(caption_history) > CAPTION_HISTORY_LINES:
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

        # Build display: optional history (older on top) + current line (no duplicate of last final)
        if CAPTION_HISTORY_LINES > 0 and caption_history:
            if displayed_caption and displayed_caption != caption_history[-1]:
                display_text = "\n".join(caption_history) + "\n" + displayed_caption
            else:
                display_text = "\n".join(caption_history)
        else:
            display_text = displayed_caption if displayed_caption else ""
        if not display_text.strip():
            display_text = "..."

        if display_text != last_caption_text or last_caption_render is None:
            frames_since_caption_change = 0
            last_caption_text = display_text
            color_name = EMOTIONS.get(emotion, EMOTIONS["neutral"])[1]
            bg_color = EMOTION_COLORS.get(color_name, CAPTION_BG_COLOR)
            last_caption_render = render_caption_pil(
                display_text,
                CAPTION_FONT_SIZE,
                bg_color=bg_color,
                speech_bubble=SPEECH_BUBBLE_TAIL,
            )
        else:
            frames_since_caption_change += 1

        # Fade-in: ramp alpha for first FADE_IN_FRAMES when caption changes
        alpha_mult = 1.0
        if FADE_IN_FRAMES > 0 and frames_since_caption_change < FADE_IN_FRAMES:
            alpha_mult = 0.5 + 0.5 * (frames_since_caption_change / FADE_IN_FRAMES)

        cx, ty = int(face_center_x), int(face_top_y)
        caption_x = cx
        caption_y = ty - CAPTION_OFFSET_ABOVE_HEAD
        if last_caption_render is not None:
            cw = last_caption_render.shape[1]
            caption_x = cx - cw // 2
            caption_y = ty - last_caption_render.shape[0] - CAPTION_OFFSET_ABOVE_HEAD
            frame = overlay_caption_on_frame(
                frame, last_caption_render, caption_x, caption_y, alpha_mult=alpha_mult
            )

        if show_face_box and face_bbox:
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            show_face_box = not show_face_box

    if stt:
        stt.stop()
    frame_reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
