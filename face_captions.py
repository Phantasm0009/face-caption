"""
Face-following captions: subtitles anchored near your face that follow you.
- Real-time webcam + face detection
- Speech-to-text captions in Minecraft-style font
- Emotion-based emoji and styling
"""

import argparse
import json
import cv2
import numpy as np
import re
import threading
import queue
import time
import os
import sys
from collections import OrderedDict

# Optional WebSocket control API
try:
    import asyncio
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# GPU resize if OpenCV built with CUDA (optional)
try:
    HAS_CUDA = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    HAS_CUDA = False

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

# Optional translation (pip install googletrans==4.0.0-rc1; may conflict with deepgram's httpcore)
try:
    from googletrans import Translator
    _translator = Translator()
    HAS_TRANSLATE = True
except (ImportError, AttributeError):
    _translator = None
    HAS_TRANSLATE = False

# --- Config (overridden by config.json and CLI) ---
CAMERA_INDEX = 0
CAPTION_FONT_SIZE = 52
CAPTION_MAX_WIDTH = 620
CAPTION_OFFSET_ABOVE_HEAD = 55
MAX_CAPTION_LEN = 220
CAPTION_TIMEOUT_SEC = 4.5
EMOTION_SMOOTH = 0.3
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
CAMERA_RESOLUTIONS = [(1280, 720), (1920, 1080), (960, 540), (0, 0)]
DISPLAY_SIZE = (1280, 720)
FACE_DETECT_SCALE = 0.28
FACE_DETECT_EVERY_N = 6
FACE_SMOOTH = 0.18
MAX_CAPTION_LINES = 2
REVEAL_CHARS_PER_FRAME = 999
CAPTION_BG_COLOR = (45, 42, 34, 230)
CAPTION_BG_PADDING = 12
CAPTION_BORDER_LIGHT = (90, 85, 72, 255)
CAPTION_BORDER_DARK = (20, 18, 15, 255)
CAPTION_BORDER_PX = 2
CAPTION_SCALE_MIN = 0.55
CAPTION_SCALE_MAX = 1.85
EMOTIONS = {
    "happy": ("😊", "yellow"),
    "sad": ("😢", "blue"),
    "surprised": ("😲", "orange"),
    "angry": ("😠", "red"),
    "neutral": ("😐", "white"),
}
EMOTION_COLORS = {
    "yellow": (45, 45, 15, 220),
    "blue": (15, 25, 45, 220),
    "orange": (50, 35, 15, 220),
    "red": (50, 20, 20, 220),
    "white": (35, 35, 35, 200),
}
SHOW_FACE_BOX = False
SPEECH_BUBBLE_TAIL = True
CAPTION_HISTORY_LINES = 2
FADE_IN_FRAMES = 1
EMOTION_HOLD_FRAMES = 3
TRANSLATION_DEST = "es"


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
    """Render caption text in up to 2 lines with emotion-tinted box and optional tail."""
    if not text or not HAS_PIL:
        return None
    font = get_minecraft_font(font_size)
    use_max_width = max_width if max_width is not None else CAPTION_MAX_WIDTH
    pad = padding if padding is not None else CAPTION_BG_PADDING
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
    """Blend caption onto frame at (x, y). Optimized for speed."""
    if caption_rgba is None or caption_rgba.size == 0:
        return frame
    h, w = caption_rgba.shape[:2]
    x = max(0, min(x, frame.shape[1] - w))
    y = max(0, min(y, frame.shape[0] - h))
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame
    roi = frame[y : y + h, x : x + w]
    if alpha_mult >= 0.99:
        alpha = caption_rgba[:, :, 3:4].astype(np.float32) / 255.0
    else:
        alpha = (caption_rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mult
    rgb = caption_rgba[:, :, :3]
    roi[:] = (alpha * rgb + (1.0 - alpha) * roi).astype(np.uint8)
    return frame


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Combined sharpening + brightness/contrast in one pass for speed."""
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    sharpened = cv2.addWeighted(frame, 1.2, blurred, -0.2, 0)
    return cv2.convertScaleAbs(sharpened, alpha=1.03, beta=3)


class FaceKalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.003
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.25
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.25
        self.initialized = False
        self.frames_lost = 0

    def update(self, x=None, y=None):
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
    def __init__(self):
        self.smoothed = "neutral"

    def update(self, _landmarks=None) -> str:
        return self.smoothed


class FrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.frames_captured = 0
        self.frames_dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            with self.lock:
                if self.latest is not None:
                    self.frames_dropped += 1
                self.latest = frame
                self.frames_captured += 1

    def start(self):
        self._thread.start()

    def read(self):
        with self.lock:
            if self.latest is not None:
                try:
                    return self.latest.copy()
                except Exception:
                    return None
        return None

    def stop(self):
        self.running = False
        self._thread.join(timeout=1.0)

    def get_stats(self):
        with self.lock:
            return self.frames_captured, self.frames_dropped


def get_script_dir():
    """Directory containing this script (for config, OBS state, etc.)."""
    return os.path.dirname(os.path.abspath(__file__))


def load_obs_window_state():
    """Load OBS window position/size and caption offsets from obs_window.json. Returns dict or None."""
    path = os.path.join(get_script_dir(), "obs_window.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_obs_window_state(window_name, caption_offset_x, caption_offset_y, window_size):
    """Save OBS window position (if available), size, and caption offsets to obs_window.json."""
    out = {
        "caption_offset_x": caption_offset_x,
        "caption_offset_y": caption_offset_y,
        "window_width": window_size[0],
        "window_height": window_size[1],
    }
    try:
        WND_X = getattr(cv2, "WND_PROP_POS_X", -1)
        WND_Y = getattr(cv2, "WND_PROP_POS_Y", -1)
        if WND_X >= 0 and WND_Y >= 0:
            x = int(cv2.getWindowProperty(window_name, WND_X))
            y = int(cv2.getWindowProperty(window_name, WND_Y))
            if x >= 0 and y >= 0:
                out["window_x"] = x
                out["window_y"] = y
    except Exception:
        pass
    path = os.path.join(get_script_dir(), "obs_window.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass


def draw_perf_overlay(frame, fps, face_detected, stt_active):
    """Draw a small performance panel (FPS, face, STT) on the frame. Modifies frame in place."""
    h, w = frame.shape[:2]
    pad = 8
    line_h = 22
    panel_w = 160
    panel_h = pad * 2 + line_h * 3
    x0, y0 = pad, pad
    overlay = frame[y0 : y0 + panel_h, x0 : x0 + panel_w]
    if overlay.size == 0:
        return
    # Semi-transparent dark background
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (100, 100, 100), 1)
    color_fps = (0, 255, 0) if fps >= 24 else (0, 200, 255) if fps >= 18 else (0, 0, 255)
    cv2.putText(frame, "FPS: {}".format(fps), (x0 + pad, y0 + pad + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_fps, 1, cv2.LINE_AA)
    face_txt = "Face: yes" if face_detected else "Face: no"
    cv2.putText(frame, face_txt, (x0 + pad, y0 + pad + 16 + line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    stt_txt = "STT: on" if stt_active else "STT: off"
    cv2.putText(frame, stt_txt, (x0 + pad, y0 + pad + 16 + line_h * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def _run_ws_server(port, command_queue):
    """Run WebSocket server in a thread; push received JSON commands into command_queue."""
    if not HAS_WEBSOCKETS:
        return

    async def handler(websocket, path):
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    command_queue.put(msg)
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass

    async def serve():
        async with websockets.serve(handler, "127.0.0.1", port, ping_interval=None):
            await asyncio.Future()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(serve())
    except Exception:
        pass
    finally:
        loop.close()


def start_control_api(port=8765, command_queue=None):
    """Start WebSocket control API in a daemon thread. Commands are put into command_queue."""
    if not HAS_WEBSOCKETS or command_queue is None:
        return None
    t = threading.Thread(target=_run_ws_server, args=(port, command_queue), daemon=True)
    t.start()
    return t


def load_config():
    """Load config.json; create with defaults if missing."""
    script_dir = get_script_dir()
    config_path = os.path.join(script_dir, "config.json")
    defaults = {
        "camera_index": 0,
        "caption_font_size": 52,
        "caption_max_width": 620,
        "obs_mode_default_size": "1280x720",
        "show_face_box": False,
        "speech_bubble_tail": True,
        "caption_history_lines": 2,
        "chroma_color": "green",
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return {**defaults, **json.load(f)}
        except Exception:
            pass
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(defaults, f, indent=2)
    except Exception:
        pass
    return defaults


def find_camera():
    """Try indices 0-4 and return the first working camera."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for idx in range(5):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return idx
    return 0


def show_obs_setup_wizard(window_name):
    """Print OBS setup steps once per install."""
    print("\n" + "=" * 60)
    print("OBS SETUP")
    print("=" * 60)
    print("\n1. Open OBS Studio")
    print("2. Add source: [+] in Sources -> Window Capture")
    print("   Name: Face Captions")
    print("   Window: " + window_name)
    print("\n3. Remove green: Right-click source -> Filters -> + -> Chroma Key")
    print("   Key Color: Green  |  Similarity: 400-500  |  Smoothness: 80-100")
    print("\n4. Position the source in your scene, then start speaking.")
    print("=" * 60)


def main():
    global HAS_CUDA
    parser = argparse.ArgumentParser(description="Face-following captions (webcam + STT)")
    parser.add_argument(
        "--obs-mode",
        action="store_true",
        help="OBS capture mode: green screen background, always-on-top. Use OBS Window Capture + Chroma Key.",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default="1280x720",
        help="Window size in OBS mode (e.g. 800x600, 1280x720).",
    )
    parser.add_argument(
        "--chroma-color",
        type=str,
        default=None,
        choices=["green", "blue", "magenta"],
        help="Background color for chroma keying in OBS mode.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera device index (overrides config).",
    )
    parser.add_argument(
        "--caption-offset-x",
        type=int,
        default=0,
        help="Horizontal caption offset in pixels.",
    )
    parser.add_argument(
        "--caption-offset-y",
        type=int,
        default=0,
        help="Vertical caption offset in pixels (negative = up).",
    )
    parser.add_argument(
        "--performance-mode",
        choices=["auto", "high", "balanced", "low"],
        default="balanced",
        help="Performance preset (low = smaller resolution, less detection).",
    )
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Start WebSocket control API on 127.0.0.1 (requires: pip install websockets).",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8765,
        help="Port for WebSocket control API (default 8765).",
    )
    parser.add_argument(
        "--show-perf",
        action="store_true",
        help="Show performance overlay (FPS / face / STT) on the video frame.",
    )
    args = parser.parse_args()
    obs_mode = args.obs_mode
    caption_offset_x = args.caption_offset_x
    caption_offset_y = args.caption_offset_y
    config = load_config()
    chroma_color = args.chroma_color or config.get("chroma_color", "green")
    if "x" in args.window_size:
        try:
            ws, hs = args.window_size.strip().lower().split("x")
            OBS_WINDOW_SIZE = (int(ws), int(hs))
        except ValueError:
            OBS_WINDOW_SIZE = DISPLAY_SIZE
    else:
        OBS_WINDOW_SIZE = DISPLAY_SIZE
    CHROMA_BGR = {"green": (0, 255, 0), "blue": (255, 0, 0), "magenta": (255, 0, 255)}
    obs_chroma_bgr = CHROMA_BGR.get(chroma_color, (0, 255, 0))

    if args.camera_index is not None:
        camera_index = args.camera_index
    else:
        cfg_cam = config.get("camera_index", 0)
        camera_index = find_camera() if cfg_cam == "auto" else int(cfg_cam)

    PERFORMANCE_PRESETS = {
        "high": {"display_size": (1920, 1080), "detect_interval_tracking": 8, "caption_cache_size": 20, "enhance_frame": True},
        "balanced": {"display_size": (1280, 720), "detect_interval_tracking": 8, "caption_cache_size": 20, "enhance_frame": True},
        "low": {"display_size": (960, 540), "detect_interval_tracking": 12, "caption_cache_size": 10, "enhance_frame": False},
    }
    perf_mode = args.performance_mode if args.performance_mode != "auto" else "balanced"
    perf = PERFORMANCE_PRESETS.get(perf_mode, PERFORMANCE_PRESETS["balanced"])
    disp_w_cfg, disp_h_cfg = perf["display_size"]
    detect_interval_tracking = perf["detect_interval_tracking"]
    caption_cache_size = perf["caption_cache_size"]
    enhance_frame_enabled = perf["enhance_frame"]

    if not HAS_PIL:
        print("Install Pillow: pip install Pillow")
        sys.exit(1)

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Could not load face cascade.")
        sys.exit(1)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index {}).".format(camera_index))
        print("\nTroubleshooting:")
        print("  1. Close other apps using the camera (Zoom, Teams, etc.)")
        print("  2. Check camera permissions in System Settings")
        print("  3. Try: python face_captions.py --camera-index 0")
        print("  4. Try: python face_captions.py --camera-index 1")
        print("\nAvailable camera indices:")
        for i in range(5):
            test = cv2.VideoCapture(i, backend)
            if test.isOpened():
                test.release()
                print("  - Camera {}: available".format(i))
            else:
                print("  - Camera {}: not available".format(i))
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if sys.platform == "win32":
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass
    for rw, rh in CAMERA_RESOLUTIONS:
        if rw and rh:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w >= 640 and actual_h >= 480:
            print("Camera: {}x{} @ 30fps (index {})".format(actual_w, actual_h, camera_index))
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
    last_final_translated = ""
    emotion_hold_prev = "neutral"
    emotion_hold_frames = 0
    window_name = "Face captions (Q=quit B=box T=tail H=history F=fade C=color M=translate +=/-=size)"
    caption_cache = OrderedDict()
    MAX_CAPTION_CACHE_SIZE = caption_cache_size
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
    debug_mode = False
    last_debug_time = time.time()

    print("Performance: {} ({}x{})".format(perf_mode, disp_w_cfg, disp_h_cfg))
    print("Caption size: + / - keys (0=100% 1=min 9=max, D=debug)")
    face_landmarker_mp = None
    if HAS_FACE_MESH and create_face_landmarker:
        face_landmarker_mp = create_face_landmarker()
        if face_landmarker_mp:
            print("Face: MediaPipe Face Mesh (emotion from expression)")
        else:
            print("Face: MediaPipe model not found; run download_face_landmarker_model.py")
    if not face_landmarker_mp:
        print("Face: OpenCV Haar cascade")

    api_command_queue = queue.Queue() if args.enable_api else None
    if args.enable_api:
        if HAS_WEBSOCKETS:
            start_control_api(port=args.api_port, command_queue=api_command_queue)
            print("Control API: ws://127.0.0.1:{} (JSON: action, toggle_speech_bubble, set_caption_offset, etc.)".format(args.api_port))
        else:
            print("Control API: install websockets (pip install websockets) to enable.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        if sys.platform == "win32":
            cv2.setWindowProperty(window_name, cv2.WND_PROP_OPENGL, cv2.WINDOW_OPENGL)
    except Exception:
        pass
    if obs_mode:
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        obs_state = load_obs_window_state()
        if obs_state:
            try:
                x, y = obs_state.get("window_x"), obs_state.get("window_y")
                if x is not None and y is not None and x >= 0 and y >= 0:
                    cv2.moveWindow(window_name, int(x), int(y))
                ww, wh = obs_state.get("window_width"), obs_state.get("window_height")
                if ww and wh:
                    cv2.resizeWindow(window_name, int(ww), int(wh))
                if "caption_offset_x" in obs_state:
                    caption_offset_x = int(obs_state["caption_offset_x"])
                if "caption_offset_y" in obs_state:
                    caption_offset_y = int(obs_state["caption_offset_y"])
            except Exception:
                pass
        print("OBS mode: {}x{} ({} chroma). In OBS: Window Capture -> Chroma Key.".format(
            OBS_WINDOW_SIZE[0], OBS_WINDOW_SIZE[1], chroma_color))

    disp_w, disp_h = (disp_w_cfg, disp_h_cfg)
    if obs_mode:
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".obs_setup_done")):
            show_obs_setup_wizard(window_name)
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".obs_setup_done"), "w") as _:
                    _.write("1")
            except Exception:
                pass
    while True:
        if api_command_queue is not None:
            while not api_command_queue.empty():
                try:
                    cmd = api_command_queue.get_nowait()
                    action = (cmd or {}).get("action")
                    if action == "toggle_speech_bubble":
                        show_speech_tail = not show_speech_tail
                    elif action == "set_caption_offset":
                        if "x" in cmd:
                            caption_offset_x = int(cmd["x"])
                        if "y" in cmd:
                            caption_offset_y = int(cmd["y"])
                    elif action == "set_theme" and "theme" in cmd:
                        pass  # placeholder for future theme support
                except (queue.Empty, TypeError, ValueError, KeyError):
                    break
        frame = frame_reader.read()
        if frame is None:
            time.sleep(0.001)
            continue
        frame_time_start = time.time()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if w > disp_w or h > disp_h:
            if HAS_CUDA:
                try:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_frame = cv2.cuda.resize(gpu_frame, (disp_w, disp_h))
                    frame = gpu_frame.download()
                except Exception as e:
                    if frame_count < 10:
                        print("GPU resize failed: {}, using CPU".format(e))
                    HAS_CUDA = False
                    frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            else:
                frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            h, w = disp_h, disp_w
        if face_center_x == 0 and face_top_y == 0:
            face_center_x, face_top_y = w / 2, h / 3

        frame_count += 1
        timestamp_ms = int(frame_count * 1000 / 30)
        rgb_frame = None
        if last_face_bbox is None:
            detect_interval = 1
            detect_scale = 0.40
        else:
            detect_interval = detect_interval_tracking
            detect_scale = 0.20

        if face_landmarker_mp and (frame_count % detect_interval == 0):
            if rgb_frame is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_mp, landmarks_mp, blendshapes_mp = detect_face(face_landmarker_mp, rgb_frame, timestamp_ms)
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
            max_items = 10
            items_processed = 0
            new_final = False
            while items_processed < max_items:
                item = text_queue.get_nowait()
                text, is_final = item if isinstance(item, tuple) and len(item) == 2 else (item, True)
                text = (text or "").strip()
                if not text:
                    items_processed += 1
                    continue
                if is_final:
                    last_final = text
                    live_partial = ""
                    new_final = True
                    if caption_history_lines > 0:
                        now = time.time()
                        caption_history.append((text, now))
                        if len(caption_history) > caption_history_lines:
                            caption_history.pop(0)
                else:
                    if not new_final:
                        live_partial = text
                items_processed += 1
        except queue.Empty:
            pass
        current_caption = live_partial if live_partial else last_final
        current_caption = re.sub(r"\s*\[\s*\]\s*", " ", current_caption)
        current_caption = re.sub(r"[\u25A0-\u25AB\u25FB\u25FC\uFFFD]", "", current_caption)  # □ ■ ▢ etc.
        current_caption = re.sub(r"\s+", " ", current_caption).strip()
        if len(current_caption) > MAX_CAPTION_LEN:
            current_caption = current_caption[-MAX_CAPTION_LEN:].strip()

        target_len = len(current_caption)
        if target_len < reveal_len:
            reveal_len = target_len
        else:
            reveal_len = min(reveal_len + REVEAL_CHARS_PER_FRAME, target_len)
        displayed_caption = current_caption[:reveal_len]

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

        now = time.time()
        if caption_history_lines > 0 and caption_history:
            caption_history[:] = [(t, ts) for t, ts in caption_history if now - ts < CAPTION_TIMEOUT_SEC]
            MAX_HISTORY_ENTRIES = 10
            if len(caption_history) > MAX_HISTORY_ENTRIES:
                caption_history[:] = caption_history[-MAX_HISTORY_ENTRIES:]
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
        scale_changed = abs(scale_key - last_caption_scale) > 0.01
        if display_text != last_caption_text or last_caption_render is None or scale_changed:
            frames_since_caption_change = 0
            last_caption_text = display_text
            last_caption_scale = scale_key
            text_hash = hash(display_text[:50]) if len(display_text) > 30 else display_text
            cache_key = (emotion, show_speech_tail, scale_key, text_hash)
            if cache_key in caption_cache and caption_cache[cache_key][1] == display_text:
                last_caption_render = caption_cache[cache_key][0]
                caption_cache.move_to_end(cache_key)
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
                        caption_cache.popitem(last=False)
                    caption_cache[cache_key] = (last_caption_render, display_text)
        else:
            frames_since_caption_change += 1

        alpha_mult = 1.0
        if fade_in_frames > 0 and frames_since_caption_change < fade_in_frames:
            alpha_mult = 0.5 + 0.5 * (frames_since_caption_change / fade_in_frames)

        if not obs_mode and enhance_frame_enabled:
            if apply_color_filter:
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            else:
                frame = enhance_frame(frame)

        lookahead_frames = 2 if face_bbox is not None else 0
        if lookahead_frames and face_tracker.initialized and hasattr(face_tracker.kf, "statePost"):
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
            caption_y = max(0, caption_y) + caption_offset_y
            caption_x = max(0, min(cx - cw // 2, w - cw)) + caption_offset_x
            frame = overlay_caption_on_frame(
                frame, last_caption_render, caption_x, caption_y, alpha_mult=alpha_mult
            )
            last_caption_x, last_caption_y, last_caption_w, last_caption_h = caption_x, caption_y, cw, ch
        else:
            last_caption_x, last_caption_y, last_caption_w, last_caption_h = None, None, None, None
        if show_face_box and face_bbox and not obs_mode:
            x, y, bw, bh = face_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

        frame_time_elapsed = time.time() - frame_time_start
        if frame_time_elapsed > (1.0 / 30) * 1.5:
            frame_count += 1
        if obs_mode:
            w_obs, h_obs = OBS_WINDOW_SIZE
            display_frame = np.full((h_obs, w_obs, 3), obs_chroma_bgr, dtype=np.uint8)
            if debug_mode:
                for i in range(0, w_obs, 100):
                    cv2.line(display_frame, (i, 0), (i, h_obs), tuple(max(0, c - 55) for c in obs_chroma_bgr), 1)
                for i in range(0, h_obs, 100):
                    cv2.line(display_frame, (0, i), (w_obs, i), tuple(max(0, c - 55) for c in obs_chroma_bgr), 1)
            if last_caption_render is not None and last_caption_x is not None:
                cw, ch = last_caption_w, last_caption_h
                if w_obs != w or h_obs != h:
                    sx, sy = w_obs / max(1, w), h_obs / max(1, h)
                    cx_obs = int(last_caption_x * sx)
                    cy_obs = int(last_caption_y * sy)
                    cw_obs = int(cw * sx)
                    ch_obs = int(ch * sy)
                    caption_scaled = cv2.resize(last_caption_render, (max(1, cw_obs), max(1, ch_obs)), interpolation=cv2.INTER_LINEAR)
                else:
                    cx_obs, cy_obs = last_caption_x, last_caption_y
                    caption_scaled = last_caption_render
                    cw_obs, ch_obs = cw, ch
                cx_obs = max(0, min(cx_obs, w_obs - cw_obs))
                cy_obs = max(0, min(cy_obs, h_obs - ch_obs))
                display_frame = overlay_caption_on_frame(
                    display_frame, caption_scaled, cx_obs, cy_obs, alpha_mult=alpha_mult
                )
            cv2.imshow(window_name, display_frame)
        else:
            cv2.imshow(window_name, frame)
        current_time = time.time()
        fps_history.append(current_time)
        fps_history = [t for t in fps_history if current_time - t < 1.0]
        fps = len(fps_history)
        if (debug_mode or args.show_perf) and not obs_mode:
            draw_perf_overlay(frame, fps, last_face_bbox is not None, stt is not None)
        if debug_mode and (current_time - last_debug_time > 1.0):
            captured, dropped = frame_reader.get_stats()
            drop_rate = (dropped / captured * 100) if captured > 0 else 0
            print("[DEBUG] FPS: {:2d} | Interval: {} | Cache: {:2d}/{} | Frame: {:4.1f}ms | Dropped: {:4.1f}%".format(
                fps, detect_interval, len(caption_cache), MAX_CAPTION_CACHE_SIZE,
                frame_time_elapsed * 1000, drop_rate))
            last_debug_time = current_time
        if fps < 20 and frame_count > 100 and frame_count % 300 == 0:
            print("WARNING: Low FPS ({}). Try: close other apps, reduce camera resolution, or press F to disable fade.".format(fps))
        if current_time - last_fps_time > 2.0:
            cv2.setWindowTitle(window_name, "Face Captions — {0} FPS (Q=quit +=/-=size)".format(fps))
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
        elif key in (ord("+"), ord("=")):
            caption_scale = min(CAPTION_SCALE_MAX, caption_scale * 1.12)
            caption_scale = round(caption_scale, 2)
            print("Caption size: {:.0%}".format(caption_scale))
        elif key == ord("-"):
            caption_scale = max(CAPTION_SCALE_MIN, caption_scale * 0.88)
            caption_scale = round(caption_scale, 2)
            print("Caption size: {:.0%}".format(caption_scale))
        elif key == ord("0"):
            caption_scale = 1.0
            print("Caption size: reset to 100%")
        elif key == ord("9"):
            caption_scale = CAPTION_SCALE_MAX
            print("Caption size: maximum ({:.0%})".format(CAPTION_SCALE_MAX))
        elif key == ord("1"):
            caption_scale = CAPTION_SCALE_MIN
            print("Caption size: minimum ({:.0%})".format(CAPTION_SCALE_MIN))
        el        if key == ord("d"):
            debug_mode = not debug_mode
            print("Debug mode: {}".format("ON" if debug_mode else "OFF"))

    if obs_mode:
        save_obs_window_state(window_name, caption_offset_x, caption_offset_y, OBS_WINDOW_SIZE)
    if stt:
        stt.stop()
    frame_reader.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
