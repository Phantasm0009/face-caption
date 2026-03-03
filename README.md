# Face-Following Captions

Real-time speech captions that follow your face. A Python app using your webcam, face tracking, and streaming speech-to-text—with **Minecraft** or **Cyberpunk 2077** caption styles and optional emotion-based styling.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Run](#run)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [OBS / Streaming](#obs--streaming)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Description |
|--------|-------------|
| **Face-following** | Captions stay above your head and move smoothly with you (position smoothing). |
| **Real-time STT** | Streaming captions (Deepgram, Vosk, or faster-whisper) or fast batch (Google). |
| **Caption styles** | **Minecraft** or **Cyberpunk 2077**; set `caption_style` in `config.json`. Use `cyberpunk_font` to pick a font (Rajdhani, Orbitron, etc.). |
| **Emotion colors** | With MediaPipe Face Mesh, caption tint reflects expression (happy, sad, etc.). |
| **Face tracking** | MediaPipe Face Landmarker when available; otherwise OpenCV Haar cascade. |
| **OBS-ready** | `--obs-mode` for green-screen overlay; use Window Capture + Chroma Key in OBS. |

---

## Quick Start

```bash
pip install -r requirements.txt
python face_captions.py
```

- **Q** = Quit. **+ / -** = Caption size. See [Keyboard Shortcuts](#keyboard-shortcuts) for more.

For **streaming captions**, run once: `python download_vosk_model.py` (or set `DEEPGRAM_API_KEY` in `.env` for cloud STT).  
For **face mesh + emotion**, run once: `python download_face_landmarker_model.py`.

---

## Setup

### Requirements

- **Python 3.8+**
- Microphone access for speech-to-text

### Install dependencies

```bash
pip install -r requirements.txt
```

On Windows, if PyAudio fails:

```bash
pip install pipwin
pipwin install pyaudio
```

### Speech-to-text (pick one)

| Option | Command / Config | Notes |
|--------|------------------|--------|
| **Deepgram** (cloud) | Add `DEEPGRAM_API_KEY` to `.env` | Best quality, low latency; needs API key. |
| **Vosk** (offline) | `python download_vosk_model.py` | No API key; use `--large` for better accuracy. |
| **faster-whisper** (offline) | `pip install faster-whisper` | Good accuracy, no extra download script. |
| **Google** (online) | (default if none of the above) | Batch mode; no setup. |

### Face tracking + emotion (optional)

```bash
python download_face_landmarker_model.py
```

Puts the model in `models/`. Without it, the app uses OpenCV’s face detector and neutral emotion.

### Fonts (optional)

- **Minecraft style**: Place `Minecraft.ttf` in the project folder or `fonts/`. Otherwise a system font is used.
- **Cyberpunk style**: Set `cyberpunk_font` in `config.json` (e.g. `"Rajdhani-Regular.ttf"`, `"Orbitron-Regular.ttf"`, `"Cyberpunk-Regular.ttf"`). Place the `.ttf` in the project root or `fonts/`. Use `null` for the default search order.

### Different camera

Set `camera_index` in `config.json` (0, 1, 2…). Run `python list_mics.py` to see microphone indices. For cameras, try different indices (0, 1, 2) until one works.

---

## Run

```bash
python face_captions.py
```

**OBS mode** (green screen for Window Capture + Chroma Key):

```bash
python face_captions.py --obs-mode
python face_captions.py --obs-mode --window-size 800x600
python face_captions.py --obs-mode --chroma-color blue
```

OBS window position and caption offsets are saved to `obs_window.json` on quit and restored on next run.

**Performance mode**: `--performance-mode high|balanced|low` (default: balanced). Use `low` for slower PCs.

**Performance overlay** (FPS, face, STT on frame): `--show-perf` or press **D** for debug.

**WebSocket control API** (e.g. Stream Deck): `pip install websockets` then `python face_captions.py --enable-api` (default port 8765). Send JSON: `{"action": "toggle_speech_bubble"}`, `{"action": "set_caption_offset", "x": 0, "y": -10}`.

See [OBS / Streaming](#obs--streaming) and **OBS_SETUP.md** for full instructions.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **B** | Toggle face bounding box |
| **T** | Toggle speech bubble tail |
| **H** | Toggle caption history (1 vs 2 lines) |
| **F** | Toggle fade-in |
| **C** | Toggle color filter (hot map) |
| **M** | Toggle translation (if googletrans installed) |
| **+ / =** | Increase caption size |
| **-** | Decrease caption size |
| **0** | Reset caption size to 100% |
| **1** | Minimum caption size |
| **9** | Maximum caption size |
| **D** | Toggle debug (FPS, cache, frame time; in OBS mode shows grid) |

---

## OBS / Streaming

Use the app as a **caption overlay** in OBS:

1. Run: `python face_captions.py --obs-mode`
2. In OBS: **Add source → Window Capture** → select the “Face captions” window
The window shows **your camera + captions** in one view. Only the app uses your camera, so you need just one source in OBS (no second camera, no Chroma Key).

Full guide: **[OBS_SETUP.md](OBS_SETUP.md)**

---

## Configuration

Main settings are in **`config.json`** (and some at the top of `face_captions.py`):

| Option | Description |
|--------|-------------|
| **`camera_index`** | Webcam device index (0, 1, 2…). |
| **`mic_input_device_index`** | Microphone for speech-to-text: `null` = default, or a number (run `python list_mics.py` for indices). |
| **`caption_font_size`**, **`caption_max_width`** | Text size and max width. |
| **`caption_style`** | `"minecraft"` (default) or `"cyberpunk"` – Cyberpunk 2077 style: chamfered corners, neon cyan border, scan lines, text glow. |
| **`cyberpunk_font`** | Font filename for Cyberpunk style (e.g. `"Rajdhani-Regular.ttf"`, `"Orbitron-Regular.ttf"`). Place `.ttf` in project root or `fonts/`. `null` = default search. |
| **`chroma_color`** | For OBS mode: `"green"`, `"blue"`, or `"magenta"`. |
| **`show_face_box`** | Draw face bounding box. |
| **`speech_bubble_tail`**, **`caption_history_lines`** | Caption behavior. |
| **`obs_mode_default_size`** | Default window size in OBS mode (e.g. `"1280x720"`). |

In `face_captions.py`: `CAPTION_TIMEOUT_SEC`, `DISPLAY_SIZE`, etc.

Optional `.env` (in project folder):

- `DEEPGRAM_API_KEY` – For Deepgram streaming STT
- `DEEPGRAM_INPUT_DEVICE_INDEX` – Microphone index for Deepgram
- `DEBUG_STT` – Extra STT logs

---

## Project Structure

```
├── face_captions.py       # Main app (webcam, face tracking, captions)
├── face_mesh.py           # MediaPipe face landmarker + emotion
├── realtime_stt.py        # Streaming STT (Deepgram, Vosk, faster-whisper, Google)
├── download_vosk_model.py # One-time Vosk model download
├── download_face_landmarker_model.py  # One-time face model download
├── list_mics.py           # List microphone devices
├── config.json            # Settings (camera, mic, caption style, fonts)
├── obs_window.json       # OBS window position (auto-saved)
├── OBS_SETUP.md           # OBS setup guide
├── requirements.txt
├── .env                   # Optional: DEEPGRAM_API_KEY, etc.
├── models/                # Face (and optional) models
├── fonts/                 # Optional: Minecraft.ttf, Rajdhani-Regular.ttf, etc.
└── *.ttf                  # Optional fonts in project root
```

---

## Troubleshooting

| Issue | What to try |
|--------|--------------|
| **No speech / wrong mic** | Check OS mic permissions; set `DEEPGRAM_INPUT_DEVICE_INDEX` in `.env`; run `list_mics.py` to see device indices. |
| **“Caption mode: fast batch”** | Install Vosk and run `download_vosk_model.py`, or set `DEEPGRAM_API_KEY` in `.env` for streaming. |
| **“Face: OpenCV Haar cascade”** | Run `download_face_landmarker_model.py` and ensure the model is in `models/`. |
| **Camera not opening** | Change `CAMERA_INDEX` (0, 1, 2…); on Windows try closing other apps using the camera. |
| **Camera works here but not in OBS** | Don’t add the camera in OBS. Use **Window Capture** on the face_captions window so only the app uses the camera. |
| **Low FPS** | Press **D** for debug; close other apps; in OBS mode use `--window-size 800x600`. |

---

## License

Use and modify as you like. If you redistribute, keep attribution.
