# OBS Setup Guide

## Quick Start (2 minutes)

### 1. Run Face Captions in OBS Mode

```bash
python face_captions.py --obs-mode
```

You'll see:

- A window showing **your camera + captions** (one window, one camera)
- Captions floating above your face
- Window always stays on top

### 2. Add to OBS

**Only the app uses your camera** — use a single source in OBS:

1. Open OBS Studio
2. Click **+** in Sources → **Window Capture**
3. Name it "Face Captions" and select the Python window: "Face captions (Q=quit...)" or similar.
4. **Position** the source in the OBS preview (drag/resize as needed).

You’ll see: your **face + captions** in one shot. No second camera and no Chroma Key needed.

### 3. Start Streaming

You're done. The captions:

- Track your face in real time
- Show your speech as you talk
- Use a Minecraft-style look
- Work with any OBS scene

---

## Advanced Options

### Window size (default 1280×720)

```bash
python face_captions.py --obs-mode --window-size 800x600
```

### Hotkeys (while the app window is focused)

- **H** – Toggle caption history (1 vs 2 lines)
- **T** – Toggle speech bubble tail
- **+ / -** – Adjust caption size
- **0** – Reset size to 100%
- **1** – Minimum size
- **9** – Maximum size
- **D** – Toggle debug (FPS, cache, frame time; in OBS mode also shows a grid on the green screen)

---

## Troubleshooting

| Issue | Fix |
|--------|-----|
| Caption is cut off | Use a larger window: `--window-size 1920x1080` |
| Captions lag | Close other apps; press **D** to check FPS |
| No speech detected | Check mic permissions and that the app can use the microphone |
| Black or no video in OBS | Make sure the Face Captions window is not minimized; select the correct window in Window Capture (e.g. "Face captions (Q=quit...)"). |

---

## Testing Checklist

1. **Basic:** Run `python face_captions.py --obs-mode` → window shows your face + caption "..."
2. **Face tracking:** Move left/right/up/down → caption follows
3. **Speech:** Speak → caption appears; stop → caption fades after a few seconds
4. **OBS:** Window Capture shows your face + captions in one source
5. **Performance:** Press **D** → FPS around 28–30, frame time &lt; 30 ms
