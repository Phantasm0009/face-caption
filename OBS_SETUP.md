# OBS Setup Guide

## Quick Start (2 minutes)

### 1. Run Face Captions in OBS Mode

```bash
python face_captions.py --obs-mode
```

You'll see:

- Green screen window
- Captions floating where your face would be
- Window always stays on top

### 2. Add to OBS

1. Open OBS Studio
2. **Add Source:** Click **+** in Sources → **Window Capture**
3. Name it "Face Captions"
4. Select the Python window: "Face captions..."
5. **Remove Green Screen:** Right-click the source → **Filters** → **+** → **Chroma Key**
   - Color: **Green**
   - Similarity: 400–500
   - Smoothness: 80–100
6. **Position** the source in the OBS preview (drag/resize as needed)

Captions will follow your face automatically.

### 3. Start Streaming

You're done. The captions:

- Track your face in real time
- Show your speech as you talk
- Use a Minecraft-style look
- Work with any OBS scene

---

## Advanced Options

### Smaller window

```bash
python face_captions.py --obs-mode --window-size 800x600
```

### Different chroma color (blue or magenta)

```bash
python face_captions.py --obs-mode --chroma-color blue
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
| Green shows through caption | Increase Chroma Key **Similarity** in OBS |
| Caption is cut off | Use a larger window: `--window-size 1920x1080` |
| Captions lag | Close other apps; press **D** to check FPS |
| No speech detected | Check mic permissions and that the app can use the microphone |
| Camera not in OBS | Don’t add the camera in OBS. Use **Window Capture** on the face_captions window so the camera is only used by the app. |

---

## Testing Checklist

1. **Basic:** Run `python face_captions.py --obs-mode` → green window, caption "..."
2. **Face tracking:** Move left/right/up/down → caption follows
3. **Speech:** Speak → caption appears; stop → caption fades after a few seconds
4. **OBS:** Window Capture finds the window; Chroma Key removes green; caption is readable
5. **Performance:** Press **D** → FPS around 28–30, frame time &lt; 30 ms
