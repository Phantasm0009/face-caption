# Face-following captions

A **Python desktop app** (no browser) that shows your webcam at **full camera quality** with subtitles anchored to your face. Captions follow your head, use a Minecraft-style font, and update in **real time** as you speak (streaming when possible).

## Features

- **Full camera quality** – Uses the camera’s native resolution (no forced downscale).
- **Face-following captions** – Text is drawn above your face and moves with you.
- **Real-time speech** – Project library `realtime_stt.py`: **streaming** captions (word-by-word as you speak) when Vosk is used; otherwise fast short-chunk recognition.
- **Minecraft font** – Place `Minecraft.ttf` in this folder or in a `fonts` subfolder.
- **Emotion emoji** – Optional emoji next to the caption.

## Setup

1. **Python 3.8+** and pip.

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   On Windows, if `PyAudio` fails:

   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

3. **Real-time captions (recommended):**  
   For **no-delay, live** text as you speak, use Vosk (offline, streaming). One-time download:

   ```bash
   python download_vosk_model.py
   ```

   This downloads the small English model (~40 MB). For **better word recognition**, use the large model (~1.8 GB):

   ```bash
   python download_vosk_model.py --large
   ```

   Without any model, the app uses Google Speech (online) in fast batch mode.

4. **Minecraft font (optional):**  
   Put `Minecraft.ttf` in this folder or in `fonts/`. Otherwise a system font is used.

5. **Microphone** – Allow the app to use your default mic.

## Run

```bash
python face_captions.py
```

- **Q** – Quit.

## Notes

- **Caption mode:** On start the app prints either “streaming (real-time)” (Vosk model present) or “fast batch” (no model). For real-time, run `download_vosk_model.py` once.
- **Camera:** To use a different device, set `CAMERA_INDEX` at the top of `face_captions.py` (e.g. `1`).
