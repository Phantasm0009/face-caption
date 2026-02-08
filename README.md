# Face-following captions

A **Python desktop app** (no browser) that shows your webcam at **full camera quality** with subtitles anchored to your face. Captions follow your head, use a Minecraft-style font, and update in **real time** as you speak (streaming when possible).

## Features

- **Full camera quality** – Uses the camera’s native resolution (no forced downscale).
- **Face-following captions** – Text is drawn above your face and moves with you.
- **Real-time speech** – Project library `realtime_stt.py`: **streaming** captions (word-by-word as you speak) when Vosk is used; otherwise fast short-chunk recognition.
- **Minecraft font** – Place `Minecraft.ttf` in this folder or in a `fonts` subfolder.
- **Emotion from your face** – When using MediaPipe Face Mesh, the caption box color and mood are driven by your expression (smile → happy, frown → sad, etc.).
- **Face tracking** – Uses **MediaPipe Face Landmarker** (face mesh) if the model is present for stable head tracking and emotion; otherwise falls back to OpenCV’s Haar cascade.

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

4. **Face mesh + emotion (optional):**  
   For better face tracking and **real emotion detection** (smile/frown → caption color), download the MediaPipe Face Landmarker model:

   ```bash
   python download_face_landmarker_model.py
   ```

   Then place the `.task` file in the `models/` folder (the script does this if the download succeeds). If the model is missing, the app uses OpenCV’s built-in face detector and neutral emotion.

5. **Minecraft font (optional):**  
   Put `Minecraft.ttf` in this folder or in `fonts/`. Otherwise a system font is used.

6. **Microphone** – Allow the app to use your default mic.

## Run

```bash
python face_captions.py
```

- **Q** – Quit. **B** – Toggle face bounding box.

## Notes

- **Caption mode:** On start the app prints either “streaming (real-time)” (Vosk model present) or “fast batch” (no model). For real-time, run `download_vosk_model.py` once.
- **Face mode:** If you see “Face: MediaPipe Face Mesh (emotion from expression)”, the app is using the face landmarker model and emotion is driven by your expression. Otherwise it uses OpenCV Haar and neutral emotion; run `download_face_landmarker_model.py` to get the model.
- **Camera:** To use a different device, set `CAMERA_INDEX` at the top of `face_captions.py` (e.g. `1`).
