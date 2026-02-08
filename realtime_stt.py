"""
Real-time speech-to-text module (project library).
- Prefers Vosk streaming for live partial + final results (no delay feel).
- Falls back to fast batch recognition using existing libs only.
"""

import os
import sys
import json
import threading
import queue
import time

# Optional: Vosk for true streaming (partial results as you speak)
try:
    import vosk
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

# Results: (text, is_final). is_final=True => final phrase; False => partial (live)
RESULT_FINAL = True
RESULT_PARTIAL = False

# Vosk model: prefer larger model first (better word recognition)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(SCRIPT_DIR, "models"),
    os.path.join(SCRIPT_DIR, "vosk-model"),
    os.path.join(os.path.expanduser("~"), ".vosk", "models"),
]
MODEL_NAMES = ["vosk-model-en-us-0.22", "vosk-model-small-en-us-0.15", "model"]


def _find_vosk_model():
    for d in MODEL_DIRS:
        for name in MODEL_NAMES:
            path = os.path.join(d, name)
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "conf", "model.conf")):
                return path
    return None


class StreamingSTT:
    """
    Pushes (text, is_final) to result_queue.
    - With Vosk: partial results while speaking, then final when phrase ends.
    - Without Vosk: short chunks, final only (fast batch).
    """

    def __init__(self, result_queue: queue.Queue, sample_rate: int = 16000):
        self.result_queue = result_queue
        self.sample_rate = sample_rate
        self.running = False
        self._thread = None
        self._use_vosk = False
        self._model = None

    def start(self) -> bool:
        if HAS_VOSK and HAS_PYAUDIO:
            model_path = _find_vosk_model()
            if model_path:
                try:
                    self._model = vosk.Model(model_path)
                    self._use_vosk = True
                except Exception:
                    pass
        if self._use_vosk:
            self._thread = threading.Thread(target=self._run_vosk, daemon=True)
        elif HAS_SR and HAS_PYAUDIO:
            self._thread = threading.Thread(target=self._run_batch, daemon=True)
        else:
            return False
        self.running = True
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_vosk(self):
        """Streaming: push partial and final results. Larger chunks = better word boundaries."""
        rec = vosk.KaldiRecognizer(self._model, self.sample_rate)
        rec.SetWords(False)
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=8000,
            )
        except Exception:
            return
        # Small chunks = partials very often = minimal delay (best for real-time)
        chunk_samples = 1200
        while self.running and stream.is_active():
            try:
                data = stream.read(chunk_samples, exception_on_overflow=False)
                if not data:
                    continue
                if rec.AcceptWaveform(data):
                    j = json.loads(rec.Result())
                    text = (j.get("text") or "").strip()
                    if text:
                        self.result_queue.put((text, RESULT_FINAL))
                else:
                    j = json.loads(rec.PartialResult())
                    partial = (j.get("partial") or "").strip()
                    if partial:
                        self.result_queue.put((partial, RESULT_PARTIAL))
            except Exception:
                time.sleep(0.05)
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

    def _run_batch(self):
        """Batch mode: language hint + ambient calibration for better word recognition."""
        recognizer = sr.Recognizer()
        chunk_sec = 0.4  # Slightly longer = more context for Google, better accuracy
        mic = None
        try:
            mic = sr.Microphone(sample_rate=16000)
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception:
            return
        while self.running and mic:
            try:
                with mic as source:
                    audio = recognizer.record(source, duration=chunk_sec)
                if not self.running:
                    break
                try:
                    text = recognizer.recognize_google(audio, language="en-US")
                    if text:
                        self.result_queue.put((text.strip(), RESULT_FINAL))
                except (sr.UnknownValueError, sr.RequestError):
                    pass
            except Exception:
                time.sleep(0.05)


def get_caption_mode():
    """Return 'streaming' if Vosk available, else 'batch'."""
    if HAS_VOSK and _find_vosk_model():
        return "streaming"
    return "batch"
