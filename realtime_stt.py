"""
Real-time speech-to-text module (project library).
- Prefers faster-whisper (best accuracy), then Vosk, then batch Google.
- Optional: Deepgram live API if DEEPGRAM_API_KEY is set.
"""

import os
import sys
import json
import threading
import queue
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional: faster-whisper (better accuracy than Vosk, offline)
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    WhisperModel = None

# Optional: Vosk for streaming (partial results as you speak)
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

# Optional: Deepgram live (best quality; set DEEPGRAM_API_KEY) — SDK 5.x API
try:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
    HAS_DEEPGRAM = True
except ImportError:
    HAS_DEEPGRAM = False
    DeepgramClient = None
    EventType = None

# Results: (text, is_final). is_final=True => final phrase; False => partial (live)
RESULT_FINAL = True
RESULT_PARTIAL = False

# Vosk model: large model first = much better accuracy (run download_vosk_model.py --large)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [
    os.path.join(SCRIPT_DIR, "models"),
    os.path.join(SCRIPT_DIR, "vosk-model"),
    os.path.join(os.getcwd(), "models"),
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
    Prefers: faster-whisper > Vosk > batch Google. Optional: Deepgram if API key set.
    """

    def __init__(self, result_queue: queue.Queue, sample_rate: int = 16000):
        self.result_queue = result_queue
        self.sample_rate = sample_rate
        self.running = False
        self._thread = None
        self._use_faster_whisper = False
        self._use_vosk = False
        self._use_deepgram = False
        self._model = None
        self._engine = "batch"
        self._dg_client = None

    def start(self) -> bool:
        api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
        if api_key and HAS_DEEPGRAM and HAS_PYAUDIO:
            try:
                self._dg_client = DeepgramClient(api_key=api_key)
                self._use_deepgram = True
                self._engine = "deepgram"
                print("STT: Using Deepgram API (live). If no speech appears, check default mic or set DEEPGRAM_INPUT_DEVICE_INDEX.")
            except Exception as e:
                print("STT: Deepgram failed ({0}), using next engine.".format(type(e).__name__))
                if os.environ.get("DEBUG_STT"):
                    print("  ", e)
                self._dg_client = None
        if self._use_deepgram:
            self._thread = threading.Thread(target=self._run_deepgram, daemon=True)
        elif HAS_FASTER_WHISPER and HAS_PYAUDIO and HAS_NUMPY:
            try:
                self._model = WhisperModel("base.en", device="cpu", compute_type="int8")
                self._use_faster_whisper = True
                self._engine = "faster-whisper"
                self._thread = threading.Thread(target=self._run_faster_whisper, daemon=True)
            except Exception as e:
                if os.environ.get("DEBUG_STT"):
                    print("Faster-whisper init failed:", e)
        if not self._thread and HAS_VOSK and HAS_PYAUDIO:
            model_path = _find_vosk_model()
            if model_path:
                try:
                    self._model = vosk.Model(model_path)
                    self._use_vosk = True
                    self._engine = "vosk"
                    self._thread = threading.Thread(target=self._run_vosk, daemon=True)
                except Exception:
                    pass
        if not self._thread:
            if HAS_SR and HAS_PYAUDIO:
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

    def _run_faster_whisper(self):
        """Chunked transcription (1.2s chunks, 75% overlap) for lower latency."""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4800,
            )
        except Exception:
            return
        chunk_duration = 1.2
        samples_per_chunk = int(self.sample_rate * chunk_duration)
        read_size = 2400
        audio_buffer = []
        while self.running and stream.is_active():
            try:
                data = stream.read(read_size, exception_on_overflow=False)
                if not data:
                    continue
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.extend(audio_np)
                if len(audio_buffer) < samples_per_chunk:
                    continue
                chunk = np.array(audio_buffer[:samples_per_chunk], dtype=np.float32)
                audio_buffer = audio_buffer[samples_per_chunk // 4:]
                segments, _ = self._model.transcribe(
                    chunk,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=250,
                        speech_pad_ms=100,
                    ),
                    no_speech_threshold=0.5,
                )
                for segment in segments:
                    text = (segment.text or "").strip()
                    if text:
                        self.result_queue.put((text, RESULT_FINAL))
            except Exception:
                time.sleep(0.01)
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

    def _on_dg_message(self, message):
        """Handle Deepgram SDK 5.x MESSAGE event (ListenV1ResultsEvent)."""
        try:
            # Skip non-Results (e.g. Metadata, UtteranceEnd)
            if not getattr(message, "channel", None):
                return
            ch = message.channel
            if not getattr(ch, "alternatives", None) or not ch.alternatives:
                return
            sentence = (ch.alternatives[0].transcript or "").strip()
            if not sentence:
                return
            is_final = getattr(message, "speech_final", None)
            if is_final is None:
                is_final = getattr(message, "is_final", True)
            self.result_queue.put((sentence, RESULT_FINAL if is_final else RESULT_PARTIAL))
        except Exception:
            pass

    def _run_deepgram(self):
        """Send mic audio to Deepgram live connection (SDK 5.x: connect + socket)."""
        if not self._dg_client:
            return
        p = pyaudio.PyAudio()
        # Optional: set DEEPGRAM_INPUT_DEVICE_INDEX if default mic is wrong (e.g. on Windows)
        input_device = None
        try:
            idx = os.environ.get("DEEPGRAM_INPUT_DEVICE_INDEX")
            if idx is not None:
                input_device = int(idx)
        except ValueError:
            pass
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=4096,
            )
        except Exception as e:
            print("Deepgram: Could not open microphone.", e)
            return
        try:
            # Required for raw PCM: encoding + sample_rate. Keep optional params minimal to avoid HTTP 400.
            with self._dg_client.listen.v1.connect(
                model="nova-2",
                language="en-US",
                encoding="linear16",
                sample_rate=str(self.sample_rate),
                interim_results="true",
            ) as socket_client:
                socket_client.on(EventType.MESSAGE, self._on_dg_message)
                listener_done = threading.Event()
                def run_listener():
                    try:
                        socket_client.start_listening()
                    except Exception:
                        pass
                    finally:
                        listener_done.set()
                t = threading.Thread(target=run_listener, daemon=True)
                t.start()
                try:
                    while self.running and stream.is_active():
                        try:
                            data = stream.read(2048, exception_on_overflow=False)
                            if data:
                                socket_client._send(data)
                        except Exception:
                            pass
                        time.sleep(0.02)
                finally:
                    try:
                        socket_client._websocket.close()
                    except Exception:
                        pass
                    listener_done.wait(timeout=2.0)
        except Exception as e:
            print("Deepgram connection error:", e)
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

    def _run_vosk(self):
        """Streaming: push partials often (instant feel) and finals (accurate phrases)."""
        rec = vosk.KaldiRecognizer(self._model, self.sample_rate)
        rec.SetWords(False)
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4096,
            )
        except Exception:
            return
        # Small chunks = partials every ~75ms so text appears instantly as you speak
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
        """Batch mode: longer chunks + calibration for better accuracy."""
        recognizer = sr.Recognizer()
        chunk_sec = 0.6  # More context = better accuracy (still responsive)
        mic = None
        try:
            mic = sr.Microphone(sample_rate=16000)
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
        except Exception:
            return
        while self.running and mic:
            try:
                with mic as source:
                    audio = recognizer.record(source, duration=chunk_sec)
                if not self.running:
                    break
                try:
                    text = recognizer.recognize_google(audio, language="en-US", show_all=False)
                    if text:
                        self.result_queue.put((text.strip(), RESULT_FINAL))
                except (sr.UnknownValueError, sr.RequestError):
                    pass
            except Exception:
                time.sleep(0.05)


def get_caption_mode():
    """Return 'streaming' if faster-whisper or Vosk available, else 'batch'."""
    if HAS_FASTER_WHISPER:
        return "streaming"
    if HAS_VOSK and _find_vosk_model():
        return "streaming"
    return "batch"
