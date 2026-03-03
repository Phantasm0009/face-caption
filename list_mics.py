"""List available microphone devices.

Use an index in config.json as "mic_input_device_index" to choose which mic
the app uses for speech-to-text. Example: "mic_input_device_index": 2
Use null for system default.
"""
import pyaudio

p = pyaudio.PyAudio()
print("\n=== Microphones (use index in config.json → mic_input_device_index) ===\n")
try:
    default_idx = p.get_default_input_device_info()["index"]
except Exception:
    default_idx = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        default = "  ← default" if (default_idx is not None and i == default_idx) else ""
        print(f"  {i}: {info['name']}{default}")
print("\nIn config.json set: \"mic_input_device_index\": <number>  (or null for default)")
p.terminate()
