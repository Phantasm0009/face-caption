"""List available microphone devices. Use the index in .env as DEEPGRAM_INPUT_DEVICE_INDEX if needed."""
import pyaudio

p = pyaudio.PyAudio()
print("\n=== Available Audio Devices ===")
try:
    default_idx = p.get_default_input_device_info()["index"]
except Exception:
    default_idx = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        default = " [DEFAULT]" if (default_idx is not None and i == default_idx) else ""
        print(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']}){default}")
p.terminate()
