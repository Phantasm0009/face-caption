"""Test Deepgram configuration separately."""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(script_dir, ".env"))
except ImportError:
    pass

api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
if not api_key:
    print("ERROR: DEEPGRAM_API_KEY not found in .env file")
    sys.exit(1)

print("API Key found: {}...{}".format(api_key[:8], api_key[-4:]))

try:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
    import pyaudio
    import time
    import threading

    print("\n✓ Deepgram SDK imported successfully")

    client = DeepgramClient(api_key=api_key)
    print("✓ Deepgram client created")

    p = pyaudio.PyAudio()
    print("✓ PyAudio initialized ({} devices found)".format(p.get_device_count()))

    print("\nAvailable input devices:")
    try:
        default_idx = p.get_default_input_device_info()["index"]
    except Exception:
        default_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            default = " [DEFAULT]" if (default_idx is not None and i == default_idx) else ""
            print("  {}: {}{}".format(i, info["name"], default))

    print("\n🎤 Starting 5-second test (speak now)...\n")

    results = []

    def on_message(message):
        try:
            if hasattr(message, "channel") and message.channel:
                transcript = message.channel.alternatives[0].transcript
                if transcript.strip():
                    is_final = getattr(message, "speech_final", False)
                    results.append((transcript, is_final))
                    print("{} {}".format("[FINAL]" if is_final else "[PARTIAL]", transcript))
        except Exception as e:
            print("Message error:", e)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4096,
    )

    with client.listen.v1.connect(
        model="nova-2",
        language="en-US",
        encoding="linear16",
        sample_rate="16000",
        interim_results="true",
        smart_format="true",
    ) as socket_client:
        socket_client.on(EventType.MESSAGE, on_message)

        listener_done = threading.Event()

        def run_listener():
            socket_client.start_listening()
            listener_done.set()

        t = threading.Thread(target=run_listener, daemon=True)
        t.start()

        start_time = time.time()
        while time.time() - start_time < 5:
            data = stream.read(2048, exception_on_overflow=False)
            socket_client._send(data)
            time.sleep(0.02)

        socket_client._websocket.close()
        listener_done.wait(timeout=2)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("\n✓ Test complete! Received {} transcript results".format(len(results)))
    if not results:
        print("⚠ No speech detected. Check microphone volume/permissions.")

except ImportError as e:
    print("✗ Import error:", e)
    print("  Install: pip install deepgram-sdk pyaudio python-dotenv")
    sys.exit(1)
except Exception as e:
    print("✗ Error:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
