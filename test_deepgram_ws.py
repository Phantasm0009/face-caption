"""Test Deepgram WebSocket handshake; prints dg-error and dg-request-id on 400."""
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
    print("No DEEPGRAM_API_KEY in .env")
    sys.exit(1)

try:
    import websockets.sync.client as ws_client
except ImportError:
    import websockets
    ws_client = websockets  # fallback

base = "wss://api.deepgram.com/v1/listen"
# Minimal params (same as realtime_stt after fix)
params = "model=nova-2&language=en-US&encoding=linear16&sample_rate=16000"
url = f"{base}?{params}"
headers = {"Authorization": f"Token {api_key}"}

print("Connecting to", base, "with params:", params)
try:
    with ws_client.connect(url, additional_headers=headers) as ws:
        rid = getattr(ws, "response_headers", None) or getattr(ws, "response", None)
        if rid and hasattr(rid, "get"):
            print("OK: WebSocket connected. Request ID:", rid.get("dg-request-id"))
        else:
            print("OK: WebSocket connected.")
        # Close immediately so script exits (no audio sent)
        ws.close()
except Exception as e:
    print("Connection failed:", type(e).__name__, e)
    if hasattr(e, "response") and e.response is not None:
        r = e.response
        if hasattr(r, "headers"):
            print("dg-error:", r.headers.get("dg-error"))
            print("dg-request-id:", r.headers.get("dg-request-id"))
    sys.exit(1)
