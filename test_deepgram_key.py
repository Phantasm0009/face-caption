"""Test if DEEPGRAM_API_KEY is valid. Run from project folder: python test_deepgram_key.py"""
import os
import sys
import urllib.request

# Load .env from same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(script_dir, ".env"))
except ImportError:
    pass

api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
if not api_key:
    print("FAIL: No DEEPGRAM_API_KEY found.")
    print("  Set it in .env or environment. Example .env:")
    print("  DEEPGRAM_API_KEY=your_key_here")
    sys.exit(1)

print("Testing Deepgram API key...")
req = urllib.request.Request(
    "https://api.deepgram.com/v1/projects",
    headers={"Authorization": "Token " + api_key},
    method="GET",
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        if 200 <= r.status < 300:
            print("OK: API key is valid. Deepgram connection succeeded.")
        else:
            print("FAIL: Unexpected response", r.status)
            sys.exit(1)
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("FAIL: Invalid or unauthorized API key.")
        print("  Check your key at https://console.deepgram.com")
    else:
        print("FAIL: HTTP", e.code, "-", e.reason)
    sys.exit(1)
except Exception as e:
    print("FAIL:", type(e).__name__, "-", e)
    sys.exit(1)
