"""Download Vosk English model for real-time captions. Better model = better word recognition."""
import os
import sys
import urllib.request
import zipfile

# Small: ~40 MB, fast. Large: ~1.8 GB, better word recognition.
MODELS = {
    "small": (
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "vosk-model-small-en-us-0.15",
        "~40 MB",
    ),
    "large": (
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "vosk-model-en-us-0.22",
        "~1.8 GB",
    ),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


def main():
    use_large = "--large" in sys.argv or "-l" in sys.argv
    key = "large" if use_large else "small"
    url, folder_name, size_str = MODELS[key]
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_dir = os.path.join(MODELS_DIR, folder_name)
    zip_path = os.path.join(SCRIPT_DIR, folder_name + ".zip")

    if os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "conf", "model.conf")):
        print("Model already at:", model_dir)
        return

    print(f"Downloading Vosk {key} English model ({size_str})...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print("Download failed:", e)
        print("Get it manually:", url)
        return
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(MODELS_DIR)
    try:
        os.remove(zip_path)
    except Exception:
        pass
    print("Done. Model at:", model_dir)
    if key == "small":
        print("For better word recognition, run: python download_vosk_model.py --large")


if __name__ == "__main__":
    main()
