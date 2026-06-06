"""Download local models for the voice agent (Piper TTS + Ollama LLM)."""

import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
PIPER_FILES = [
    ("en_US-lessac-medium.onnx", f"{PIPER_BASE}/en_US-lessac-medium.onnx"),
    ("en_US-lessac-medium.onnx.json", f"{PIPER_BASE}/en_US-lessac-medium.onnx.json"),
]
PIPER_DIR = Path("models/piper")

OLLAMA_MODEL = "qwen3:4b"


def download_piper():
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in PIPER_FILES:
        dest = PIPER_DIR / filename
        if dest.exists():
            print(f"  [skip] {dest} already exists")
            continue
        print(f"  [download] {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  [done] {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def pull_ollama():
    print(f"  Pulling {OLLAMA_MODEL} via Ollama...")
    print("  (Make sure 'ollama serve' is running in another terminal)")
    try:
        subprocess.run(
            ["ollama", "pull", OLLAMA_MODEL],
            check=True,
        )
    except FileNotFoundError:
        print()
        print("  Ollama not found. Install it first:")
        print("    macOS:  brew install ollama")
        print("    Linux:  curl -fsSL https://ollama.com/install.sh | sh")
        print("    Or:     https://ollama.com/download")
        sys.exit(1)


def download_nltk():
    import nltk

    nltk.download("punkt_tab", quiet=True)
    print("  [done] punkt_tab tokenizer")


def main():
    print("=== Downloading Piper TTS voice ===")
    download_piper()
    print()
    print("=== Pulling Ollama LLM model ===")
    pull_ollama()
    print()
    print("=== Downloading NLTK data ===")
    download_nltk()
    print()
    print("All models ready.")


if __name__ == "__main__":
    main()
