"""Download local models for the voice agent (Piper TTS + Ollama LLM)."""

import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

PIPER_VOICES = {
    "en": {
        "base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium",
        "files": ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"],
    },
    "de": {
        "base": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/eva_k/x_low",
        "files": ["de_DE-eva_k-x_low.onnx", "de_DE-eva_k-x_low.onnx.json"],
    },
}
PIPER_DIR = Path("models/piper")

OLLAMA_MODEL = "qwen2.5:7b"


def download_piper():
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    for lang, voice in PIPER_VOICES.items():
        print(f"  [{lang}]")
        for filename in voice["files"]:
            dest = PIPER_DIR / filename
            if dest.exists():
                print(f"    [skip] {dest} already exists")
                continue
            url = f"{voice['base']}/{filename}"
            print(f"    [download] {filename}...")
            urllib.request.urlretrieve(url, dest)
            print(f"    [done] {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


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
    print("=== Downloading Piper TTS voices ===")
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
