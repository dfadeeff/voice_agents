"""Text-to-speech providers.

Add a vendor by writing a builder and adding one ``_BUILDERS`` entry. Vendor SDK
imports stay inside each builder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import Settings

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

TtsBuilder = Callable[[Settings], Any]


def _build_piper(settings: Settings) -> Any:
    from app.pipeline.local_piper import LocalPiperTTSService

    model_path = Path(settings.piper_model_path)
    if not model_path.is_absolute():
        model_path = _BACKEND_ROOT / model_path
    return LocalPiperTTSService(
        sentence_pause_ms=settings.piper_sentence_pause_ms,
        settings=LocalPiperTTSService.Settings(voice=model_path.stem),
        download_dir=model_path.parent,
    )


def _build_elevenlabs(settings: Settings) -> Any:
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    return ElevenLabsTTSService(api_key=settings.elevenlabs_api_key)


def _build_cartesia(settings: Settings) -> Any:
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.transcriptions.language import Language

    if not settings.cartesia_voice_id:
        raise ValueError(
            "Cartesia needs CARTESIA_VOICE_ID — pick a multilingual voice in the "
            "Cartesia dashboard (German-capable) and set it in your env."
        )
    return CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        voice_id=settings.cartesia_voice_id,
        model=settings.cartesia_model,
        # Speak the configured language (e.g. "de" -> Language.DE) so a
        # multilingual voice renders German rather than defaulting to English.
        params=CartesiaTTSService.InputParams(language=Language(settings.language)),
    )


_BUILDERS: dict[str, TtsBuilder] = {
    "piper": _build_piper,
    "elevenlabs": _build_elevenlabs,
    "cartesia": _build_cartesia,
}


def create_tts(settings: Settings) -> Any:
    try:
        builder = _BUILDERS[settings.tts_provider]
    except KeyError:
        raise ValueError(
            f"Unknown TTS provider {settings.tts_provider!r}; known: {sorted(_BUILDERS)}"
        ) from None
    return builder(settings)
