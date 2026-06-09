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

    if not settings.elevenlabs_voice_id:
        raise ValueError(
            "ElevenLabs needs ELEVENLABS_VOICE_ID — pick a German voice in the "
            "ElevenLabs Voice Library (filter by language = German) and set it in "
            "your env."
        )
    # The multilingual model (default) renders German from the German text; the
    # voice supplies the accent. (Language isn't forced — eleven_multilingual_v2
    # auto-detects per text, and not every model accepts a language_code.)
    return ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        settings=ElevenLabsTTSService.Settings(
            model=settings.elevenlabs_model,
            voice=settings.elevenlabs_voice_id,
        ),
    )


def _build_cartesia(settings: Settings) -> Any:
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.transcriptions.language import Language

    if not settings.cartesia_voice_id:
        raise ValueError(
            "Cartesia needs CARTESIA_VOICE_ID — pick a multilingual voice in the "
            "Cartesia dashboard (German-capable) and set it in your env."
        )
    # Use the current Settings API (the old voice_id/model/params kwargs are
    # deprecated). `language=Language.DE` makes the multilingual model render
    # German rather than reading it with English pronunciation.
    return CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        settings=CartesiaTTSService.Settings(
            model=settings.cartesia_model,
            voice=settings.cartesia_voice_id,
            language=Language(settings.language),
        ),
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
