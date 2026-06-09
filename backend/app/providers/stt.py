"""Speech-to-text providers.

Add a vendor by writing a builder and adding one ``_BUILDERS`` entry — no caller
changes. Vendor SDK imports stay inside each builder so an unused provider's
dependency is never imported.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.config import Settings

SttBuilder = Callable[[Settings], Any]


def _build_whisper(settings: Settings) -> Any:
    from app.pipeline.local_whisper import LocalWhisperSTTService

    return LocalWhisperSTTService(
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter,
        settings=LocalWhisperSTTService.Settings(
            model=settings.whisper_model_size,
            language=settings.language,
        ),
    )


def _build_deepgram(settings: Settings) -> Any:
    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(language=settings.language),
    )


_BUILDERS: dict[str, SttBuilder] = {
    "whisper": _build_whisper,
    "deepgram": _build_deepgram,
}


def create_stt(settings: Settings) -> Any:
    try:
        builder = _BUILDERS[settings.stt_provider]
    except KeyError:
        raise ValueError(
            f"Unknown STT provider {settings.stt_provider!r}; known: {sorted(_BUILDERS)}"
        ) from None
    return builder(settings)
