"""Factory functions to create Pipecat service instances from app config."""

from pathlib import Path

from app.config import Settings

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


def create_stt(settings: Settings):
    if settings.stt_provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=settings.deepgram_api_key,
            settings=DeepgramSTTService.Settings(language=settings.language),
        )

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


def create_llm(settings: Settings):
    if settings.llm_provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(api_key=settings.openai_api_key, model=settings.openai_model)

    from pipecat.services.ollama.llm import OLLamaLLMService

    max_tokens = 400 if "qwen3" in settings.ollama_model else 200
    return OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=settings.ollama_model,
            max_tokens=max_tokens,
        ),
        base_url=f"{settings.ollama_base_url}/v1",
    )


def create_tts(settings: Settings):
    if settings.tts_provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        return ElevenLabsTTSService(api_key=settings.elevenlabs_api_key)

    from pipecat.services.piper.tts import PiperTTSService

    model_path = Path(settings.piper_model_path)
    if not model_path.is_absolute():
        model_path = _BACKEND_ROOT / model_path
    return PiperTTSService(
        settings=PiperTTSService.Settings(voice=model_path.stem),
        download_dir=model_path.parent,
    )
