"""Factory functions to create Pipecat service instances from app config."""

from pathlib import Path

from app.config import Settings


def create_stt(settings: Settings):
    if settings.stt_provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=settings.deepgram_api_key,
            settings=DeepgramSTTService.Settings(language=settings.language),
        )

    from pipecat.services.whisper.stt import WhisperSTTService

    return WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=settings.whisper_model_size,
            language=settings.language,
        ),
    )


def create_llm(settings: Settings):
    if settings.llm_provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(api_key=settings.openai_api_key, model=settings.openai_model)

    from pipecat.services.ollama.llm import OLLamaLLMService

    return OLLamaLLMService(
        settings=OLLamaLLMService.Settings(model=settings.ollama_model),
        base_url=f"{settings.ollama_base_url}/v1",
    )


def create_tts(settings: Settings):
    if settings.tts_provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        return ElevenLabsTTSService(api_key=settings.elevenlabs_api_key)

    from pipecat.services.piper.tts import PiperTTSService

    model_dir = str(Path(settings.piper_model_path).parent)
    voice_name = Path(settings.piper_model_path).stem
    return PiperTTSService(
        settings=PiperTTSService.Settings(voice=voice_name),
        download_dir=model_dir,
    )
