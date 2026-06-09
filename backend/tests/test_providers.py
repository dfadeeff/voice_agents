"""Provider abstraction: registries, error handling, and Cartesia wiring."""

import pytest
from app.config import Settings
from app.providers import ConversationAware
from app.providers.llm import _BUILDERS as LLM_BUILDERS
from app.providers.llm import _ollama_max_tokens
from app.providers.stt import _BUILDERS as STT_BUILDERS
from app.providers.stt import create_stt
from app.providers.tts import _BUILDERS as TTS_BUILDERS
from app.providers.tts import _build_cartesia, create_tts


class TestRegistries:
    def test_known_providers_registered(self):
        assert set(STT_BUILDERS) == {"whisper", "deepgram"}
        assert set(LLM_BUILDERS) == {"ollama", "openai"}
        assert {"piper", "elevenlabs", "cartesia"} <= set(TTS_BUILDERS)

    def test_unknown_provider_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unknown STT provider 'bogus'"):
            create_stt(Settings(stt_provider="bogus"))
        with pytest.raises(ValueError, match="Unknown TTS provider 'nope'"):
            create_tts(Settings(tts_provider="nope"))


class TestOllamaMaxTokens:
    def test_default_by_family(self):
        assert _ollama_max_tokens(Settings(ollama_model="qwen2.5:7b")) == 150
        assert _ollama_max_tokens(Settings(ollama_model="qwen3:8b")) == 400

    def test_explicit_override_wins(self):
        assert _ollama_max_tokens(Settings(ollama_model="qwen3:8b", ollama_max_tokens=120)) == 120


class TestCartesia:
    def test_missing_voice_id_raises_helpful_error(self):
        # The whole point: don't silently build a voiceless/English service.
        with pytest.raises(ValueError, match="CARTESIA_VOICE_ID"):
            _build_cartesia(Settings(cartesia_api_key="k", cartesia_voice_id=""))

    def test_german_language_maps_to_enum(self):
        # "de" must resolve to a real Cartesia Language member, not blow up.
        from pipecat.transcriptions.language import Language

        assert Language(Settings().language) is Language.DE


class TestElevenLabs:
    def test_missing_voice_id_raises_helpful_error(self):
        # Same guard as Cartesia: don't build the default English voice silently.
        from app.providers.tts import _build_elevenlabs

        with pytest.raises(ValueError, match="ELEVENLABS_VOICE_ID"):
            _build_elevenlabs(Settings(elevenlabs_api_key="k", elevenlabs_voice_id=""))

    def test_multilingual_model_is_default(self):
        # German correctness comes from the multilingual model, not the voice.
        assert Settings().elevenlabs_model == "eleven_multilingual_v2"


class TestConversationAware:
    def test_local_whisper_is_conversation_aware(self):
        from app.pipeline.local_whisper import LocalWhisperSTTService

        svc = LocalWhisperSTTService.__new__(LocalWhisperSTTService)
        assert isinstance(svc, ConversationAware)
