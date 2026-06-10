from pathlib import Path

from pydantic_settings import BaseSettings

# backend/app/config.py → parents[2] is the repo root, where .env lives.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    language: str = "de"

    stt_provider: str = "whisper"
    llm_provider: str = "ollama"
    tts_provider: str = "piper"

    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    whisper_vad_filter: bool = False

    deepgram_api_key: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    # Low temperature keeps a phone receptionist consistent and curbs the
    # invented-compound-word hallucinations qwen2.5 produces at higher temps.
    llm_temperature: float = 0.3
    # Completion-token cap for the local model. 0 = pick by model family
    # (qwen3 "thinking" models need headroom for the reasoning they then drop).
    ollama_max_tokens: int = 0

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # LLM-assisted extraction for the open-ended slots (spoken email + matter-type
    # classification). Keyword/regex still runs first for free; the LLM only fires
    # as a rescue when the deterministic parse fails. Spoken email over local
    # Whisper is the field that most needs this reassembly ("lang m at gmail punkt
    # com" → langm@gmail.com), so it is on by default (also implicitly on for the
    # OpenAI provider). Set false to force regex-only.
    llm_assist: bool = True

    piper_model_path: str = "models/piper/de_DE-eva_k-x_low.onnx"
    piper_data_path: str = "models/piper/de_DE-eva_k-x_low.onnx.json"
    # Silence appended after each synthesized sentence so consecutive sentences
    # don't run together in the audio. ~150-200ms reads as a natural breath.
    piper_sentence_pause_ms: int = 180

    elevenlabs_api_key: str = ""
    # ElevenLabs (cloud TTS). voice_id is required; the multilingual model speaks
    # German from German text, while the *voice* determines the accent — pick a
    # German voice in the ElevenLabs Voice Library for a native accent.
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"

    # Cartesia (cloud TTS) — streaming-first, lowest first-audio latency. voice_id
    # is required and must be a multilingual voice for German; sonic-2 is the
    # multilingual model.
    cartesia_api_key: str = ""
    cartesia_voice_id: str = ""
    cartesia_model: str = "sonic-2"

    vad_threshold: float = 0.5
    # Silence after which a normal turn is considered finished (VAD stop_secs).
    # Kept snappy for conversational turns.
    silence_timeout_ms: int = 800
    # A longer window used only while the caller is dictating a number/email
    # (insurance, phone, email) — those are read out with mid-utterance pauses
    # ("F fünf vier … sechs") that a short window would chop into separate turns.
    # Applied per-turn based on what the agent just asked for.
    silence_timeout_dictation_ms: int = 2000
    filler_delay_ms: int = 1500

    host: str = "0.0.0.0"
    port: int = 8000
    db_url: str = "sqlite+aiosqlite:///data/voice_agent.db"

    # Honest local default: the shared Whisper model serializes transcription on
    # CPU and the per-turn SQLite upsert is a small blocking write, so more than a
    # couple of simultaneous local calls degrade each other anyway. Raise it on a
    # cloud stack (streaming STT, async DB); the gate itself is per-process.
    max_concurrent_calls: int = 2

    # Allowed CORS origins (comma-separated). Defaults to localhost — the browser
    # demo is served same-origin, so it doesn't need "*", and this service handles
    # legal PII. Set to your frontend origin(s) in production; "*" opts out.
    cors_allow_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    use_tools_local: bool = True

    # Pin .env to the repo root: a bare ".env" resolves against the CWD, but
    # `make run` does `cd backend`, so the root .env would otherwise be missed and
    # every setting (including cloud API keys) would silently fall back to defaults.
    model_config = {"env_file": str(_REPO_ROOT / ".env"), "env_prefix": "", "extra": "ignore"}
