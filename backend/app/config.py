from pydantic_settings import BaseSettings


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

    sample_rate: int = 16000
    vad_threshold: float = 0.5
    silence_timeout_ms: int = 700
    filler_delay_ms: int = 1500

    host: str = "0.0.0.0"
    port: int = 8000
    db_url: str = "sqlite+aiosqlite:///data/voice_agent.db"

    redis_url: str = ""
    max_concurrent_calls: int = 10

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    use_tools_local: bool = True

    evaluator_enabled: bool = False
    evaluator_provider: str = "ollama"
    evaluator_model: str = "qwen3:4b"
    admin_api_key: str = ""

    model_config = {"env_file": ".env", "env_prefix": "", "extra": "ignore"}
