from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    language: str = "de"

    stt_provider: str = "whisper"
    llm_provider: str = "ollama"
    tts_provider: str = "piper"

    whisper_model_size: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    whisper_vad_filter: bool = False

    deepgram_api_key: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b-nothink"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    piper_model_path: str = "models/piper/de_DE-eva_k-x_low.onnx"
    piper_data_path: str = "models/piper/de_DE-eva_k-x_low.onnx.json"

    elevenlabs_api_key: str = ""

    sample_rate: int = 16000
    vad_threshold: float = 0.5
    silence_timeout_ms: int = 700

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
