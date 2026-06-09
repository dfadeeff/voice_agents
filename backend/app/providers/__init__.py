"""Provider abstraction: vendor-agnostic factories for the pipeline services.

The pipeline asks for a service by modality (``create_stt``/``create_llm``/
``create_tts``); the concrete vendor is chosen by name (env var) inside the
per-modality registries. Pipeline code depends only on these factories and the
``ConversationAware`` contract — never on a specific vendor SDK.
"""

from app.providers.base import ConversationAware
from app.providers.llm import create_llm
from app.providers.stt import create_stt
from app.providers.tts import create_tts

__all__ = ["ConversationAware", "create_llm", "create_stt", "create_tts"]
