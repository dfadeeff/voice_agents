"""Provider-layer contracts.

The pipeline depends on these abstractions, not on concrete vendor SDKs. A
provider is selected by name (env var) and built by the per-modality registries
in ``stt`` / ``llm`` / ``tts``; the orchestrator wires the returned service into
the Pipecat pipeline without knowing which vendor produced it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager


@runtime_checkable
class ConversationAware(Protocol):
    """A service that can be handed the live conversation so it can adapt to the
    current turn — e.g. local STT biasing its decoder toward the field the agent
    just asked for. Cloud services that don't need per-turn context simply don't
    implement this, and the orchestrator skips them (``isinstance`` is False)."""

    def set_conversation(self, conversation: ConversationManager) -> None: ...
