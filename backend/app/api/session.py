"""Shared call-session setup for every transport.

Both the browser WebSocket (`api/ws.py`) and Twilio Media Streams (`api/twilio.py`)
run the *same* business logic, so the provider creation, manager wiring (language
+ calendar), tool gating, and the final caller upsert live here — not duplicated
(and, as happened on the Twilio path, not silently missing) in each endpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.conversation.manager import ConversationManager
from app.pipeline.orchestrator import create_pipeline
from app.providers import create_llm, create_stt, create_tts
from app.services.calendar import CalendarService

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


def use_tools_enabled(settings: Settings) -> bool:
    """Tool calling is on for cloud LLMs always, and for local Ollama only when
    explicitly enabled (small local models call tools unreliably)."""
    return settings.llm_provider != "ollama" or settings.use_tools_local


def build_conversation(
    settings: Settings, calendar: CalendarService, call_id: str
) -> ConversationManager:
    """Wire a manager with the call language and calendar.

    The Twilio path previously built ``ConversationManager(call_id=call_id)`` with
    neither, so booking silently no-opped (calendar=None) and the language was
    English-or-luck. Both transports now go through here.
    """
    return ConversationManager(call_id=call_id, lang=settings.language, calendar=calendar)


async def start_call(app, transport, websocket, call_id: str):
    """Build providers + manager + pipeline for a call on the given transport.

    Returns ``(conversation, task, runner)``. The caller owns the transport (its
    serializer/params differ per channel) and the run loop; everything downstream
    of the transport is identical across channels.
    """
    settings: Settings = app.state.settings
    stt = create_stt(settings)
    llm = create_llm(settings)
    tts = create_tts(settings)

    conversation = build_conversation(settings, app.state.calendar, call_id)
    use_tools = use_tools_enabled(settings)
    task, runner = await create_pipeline(
        stt,
        llm,
        tts,
        transport,
        websocket,
        conversation,
        app.state.tool_registry,
        use_tools=use_tools,
    )
    logger.info("[%s] Pipeline ready (tools=%s)", call_id, use_tools)
    return conversation, task, runner


def save_caller(app, conversation: ConversationManager) -> None:
    """Final upsert of caller data with the call outcome, on teardown.

    ``manager`` persists incrementally during the call; this records the terminal
    outcome (booked/callback/escalation) known only at the end. Runs for *every*
    transport now (the Twilio path skipped it entirely before). Uses the same
    upsert key, so it can never create a duplicate row.
    """
    calendar: CalendarService = app.state.calendar
    state = conversation.state
    entities = state.entities

    def _val(field: str) -> str:
        e = entities.get(field)
        return e.value if e else ""

    if not _val("name") and not _val("phone"):
        return

    outcome = state.phase.value
    if state.booking_confirmed:
        outcome = "booked"
    elif state.callback_requested:
        outcome = "callback"
    elif state.escalation_requested:
        outcome = "escalation"

    calendar.upsert_caller_sync(
        call_id=state.call_id,
        name=_val("name"),
        phone=_val("phone"),
        email=_val("email"),
        legal_area=state.legal_area.value,
        matter_type=_val("matter_type"),
        matter_summary=state.matter_summary or "",
        case_reference=_val("case_reference"),
        insurance_number=_val("insurance_number"),
        outcome=outcome,
        preferred_time=state.preferred_time or "",
    )
