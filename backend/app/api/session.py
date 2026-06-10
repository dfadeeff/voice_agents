"""Shared call-session setup for every transport.

Both the browser WebSocket (`api/ws.py`) and Twilio Media Streams (`api/twilio.py`)
run the *same* business logic, so the provider creation, manager wiring (language
+ calendar), tool gating, capacity limiting, and the final caller upsert live here
— not duplicated (and, as happened on the Twilio path, not silently missing) in
each endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

from app.conversation.manager import ConversationManager
from app.pipeline.orchestrator import create_pipeline
from app.providers import create_llm, create_stt, create_tts
from app.services.calendar import CalendarService

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

_call_semaphore: asyncio.Semaphore | None = None


def call_capacity(settings: Settings) -> asyncio.Semaphore:
    """The one capacity gate for live calls, shared by every transport.

    Endpoints check ``capacity.locked()`` (public API, no private-attribute peek)
    and immediately enter ``async with capacity:`` — no await between check and
    acquire, so on a single-threaded event loop the check cannot go stale.
    """
    global _call_semaphore
    if _call_semaphore is None:
        _call_semaphore = asyncio.Semaphore(settings.max_concurrent_calls)
    return _call_semaphore


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
    upsert key, so it can never create a duplicate row. The manager owns the
    snapshot (outcome computation, what counts as "material"); we just write it.
    """
    record = conversation.snapshot_for_persistence()
    if record is None:
        return
    calendar: CalendarService = app.state.calendar
    calendar.upsert_caller_sync(**asdict(record))
