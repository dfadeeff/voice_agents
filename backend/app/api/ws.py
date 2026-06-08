import asyncio
import logging
import uuid

from fastapi import APIRouter, WebSocket
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from app.conversation.manager import ConversationManager
from app.pipeline.orchestrator import create_pipeline
from app.pipeline.services import create_llm, create_stt, create_tts
from app.services.calendar import CalendarService

logger = logging.getLogger(__name__)
router = APIRouter()

_call_semaphore: asyncio.Semaphore | None = None


async def _save_caller(calendar: CalendarService, conv: ConversationManager) -> None:
    """Persist caller details to the callers table after every call."""
    state = conv.state
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

    await calendar.save_caller(
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


def _get_semaphore(max_calls: int) -> asyncio.Semaphore:
    global _call_semaphore
    if _call_semaphore is None:
        _call_semaphore = asyncio.Semaphore(max_calls)
    return _call_semaphore


@router.websocket("/ws/call/{call_id}")
async def websocket_call(websocket: WebSocket, call_id: str = "new"):
    settings = websocket.app.state.settings
    sem = _get_semaphore(settings.max_concurrent_calls)

    if not sem._value:
        await websocket.close(code=1013, reason="Server at capacity")
        return

    await websocket.accept()

    if call_id == "new":
        call_id = str(uuid.uuid4())

    async with sem:
        transport = FastAPIWebsocketTransport(
            websocket,
            FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                add_wav_header=False,
                serializer=ProtobufFrameSerializer(),
            ),
        )

        stt = create_stt(settings)
        llm = create_llm(settings)
        tts = create_tts(settings)

        conversation = ConversationManager(call_id=call_id, lang=settings.language)
        tools = websocket.app.state.tool_registry

        use_tools = settings.llm_provider != "ollama" or settings.use_tools_local
        task, runner = await create_pipeline(
            stt,
            llm,
            tts,
            transport,
            websocket,
            conversation,
            tools,
            use_tools=use_tools,
        )

        logger.info("[%s] Starting pipeline (tools=%s)", call_id, use_tools)
        try:
            await runner.run(task)
        except Exception:
            logger.exception("[%s] Pipeline crashed", call_id)

        calendar: CalendarService = websocket.app.state.calendar
        try:
            await _save_caller(calendar, conversation)
        except Exception:
            logger.exception("[%s] Failed to save caller data", call_id)
        logger.info("[%s] Pipeline finished", call_id)
