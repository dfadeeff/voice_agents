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

logger = logging.getLogger(__name__)
router = APIRouter()

_call_semaphore: asyncio.Semaphore | None = None


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
        await runner.run(task)
        logger.info("[%s] Pipeline finished", call_id)
