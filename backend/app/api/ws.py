import logging
import uuid

from fastapi import APIRouter, WebSocket

from pipecat.frames.frames import EndFrame
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from app.conversation.manager import ConversationManager
from app.pipeline.orchestrator import create_pipeline
from app.pipeline.services import create_stt, create_tts, create_llm

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/call/{call_id}")
async def websocket_call(websocket: WebSocket, call_id: str = "new"):
    await websocket.accept()

    if call_id == "new":
        call_id = str(uuid.uuid4())

    settings = websocket.app.state.settings

    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            add_wav_header=False,
            vad_enabled=True,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    stt = create_stt(settings)
    llm = create_llm(settings)
    tts = create_tts(settings)

    conversation = ConversationManager(call_id=call_id)
    tools = websocket.app.state.tool_registry

    use_tools = settings.llm_provider != "ollama"
    task, runner = await create_pipeline(
        stt, llm, tts, transport, websocket, conversation, tools,
        use_tools=use_tools,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, ws):
        logger.info("[%s] Client disconnected", call_id)
        await task.queue_frame(EndFrame())

    logger.info("[%s] Starting pipeline (tools=%s)", call_id, use_tools)
    await runner.run(task)
    logger.info("[%s] Pipeline finished", call_id)
