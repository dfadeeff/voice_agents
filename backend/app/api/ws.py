import logging
import uuid

from fastapi import APIRouter, WebSocket
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from app.api.session import call_capacity, save_caller, start_call

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/call/{call_id}")
async def websocket_call(websocket: WebSocket, call_id: str = "new"):
    settings = websocket.app.state.settings
    capacity = call_capacity(settings)

    # locked() → acquire with no await in between: the check cannot go stale on a
    # single-threaded event loop, so a caller is either rejected here or gets a slot
    # (never silently queued waiting for one).
    if capacity.locked():
        await websocket.close(code=1013, reason="Server at capacity")
        return

    async with capacity:
        await websocket.accept()

        if call_id == "new":
            call_id = str(uuid.uuid4())

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

        conversation, task, runner = await start_call(websocket.app, transport, websocket, call_id)

        logger.info("[%s] Starting pipeline", call_id)
        try:
            await runner.run(task)
        except Exception:
            logger.exception("[%s] Pipeline crashed", call_id)

        try:
            save_caller(websocket.app, conversation)
        except Exception:
            logger.exception("[%s] Failed to save caller data", call_id)
        logger.info("[%s] Pipeline finished", call_id)
