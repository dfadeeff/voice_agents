import json
import logging
import uuid

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response
from pipecat.frames.frames import EndFrame
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from app.api.session import save_caller, start_call

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/twilio")


@router.post("/voice")
async def twilio_voice_webhook(request: Request):
    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/twilio/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.websocket("/stream")
async def twilio_media_stream(websocket: WebSocket):
    await websocket.accept()

    call_id = str(uuid.uuid4())
    settings = websocket.app.state.settings

    # Wait for the Twilio "start" event to get stream_sid
    start_data = await websocket.receive_text()
    start_msg = json.loads(start_data)
    while start_msg.get("event") != "start":
        start_data = await websocket.receive_text()
        start_msg = json.loads(start_data)

    stream_sid = start_msg["start"]["streamSid"]
    call_sid = start_msg["start"].get("callSid")
    logger.info("[%s] Twilio stream started: %s", call_id, stream_sid)

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid or None,
        auth_token=settings.twilio_auth_token or None,
        params=TwilioFrameSerializer.InputParams(
            sample_rate=16000,
            auto_hang_up=bool(settings.twilio_account_sid and settings.twilio_auth_token),
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            serializer=serializer,
            vad_enabled=True,
        ),
    )

    # Same wiring as the browser path (language + calendar + tool gating) so phone
    # calls actually book — previously this built a manager with no calendar and
    # every caller hit the "no free slots" branch.
    conversation, task, runner = await start_call(websocket.app, transport, websocket, call_id)

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, ws):
        logger.info("[%s] Twilio stream disconnected", call_id)
        await task.queue_frame(EndFrame())

    logger.info("[%s] Starting Twilio pipeline", call_id)
    try:
        await runner.run(task)
    finally:
        try:
            save_caller(websocket.app, conversation)
        except Exception:
            logger.exception("[%s] Failed to save caller data", call_id)
    logger.info("[%s] Twilio pipeline finished", call_id)
