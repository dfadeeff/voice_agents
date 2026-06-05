# Voice AI Agent — Architecture

## Context

Inbound voice agent for a law firm. Handles calls end-to-end: greeting, routing by legal area, entity capture with confidence handling, consultation booking, and escalation to humans. Runs locally with zero API keys by default; swaps to cloud providers (Deepgram, OpenAI, ElevenLabs) via env vars for production deployment with Twilio telephony.

## Stack

| Layer | Local (default) | Cloud (production) |
|-------|----------------|---------------------|
| **STT** | faster-whisper (base model) | Deepgram Nova-2 (streaming) |
| **LLM** | Ollama llama3.1 8B | OpenAI GPT-4o-mini |
| **TTS** | Piper (en_US-lessac-medium) | ElevenLabs (streaming) |
| **VAD** | Silero VAD | same |
| **Transport** | Browser WebSocket | + Twilio Media Streams |
| **DB** | SQLite (aiosqlite) | Postgres (same SQLAlchemy models) |
| **Telephony** | none (mic/speaker) | Twilio (inbound phone number) |

## Project Structure

```
voice_agent/
├── ARCHITECTURE.md
├── README.md
├── .env.example
├── Makefile                         # make setup, make run, make seed
│
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py                  # FastAPI app, lifespan, CORS, static files
│   │   ├── config.py                # Pydantic Settings (env-driven provider selection)
│   │   │
│   │   ├── api/
│   │   │   ├── ws.py                # WebSocket /ws/call/{call_id} (browser transport)
│   │   │   ├── twilio.py            # POST /twilio/voice + Media Streams adapter
│   │   │   └── health.py            # GET /health, /ready
│   │   │
│   │   ├── pipeline/
│   │   │   ├── orchestrator.py      # Pipecat pipeline factory + tool registration
│   │   │   └── services.py          # Create Pipecat STT/LLM/TTS from config
│   │   │
│   │   ├── conversation/
│   │   │   ├── state.py             # ConversationState dataclass (serializable)
│   │   │   ├── manager.py           # Manages state per call, confidence scoring
│   │   │   └── prompts.py           # System prompt + law-type fragments
│   │   │
│   │   ├── tools/
│   │   │   ├── registry.py          # Tool name -> callable + JSON schema
│   │   │   ├── routing.py           # classify_legal_area
│   │   │   ├── extraction.py        # extract_caller_details + confidence scoring
│   │   │   ├── booking.py           # check_availability, book_consultation
│   │   │   └── escalation.py        # escalate_to_human
│   │   │
│   │   ├── services/
│   │   │   └── calendar.py          # SQLite calendar: slots, bookings, call logs
│   │   │
│   │   └── models/
│   │       └── schemas.py           # Dataclasses: CallPhase, LegalArea, entities
│   │
│   └── tests/
│
├── frontend/
│   ├── index.html                   # Single page, no build step
│   ├── style.css
│   ├── app.js                       # WebSocket client, transcript UI
│   └── components/
│       ├── protobuf.js              # Pipecat frame encoder/decoder
│       └── audio.js                 # AudioWorklet mic capture + playback
│
├── scripts/
│   ├── seed_calendar.py             # Populate 2 weeks of slots
│   └── download_models.py           # Download Piper voice + pull Ollama model
│
└── data/                            # .gitignored, created at runtime
    └── voice_agent.db
```

## System Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           TRANSPORT LAYER                │
                    │                                          │
                    │  ┌──────────────┐  ┌─────────────────┐  │
                    │  │ Browser UI   │  │ Twilio Media     │  │
                    │  │ (WebSocket)  │  │ Streams          │  │
                    │  │              │  │ (mulaw 8kHz)     │  │
                    │  │ mic -> PCM   │  │                  │  │
                    │  │ 16kHz mono   │  │ phone -> mulaw   │  │
                    │  └──────┬───────┘  └────────┬────────┘  │
                    │         │                    │           │
                    │         │    ┌───────────┐   │           │
                    │         └───>│ Audio     │<──┘           │
                    │              │ Adapter   │               │
                    │              │ (normalize│               │
                    │              │  to PCM   │               │
                    │              │  16kHz)   │               │
                    │              └─────┬─────┘               │
                    └────────────────────┼─────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                   PIPECAT PIPELINE (orchestrator.py)                 │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐  │
│  │ Silero   │──>│ Whisper STT  │──>│ Ollama LLM   │──>│ Piper  │  │
│  │ VAD      │   │ (Pipecat     │   │ (Pipecat     │   │  TTS   │  │
│  │ (built-  │   │  service)    │   │  service)    │   │(Pipecat│  │
│  │  in)     │   │              │   │              │   │service)│  │
│  │          │   │ or Deepgram  │   │ or OpenAI    │   │        │  │
│  │ speech   │   │ (cloud)      │   │ (cloud)      │   │or 11L  │  │
│  │ detect + │   │              │   │              │   │(cloud) │  │
│  │ endpoint │   │ -> text      │   │ -> text +    │   │->audio │  │
│  │          │   │              │   │    tool calls│   │  chunks│  │
│  └──────────┘   └──────────────┘   └──────┬───────┘   └────────┘  │
│       │                                    │                        │
│  barge-in:                           tool calls                     │
│  speech during                            │                         │
│  TTS -> cancel               ┌────────────┼────────────┐           │
│  + process new               │            │            │           │
│                              ▼            ▼            ▼           │
│                       ┌───────────┐┌───────────┐┌───────────┐      │
│                       │classify_  ││extract_   ││check_     │      │
│                       │legal_area ││entities   ││availability│      │
│                       │           ││           ││book_consult│      │
│                       │employment ││name: 0.92 ││           │      │
│                       │tenancy    ││email: 0.45 ││  SQLite   │      │
│                       │unknown->  ││  -> CONFIRM││  calendar │      │
│                       │ escalate  ││           ││           │      │
│                       └───────────┘└───────────┘└───────────┘      │
│                              │                                      │
│                              ▼                                      │
│                       ┌───────────┐                                 │
│                       │escalate_  │                                 │
│                       │to_human   │                                 │
│                       │           │                                 │
│                       │logs context                                 │
│                       │for handoff│                                 │
│                       └───────────┘                                 │
│                                                                     │
│              ┌──────────────────────────┐                           │
│              │   Conversation State     │                           │
│              │   (in-memory | Redis)    │                           │
│              │                          │                           │
│              │  phase: greeting|routing │                           │
│              │         |capture|booking │                           │
│              │         |escalate|done   │                           │
│              │  legal_area: ?           │                           │
│              │  entities: {}            │                           │
│              │  messages: [...]         │                           │
│              └──────────────────────────┘                           │
└────────────────────────────────────────────────────────────────────┘
```

## Call Flow

### High-Level State Machine

```
                          ┌───────────┐
                          │ GREETING  │
                          │ "Hello,   │
                          │ how can I │
                          │ help?"    │
                          └─────┬─────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
        ┌──────────┐     ┌──────────┐       ┌──────────┐
        │ ROUTING  │     │  DIRECT  │       │ ESCALATE │
        │ "what's  │     │ BOOKING  │       │ unclear/ │
        │ your     │     │ "book a  │       │ complex  │
        │ issue?"  │     │ consult" │       │          │
        └────┬─────┘     └────┬─────┘       └──────────┘
             │                │                    ▲
             ▼                │              at ANY point:
        ┌──────────┐         │              - "talk to someone"
        │ INFO     │         │              - 3 misunderstandings
        │ branch   │         │              - out-of-scope matter
        │ by law   │         │              - frustration detected
        │ type     │         │
        └────┬─────┘         │
             │               │
             ▼               ▼
        ┌───────────────────────┐
        │ CAPTURE               │
        │                       │◄──── low confidence?
        │ name ──── confirm? ───┤───── spell back
        │ email ─── confirm? ───┤───── read back letter by letter
        │ phone ─── confirm? ───┤───── repeat digit by digit
        │ preferred_time ───────┤
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ BOOKING               │
        │                       │
        │ check_availability()  │
        │ ├─ available → CONFIRM_ALL → book_slot() → FAREWELL
        │ └─ unavailable → OFFER_ALTERNATIVES → caller picks
        └───────────────────────┘
```

Not a rigid FSM -- the LLM manages flow naturally via system prompt. States are implicit: calling `classify_legal_area` = routing, calling `extract_entities` = capture, etc.

### Single Turn Data Flow

```
Caller speaks: "I think I'm being unfairly dismissed from my job"
         │
         ▼
    ┌─────────┐
    │ Silero  │  monitors audio energy continuously
    │ VAD     │  detects 300ms+ silence after speech -> end-of-utterance
    └────┬────┘
         │ complete audio segment (e.g. 2.3 seconds of PCM)
         ▼
    ┌──────────────┐
    │ STT Provider │  transcribes full segment (whisper) or streams (deepgram)
    │              │  returns: text + per-word confidence scores
    └──────┬───────┘
           │ "I think I'm being unfairly dismissed from my job"
           │  word confidences: [0.95, 0.97, 0.92, 0.88, 0.94, ...]
           ▼
    ┌──────────────┐
    │ LLM Provider │  sees: system prompt + conversation history + transcript
    │              │  decides: employment law -> call classify_legal_area()
    │              │  tool returns: {type: "employment", sub: "unfair_dismissal"}
    │              │  generates response (streaming tokens):
    │              │    "I understand you're dealing with a potential unfair
    │              │     dismissal. I can help connect you with our employment
    │              │     law team. Can you tell me..."
    └──────┬───────┘
           │ tokens stream out -> split at sentence boundaries
           ▼
    ┌──────────────┐
    │ TTS Provider │  receives first sentence immediately
    │              │  generates audio while LLM still producing next sentence
    └──────┬───────┘
           │ audio chunks
           ▼
      Transport plays audio to caller
      (browser speaker or Twilio media stream)
```

## Provider Abstraction

Uses Pipecat's built-in service classes. Provider swapping via env var — same interface, different backend.

### Service Factory (backend/app/pipeline/services.py)

```python
def create_stt(settings):
    if settings.stt_provider == "deepgram":
        return DeepgramSTTService(api_key=settings.deepgram_api_key)
    return WhisperSTTService(model=settings.whisper_model_size, ...)

def create_llm(settings):
    if settings.llm_provider == "openai":
        return OpenAILLMService(api_key=settings.openai_api_key, model=settings.openai_model)
    return OLLamaLLMService(model=settings.ollama_model, base_url=...)

def create_tts(settings):
    if settings.tts_provider == "elevenlabs":
        return ElevenLabsTTSService(api_key=settings.elevenlabs_api_key)
    return PiperTTSService(voice_id=..., download_dir=...)
```

All services are Pipecat-native — they plug directly into the pipeline and handle streaming, audio format conversion, and frame passing internally.

## Confidence-Aware Entity Extraction

The "hard part" the brief emphasizes. Two sources of confidence:

### 1. STT-level confidence (per-word)

faster-whisper provides `log_prob` per word. Convert: `confidence = exp(log_prob)`.
Deepgram provides `confidence` directly (0.0-1.0).

### 2. LLM-level confidence (semantic)

The LLM calls `extract_entities` tool. The tool cross-references extracted values against the STT word confidence scores:

```
Caller: "My name is Siobhan McNally"
                          │
        faster-whisper:   │
        "shivan mcnally"  │
        word_conf: [0.62, 0.71]   <- LOW for "Siobhan"
                          │
        extract_entities returns:
        {
          "name": {"value": "Shivan McNally", "confidence": 0.62},
          "needs_confirmation": [
            {"field": "name", "confidence": 0.62,
             "suggestion": "Please confirm spelling of first name"}
          ]
        }
                          │
        LLM (per system prompt):
        "I want to make sure I have your name right.
         Could you spell your first name for me?"
```

Threshold: 0.7. Below -> MUST confirm before proceeding to booking.

## Twilio Telephony Integration

### How it works

```
Phone call ──> Twilio ──> POST /twilio/voice (webhook)
                              │
                              ▼
                         Return TwiML:
                         <Connect>
                           <Stream url="wss://your-server/twilio/stream"/>
                         </Connect>
                              │
                              ▼
                    Twilio opens WebSocket
                    to /twilio/stream
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Pipecat FastAPI Transport   │
              │   + TwilioFrameSerializer     │
              │                               │
              │   Receives: mulaw 8kHz audio  │
              │   Converts: -> PCM 16kHz      │
              │   Feeds: -> same Pipeline     │
              │                               │
              │   Receives: PCM from TTS      │
              │   Converts: -> mulaw 8kHz     │
              │   Sends: -> Twilio stream     │
              └───────────────────────────────┘
```

### Key: transport is just an adapter

The Pipecat pipeline doesn't know or care whether audio comes from a browser WebSocket or a Twilio Media Stream. Both use `FastAPIWebsocketTransport` with different serializers:
- **Browser**: `ProtobufFrameSerializer` — PCM 16kHz, no conversion needed
- **Twilio**: `TwilioFrameSerializer` — handles mulaw 8kHz <-> PCM 16kHz conversion automatically

### Twilio config (.env)

```
TWILIO_ACCOUNT_SID=           # Only needed for phone call support
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
```

### Deployment for phone calls

1. Deploy backend to Railway/Fly.io (free tier)
2. Set env vars to cloud providers (Deepgram + OpenAI + ElevenLabs)
3. Configure Twilio webhook: POST https://your-app.railway.app/twilio/voice
4. Evaluators can call the Twilio number directly

## Scaling Architecture

### What's ready now (prototype)

- In-memory conversation state per WebSocket connection
- SQLite database
- Single-process FastAPI with ThreadPoolExecutor for CPU work
- Local audio I/O (browser or Twilio)

### What changes for scale (config, not rewrite)

| Concern | Prototype | Production |
|---------|-----------|------------|
| State | In-memory dict | Redis (set `REDIS_URL`) |
| DB | SQLite | Postgres (change `DATABASE_URL`) |
| STT/TTS compute | ThreadPoolExecutor | Celery workers (same async interface) |
| Concurrency | Semaphore(10) | Multiple instances + load balancer |
| Monitoring | Stdout logging | Prometheus metrics + structured logging |
| Transport | Single WebSocket server | Multiple behind sticky-session LB |

### Connection management

```python
class ConnectionManager:
    def __init__(self, max_concurrent: int):
        self.active: dict[str, CallOrchestrator] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
```

### Health endpoints

```
GET /health  -> {"status": "ok"}
GET /ready   -> {"ready": true, "checks": {"database": true, "tools": true}}
```

## Conversation Design

### System Prompt Structure

```
BASE PROMPT (always present):
  - Persona: warm, professional receptionist for Mitchell & Associates
  - Rules: be concise (1-3 sentences), never guess details, confirm low-confidence
  - Available tools and when to use them
  - Escalation triggers

+ LAW-TYPE FRAGMENT (injected after routing):
  - Employment: unfair dismissal, discrimination, contracts, harassment
    Ask about: employed/terminated, timeline, documentation
  - Tenancy: eviction, deposits, repairs, lease reviews
    Ask about: tenant/landlord, tenancy type, timeline, urgency
```

### Escalation Triggers

1. Caller explicitly asks for a human
2. Out-of-scope legal area (not employment or tenancy)
3. 3+ consecutive misunderstandings
4. Complex multi-party situation
5. Caller expresses frustration or distress

### Booking Flow

1. Collect: name, email, phone, matter type, preferred date/time
2. Confirm ALL details back: "I have you down for a 30-minute employment consultation on Tuesday June 10th at 2pm. Name: John Smith. Email: j-o-h-n at gmail dot com. Is that all correct?"
3. If slot unavailable: "That slot is taken. I have openings at 10am and 3pm the same day, or 2pm the following day. Which works best?"
4. Book and confirm: "You're all set. You'll receive a confirmation at john@gmail.com."

## Build Sequence

### Phase 1: Skeleton (~30 min)
- Project structure, pyproject.toml, config.py, main.py, health endpoint
- Verify: `uvicorn app.main:app` serves /health

### Phase 2: Providers (~1 hr)
- base.py ABCs, whisper.py, ollama.py, piper.py, factory.py
- Test each independently with a WAV file / hardcoded text

### Phase 3: Pipeline (~1.5 hr)
- Pipecat pipeline: orchestrator.py (pipeline factory), services.py (provider creation)
- Browser WebSocket endpoint (ws.py) using FastAPIWebsocketTransport
- Test: browser mic -> server -> basic LLM response -> speaker (no tools yet)

### Phase 4: Conversation + Tools (~1.5 hr)
- state.py, manager.py, prompts.py (with employment + tenancy fragments)
- All 5 tools: routing, extraction (with confidence), booking (with unavailable handling), escalation
- SQLite schema + seed script (2 weeks of slots, some pre-booked)
- Wire tools into orchestrator

### Phase 5: Frontend (~45 min)
- index.html with transcript panel, call controls, state indicator
- app.js with AudioWorklet mic capture + WebSocket + audio playback
- style.css

### Phase 6: Twilio Transport (~1 hr)
- api/twilio.py: POST /twilio/voice webhook + Media Streams WebSocket adapter
- utils/audio.py: mulaw<->PCM conversion, resampling
- Test with Twilio dev phone number

### Phase 7: Polish (~45 min)
- Interruption handling (barge-in)
- Sentence-level TTS streaming from LLM output
- Confidence confirmation loop end-to-end test
- Docker compose, README, .env.example

## Verification

### Local (take-home)
1. `make setup` -> installs deps, downloads models, pulls Ollama model, seeds DB
2. `make run` -> starts backend, opens browser
3. Test scenarios:
   - Employment law -> book consultation -> happy path
   - Tenancy law -> slot unavailable -> offered alternative
   - Ambiguous name/email -> agent confirms via spell-back
   - Out-of-scope -> escalation
   - "Let me speak to someone" -> immediate escalation

### Deployed (phone demo)
1. Deploy to Railway/Fly.io with cloud provider env vars
2. Configure Twilio webhook
3. Call the number, run through same scenarios over phone
4. Record calls for submission

## Honest Tradeoffs

### Local stack limitations
- **STT latency**: faster-whisper processes complete segments (~300-500ms), not streaming. Cloud Deepgram streams interim results during speech.
- **LLM tool use**: Ollama llama3.1 8B sometimes misformats tool calls. We validate and retry, then fall back to structured JSON parsing.
- **TTS quality**: Piper is functional but noticeably synthetic vs ElevenLabs.
- **End-to-end latency**: ~1-2s locally vs ~500-800ms with cloud stack.

### Design tradeoffs
- **No rigid FSM**: LLM-driven flow is more natural but less predictable. Mitigated by strong system prompt constraints.
- **Pipecat framework**: Handles VAD, turn-taking, interruptions, and streaming TTS out of the box. We focus on business logic (tools, prompts, state) instead of audio plumbing.
- **Sentence-level TTS**: Pipecat splits LLM output at sentence boundaries and feeds each to TTS immediately.