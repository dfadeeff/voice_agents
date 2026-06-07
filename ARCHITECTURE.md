# Voice AI Agent — Architecture

## Context

Inbound voice agent for a law firm. Handles calls end-to-end: greeting, routing by legal area, entity capture with confidence handling, consultation booking, and escalation to humans. Runs locally with zero API keys by default; swaps to cloud providers (Deepgram, OpenAI, ElevenLabs) via env vars for production deployment with Twilio telephony.

## Stack

| Layer | Local (default) | Cloud (production) |
|-------|----------------|---------------------|
| **STT** | faster-whisper (medium, CPU int8) | Deepgram Nova-2 (streaming) |
| **LLM** | Ollama Qwen2.5-7B | OpenAI GPT-4o-mini |
| **TTS** | Piper (de_DE-eva_k-x_low) | ElevenLabs (streaming) |
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
│   │   │   ├── processors.py        # Transcript display + call logging
│   │   │   └── services.py          # Create Pipecat STT/LLM/TTS from config
│   │   │
│   │   ├── conversation/
│   │   │   ├── flow.py              # Deterministic state machine (next_phase, phase tools)
│   │   │   ├── state.py             # ConversationState dataclass (serializable)
│   │   │   ├── manager.py           # Manages state per call, phase transitions
│   │   │   ├── prompts.py           # System prompt builder (locale-aware)
│   │   │   └── locales/             # Per-language prompt constants
│   │   │       ├── de.py            # German (default) — preamble, phase prompts, fragments
│   │   │       └── en.py            # English
│   │   │
│   │   ├── tools/
│   │   │   ├── registry.py          # Tool name -> callable + JSON schema
│   │   │   ├── intent.py            # classify_caller_intent
│   │   │   ├── routing.py           # classify_legal_area
│   │   │   ├── intake.py            # complete_intake
│   │   │   ├── extraction.py        # extract_caller_details + confidence + regex validation
│   │   │   ├── conflict.py          # record_conflict_info (employer + insurance)
│   │   │   ├── additional.py        # record_additional_info
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
│   ├── download_models.py           # Download Piper voice + pull Ollama model
│   └── benchmark_models.py          # Model latency + tool calling benchmark
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
│                   PIPECAT PIPELINE (orchestrator.py)                │
│                                                                    │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌────────┐ │
│  │ Silero   │─>│Whisper STT │─>│ Ollama LLM │─>│ PreTTS    │─>│ Piper  │ │
│  │ VAD      │  │(or Deepgram│  │(or OpenAI) │  │ Sanitizer │  │  TTS   │ │
│  │          │  │  cloud)    │  │            │  │           │  │(or 11L)│ │
│  │ speech   │  │            │  │ -> text +  │  │ strip:    │  │        │ │
│  │ detect + │  │ -> text +  │  │  tool calls│  │ think tags│  │->audio │ │
│  │ endpoint │  │  word conf │  │            │  │ tool names│  │  chunks│ │
│  │          │  │            │  │            │  │ CJK/JSON  │  │        │ │
│  │          │  │            │  │            │  │ false book│  │        │ │
│  └──────────┘  └────────────┘  └──────┬─────┘  └───────────┘  └────────┘ │
│                                           │                       │
│                                     tool calls                    │
│                                           │                       │
│                     ┌─────────────────────┼──────────────────┐    │
│                     │                     │                  │    │
│                     ▼                     ▼                  ▼    │
│              ┌───────────┐         ┌───────────┐     ┌──────────┐│
│              │classify_  │         │extract_   │     │check_    ││
│              │intent /   │         │entities   │     │availabil.││
│              │legal_area │         │+ regex    │     │book_     ││
│              │           │         │+ STT conf │     │consult.  ││
│              │employment │         │           │     │          ││
│              │tenancy    │         │name: 0.92 │     │  SQLite  ││
│              │traffic    │         │email: 0.45│     │  calendar││
│              │unknown    │         │  -> SPELL │     │          ││
│              └─────┬─────┘         └─────┬─────┘     └────┬─────┘│
│                    │                     │                 │      │
│                    └─────────┬───────────┘                 │      │
│                              │                             │      │
│                              ▼                             │      │
│              ┌──────────────────────────┐                  │      │
│              │ CONVERSATION MANAGER     │◄─────────────────┘      │
│              │                          │                         │
│              │  advance_phase() ──────> flow.py: next_phase()     │
│              │  _update_system_prompt() │                         │
│              │                          │                         │
│              │  State machine decides:  │                         │
│              │  ├─ next phase           │                         │
│              │  ├─ available tools      │                         │
│              │  └─ system prompt        │                         │
│              └──────────────────────────┘                         │
│              ┌──────────────────────────┐                         │
│              │ CONVERSATION STATE       │                         │
│              │ (in-memory | Redis)      │                         │
│              │                          │                         │
│              │ phase: CAPTURE           │                         │
│              │ intent: book_consultation│                         │
│              │ legal_area: employment   │                         │
│              │ entities: {name: ✓, ...} │                         │
│              │ booking_confirmed: false  │                         │
│              └──────────────────────────┘                         │
└───────────────────────────────────────────────────────────────────┘
```

## Conversation Design: Hybrid State Machine + LLM

The call is always in one explicit phase. Code controls transitions; the LLM handles language understanding and natural phrasing.

```
State machine = brainstem / process controller
LLM = language and reasoning layer
Tools = calendar, validators, extraction functions
```

### Why not LLM-driven flow?

For a law-firm intake call, business-critical transitions must not depend on the LLM "deciding what feels right":
- Did we route the caller correctly?
- Did we collect name/email/phone?
- Did we confirm uncertain fields?
- Did we check slot availability?
- Did we know when to hand off?

A state machine makes all of this explicit, testable, and deterministic.

### Decision Ownership

| Decision | Owner | Why |
|---|---|---|
| Caller intent | LLM proposes → state manager validates | LLM interprets messy speech; code enforces valid values |
| Legal area | LLM classifies → code checks supported values | Unknown areas trigger escalation, not a retry loop |
| Required fields | Code (`REQUIRED_FIELDS`) | Business requirement, not LLM judgment |
| Field confidence | STT scores + regex + code policy | LLM cannot assess audio quality |
| Slot availability | Calendar tool (SQLite) | Source of truth is the database |
| Booking permission | Code (all fields confirmed) | Safety gate the LLM cannot bypass |
| When to escalate | Code policy + LLM signal | LLM detects distress; code enforces policy |
| Spoken wording | LLM (phase-constrained) | Natural language is what LLMs are good at |

### Why not a rigid phone tree?

Real callers give multiple details at once ("Hi, I'm Dmitry, I need help with my landlord keeping my deposit"). A rigid "ask name → ask email → ask phone" sequence sounds robotic. Instead:
- The state machine tracks **required fields**
- The LLM extracts **any fields from every turn**
- Code asks **only for missing or uncertain fields**

### State Machine (flow.py)

`next_phase()` is a **projection from accumulated state to phase** — it doesn't step forward, it computes where we should be based on what data has been collected:

```
                          ┌───────────┐
                          │ GREETING  │
                          │ "Hello,   │
                          │ how can I │  turn_count >= 1
                          │ help?"    │──────────────────┐
                          └───────────┘                  │
                                                         ▼
                                                   ┌───────────┐
                                                   │ INTENT    │
                                                   │ DETECTION │
                                                   │           │  intent classified
                                                   │ classify_ │──────────────────┐
                                                   │ caller_   │                  │
                                                   │ intent    │                  │
                                                   └───────────┘                  │
                                                                                  ▼
                                                                            ┌───────────┐
                                                                            │ ROUTING   │
                                                                            │           │
                                                                            │ classify_ │
                                                                            │ legal_area│
                                                                            └─────┬─────┘
                                                                                  │
                                           ┌──────────────────────────────────────┤
                                           │                                      │
                              area == UNKNOWN                      area known + intent
                                           │                                      │
                                           ▼                    ┌─────────────────┤
                                     ┌───────────┐              │                 │
                                     │ ESCALATION│    GENERAL_INFO     BOOK_CONSULTATION
                                     │           │              │                 │
                                     │ hand off  │              ▼                 ▼
                                     │ to human  │        ┌───────────┐    ┌───────────┐
                                     └───────────┘        │INFORMATION│    │ INTAKE    │
                                           ▲              │           │    │           │
                                           │              │ answer Qs │    │ follow-up │
                                 at ANY point:            │ offer to  │    │ questions │
                                 - caller asks            │ book      │    │ complete_ │
                                 - 3+ misunderstandings   └───────────┘    │ intake    │
                                 - escalation_requested                    └─────┬─────┘
                                                                                │
                                                                     intake complete
                                                                                │
                                                                                ▼
                                                                          ┌───────────┐
                                                                          │ CAPTURE   │
                                                                          │           │
                                                                          │ name ✓?   │◄─┐
                                                                          │ email ✓?  │  │
                                                                          │ phone ✓?  │──┘
                                                                          └─────┬─────┘
                                                                                │
                                                                     all confirmed
                                                                                │
                                                                                ▼
                                                                          ┌───────────┐
                                                                          │ CONFLICT  │
                                                                          │ CHECK     │
                                                                          │           │
                                                                          │ employer? │
                                                                          │ insurance?│
                                                                          └─────┬─────┘
                                                                                │
                                                                                ▼
                                                                          ┌───────────┐
                                                                          │ADDITIONAL │
                                                                          │   INFO    │
                                                                          │           │
                                                                          │"anything  │
                                                                          │ else?"    │
                                                                          └─────┬─────┘
                                                                                │
                                                                                ▼
                                                                          ┌───────────┐
                                                                          │ BOOKING   │
                                                                          │           │
                                                                          │ check     │
                                                                          │ available │◄─┐
                                                                          │ book slot │  │
                                                                          └─────┬─────┘  │
                                                                                │        │
                                                                     slot unavailable?───┘
                                                                                │
                                                                       booking confirmed
                                                                                │
                                                                                ▼
                                                                         ┌────────────┐
                                                                         │CONFIRMATION│
                                                                         │            │
                                                                         │ read back  │
                                                                         │ all details│
                                                                         └──────┬─────┘
                                                                                │
                                                                                ▼
                                                                          ┌───────────┐
                                                                          │ FAREWELL  │
                                                                          └───────────┘
```

### Per-Phase LLM Constraints

Each phase gives the LLM a narrow, specific task — not the full call flow:

| Phase | LLM's job | Available tools |
|-------|-----------|-----------------|
| GREETING | Greet warmly, ask how to help | none |
| INTENT_DETECTION | Classify: general info or consultation? | classify_caller_intent |
| ROUTING | Classify legal area (employment/tenancy/traffic) | classify_legal_area |
| INTAKE | Area-specific follow-up questions, summarize issue | complete_intake |
| INFORMATION | Answer questions about the legal area | classify_caller_intent (to switch to booking) |
| CAPTURE | Extract caller details, confirm uncertain ones | extract_caller_details |
| CONFLICT_CHECK | Ask about employer and legal insurance | record_conflict_info |
| ADDITIONAL_INFO | "Anything else?" before booking | record_additional_info |
| BOOKING | Help caller find and book a slot | check_availability, book_consultation |
| CONFIRMATION | Read back all booking details | none |
| ESCALATION | Explain handoff, be reassuring | escalate_to_human |
| FAREWELL | Thank caller, wish well | none |

### How Tool Calls Drive State

```
Caller: "I was unfairly dismissed from my job"
  │
  ▼
LLM calls classify_caller_intent(intent="book_consultation")
  │
  ▼
Tool handler stores intent → advance_phase() computes:
  intent known, legal_area unknown → ROUTING
  │
  ▼
System prompt updates to ROUTING phase:
  "Determine the caller's legal area. Call classify_legal_area."
  │
  ▼
LLM calls classify_legal_area(legal_area="employment")
  │
  ▼
Tool handler stores area → advance_phase() computes:
  intent=book, area=employment, intake not complete → INTAKE
  │
  ▼
System prompt updates to INTAKE phase:
  "Capture a brief summary of the caller's employment issue."
```

### Confidence-Aware Entity Extraction

Two layers of validation (defense-in-depth):

**Layer 1: STT confidence (per-word)**
```
Caller: "My name is Siobhan McNally"
faster-whisper: "shivan mcnally" — confidence: [0.62, 0.71]
Below 0.7 threshold → flag for confirmation
```

**Layer 2: Regex validation (format)**
```
email: must match basic email pattern
phone: must have 7-20 digits
Invalid format → reject, ask caller to repeat
```

The state machine refuses to advance from CAPTURE to BOOKING until all required fields pass both checks and are confirmed.

### Single Turn Data Flow

```
Caller speaks: "My email is d fadeev at gmail maybe no wait fadejeff at gmail"
         │
         ▼
    ┌─────────┐
    │ Silero  │  detects speech end (700ms silence)
    │ VAD     │
    └────┬────┘
         │
         ▼
    ┌──────────────┐
    │ STT          │  "my email is d fadeev at gmail maybe no wait fadejeff at gmail"
    │ (whisper)    │  word_conf: [0.95, 0.92, ..., 0.61, ...]
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ LLM          │  phase: CAPTURE → constrained to extraction task
    │ (Qwen2.5-7B) │  calls: extract_caller_details(email="fadejeff@gmail.com")
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Tool handler │  1. regex validates email format → OK
    │ (extraction) │  2. STT confidence for "fadejeff" → 0.61 (below 0.7)
    │              │  3. stores entity, marks needs_confirmation
    │              │  4. advance_phase() → still CAPTURE (unconfirmed field)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ LLM          │  system prompt says: "Email needs confirmation. Read back."
    │ (response)   │  generates: "I heard fadejeff@gmail.com. Is that correct?"
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ TTS          │  speaks response to caller
    └──────────────┘
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

All services are Pipecat-native — they plug directly into the pipeline and handle streaming, audio format conversion, and frame passing internally. Swapping models is a one-line `.env` change.

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

### Health endpoints

```
GET /health  -> {"status": "ok"}
GET /ready   -> {"ready": true, "checks": {"database": true, "slots_available": true, "tools": true}}
```

## Escalation Triggers

The agent escalates to a human when any of these conditions are met:

| Trigger | Detection | Implementation |
|---|---|---|
| Caller asks for a human | LLM calls `escalate_to_human` | `reason: caller_requested_human` |
| Out-of-scope legal area | `classify_legal_area(area="unknown")` | Sets `escalation_requested=True` in routing.py |
| 3+ consecutive misunderstandings | State machine checks `misunderstanding_streak >= 3` | `next_phase()` returns ESCALATION |
| Complex multi-party situation | LLM judgment | `reason: complex_situation` |
| Caller frustrated or distressed | LLM judgment | `reason: caller_frustrated` |
| Caller asks for legal advice | Prompt instructs refusal + offer to connect | LLM explains boundary, may escalate |
| Booking tool failure | Calendar service error | Tool returns error, LLM offers human help |

## Production Warm Transfer Design

In production, escalation triggers a warm transfer via Twilio:

```
Agent decides to escalate
  │
  ▼
Build handoff summary:
  caller name, legal area, issue description,
  collected entities, call duration, escalation reason
  │
  ▼
Dial human recipient (lawyer / intake team)
  │
  ├── Human answers → play/display summary → bridge caller
  ├── No answer (30s) → return to caller, offer voicemail/callback
  ├── Busy (SIP 486) → try backup recipient or schedule callback
  └── Timeout (SIP 408) / Unavailable (SIP 480) → try backup route
```

The `escalate_to_human` tool already builds `context_for_human` with all collected state — this is the payload that would be sent to the receiving human agent.

## TTS Preprocessing

Two layers of text sanitization in the pipeline:

**Layer 1: PreTTSSanitizer** (between LLM and TTS, chunk-level)
- Strips `<think>` tags (stateful across chunks — handles tags split across frames)
- Removes CJK characters (Qwen sometimes emits Chinese)
- Strips tool names and JSON leaks from LLM output
- Blocks false booking language ("Termin gebucht") unless `booking_confirmed` is true

**Layer 2: AgentTextProcessor** (after TTS, sentence-level safety net)
- **Phone numbers**: digit-by-digit expansion (`+49 151 9823` → `plus 4, 9, 1, 5, 1, 9, 8, 2, 3`)
- **Email addresses**: locale-aware expansion (`@` → `at`, `.` → `Punkt` in German, `dot` in English)
- Second pass of tool name stripping and false booking guard
- Logs cleaned text to transcript and sends to frontend

Production improvements would add:
- Date/time formatting (`2026-06-09T14:00` → `Dienstag, 9. Juni um 14 Uhr`)
- Name spelling normalization (NATO alphabet for confirmation)
- Legal term pronunciation hints

## Honest Tradeoffs

### Local stack limitations
- **STT latency**: faster-whisper processes complete segments (~300-500ms), not streaming. Cloud Deepgram streams interim results during speech.
- **TTS quality**: Piper is functional but noticeably synthetic vs ElevenLabs or Kokoro.
- **End-to-end latency**: ~1-2s locally vs ~500-800ms with cloud stack. LLM inference is the bottleneck.

### Design tradeoffs
- **Hybrid state machine**: More engineering effort than pure LLM-driven, but provides deterministic, testable flow control for business-critical transitions. The LLM cannot skip fields, bypass confirmation, or forget to escalate.
- **Pipecat framework**: Handles VAD, turn-taking, interruptions, and streaming TTS out of the box. We focus on business logic (tools, prompts, state) instead of audio plumbing.
- **Sentence-level TTS**: Pipecat splits LLM output at sentence boundaries and feeds each to TTS immediately.
- **State as projection**: `next_phase()` computes the correct phase from accumulated data, not from the previous phase. This makes it idempotent and robust to out-of-order tool calls — important when a caller gives name + email + legal issue in a single sentence.