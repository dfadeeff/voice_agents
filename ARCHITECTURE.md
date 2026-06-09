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
│   │   │   ├── local_whisper.py     # Local STT (ConversationAware: per-turn decoder bias)
│   │   │   └── local_piper.py       # Local TTS (sentence-pause shaping)
│   │   │
│   │   ├── providers/               # Vendor-agnostic STT/LLM/TTS factories + registries
│   │   │   ├── base.py              # ConversationAware protocol
│   │   │   ├── stt.py               # whisper | deepgram
│   │   │   ├── llm.py               # ollama | openai
│   │   │   └── tts.py               # piper | elevenlabs
│   │   │
│   │   ├── conversation/
│   │   │   ├── flow.py              # Deterministic state machine (next_phase, phase tools)
│   │   │   ├── state.py             # ConversationState dataclass (serializable, incl. `awaiting`)
│   │   │   ├── manager.py           # State per call, phase transitions, next_prompt(), reply parsing
│   │   │   ├── script.py            # compute_prompt(): the scripted question driver (the spine)
│   │   │   ├── phone.py             # Spoken-number normalization
│   │   │   ├── policy.py            # Handoff-request detection, target-person extraction
│   │   │   ├── prompts.py           # System prompt builder (locale-aware)
│   │   │   └── locales/             # Per-language prompt + scripted-line constants
│   │   │       ├── de.py            # German (default) — preamble, phase prompts, SCRIPTED lines
│   │   │       └── en.py            # English
│   │   │
│   │   ├── tools/
│   │   │   ├── registry.py          # Tool name -> callable + JSON schema
│   │   │   ├── route.py             # route_call (intent + legal area, one step)
│   │   │   ├── extraction.py        # capture_caller_details, confirm_caller_detail (+ regex/conf validation)
│   │   │   ├── booking.py           # calendar tools (booking itself is deterministic in the manager)
│   │   │   └── handoff.py           # request_handoff
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
│              │route_call │         │deterministic│   │ book     ││
│              │(LLM)      │         │parse +      │   │(in-code) ││
│              │intent +   │         │regex + STT  │   │          ││
│              │legal_area │         │conf, gated  │   │ SQLite   ││
│              │employment │         │on awaiting  │   │ calendar ││
│              │tenancy    │         │name: 0.92   │   │ slots +  ││
│              │traffic    │         │email: 0.45  │   │ bookings ││
│              │unknown    │         │ -> read back│   │          ││
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

## Conversation Design: Scripted Spine + LLM at the Edges

The call is always in one explicit phase. With a local 7B model that intermittently
drops tool calls and drifts, the **data-collection spine is driven deterministically**:
the state machine knows what it just asked (`state.awaiting`), speaks a **scripted**
question (`manager.next_prompt()` → `script.compute_prompt()`), and parses the caller's
reply against `awaiting` — never against the agent's own previous sentence. On the
scripted phases the LLM is fast-pathed out entirely, so it cannot hallucinate, ask the
wrong thing, skip a field, or fake a booking.

The LLM keeps the two jobs it is genuinely good at and cannot corrupt:
- **ROUTING** — interpreting messy free speech into intent + legal area (keyword fallback backs it up)
- **INFORMATION** — open-ended answers about what the firm handles

```
State machine   = process controller (phases, gates)
Scripted spine  = every data-collection question + deterministic reply parsing
LLM             = routing classification + general-info answers
Tools           = calendar, validators, extraction (cloud/fallback)
```

> The balance is deliberately tilted to deterministic for the **local** model. A cloud
> model (reliable tool-calling) can re-enable LLM-driven capture without touching the spine.

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
| Caller intent + legal area | LLM `route_call` → code validates (keyword fallback) | LLM interprets messy speech; unknown areas escalate, not retry |
| Which question to ask next | Code (`compute_prompt`, gated on `state`) | The spine must not depend on the LLM remembering the flow |
| Reply interpretation (name/email/phone/insurance/slot) | Code, dispatched on `state.awaiting` | Deterministic parsers; reliable on a local 7B |
| Required fields | Code (`flow.py` gates) | Business requirement, not LLM judgment |
| Field confidence | STT scores + regex + code policy | LLM cannot assess audio quality; low conf → read-back |
| Slot availability + booking | Calendar (SQLite) + code | Source of truth is the DB; booking writes are deterministic |
| Booking permission | Code (all fields confirmed) | Safety gate the LLM cannot bypass |
| When to escalate | Code policy + LLM signal | LLM detects distress; code enforces policy |
| Routing / general-info wording | LLM (phase-constrained) | The two open-ended spots; natural language is the LLM's strength |
| Scripted-phase wording | Code (locale `SCRIPTED` templates) | Exact, drift-free; accuracy-critical read-backs must be verbatim |

### Why not a rigid phone tree?

Real callers give multiple details at once ("Hi, I'm Dmitry, I need help with my landlord keeping my deposit"). A rigid "ask name → ask email → ask phone" sequence sounds robotic. Instead:
- The state machine tracks **required fields**
- The LLM extracts **any fields from every turn**
- Code asks **only for missing or uncertain fields**

### State Machine (flow.py)

`next_phase()` is a **projection from accumulated state to phase** — it doesn't step forward, it computes where we should be based on what data has been collected:

```
GREETING ──(turn>=1)──> ROUTING ──(intent + area known)──┐
                          │                              │
              area==unknown / handoff             GENERAL_INFO ── INFORMATION
                          ▼                              │            │ (wants to book)
                     ESCALATION                          └─────┬──────┘
                          ▲                                    ▼
        at ANY point: caller asks for a human,         QUALIFICATION
        3+ misunderstandings, out-of-scope area      matter_type → matter_details
                                                       (traffic: → insurance)
                                                             │
                                                             ▼
                                                          CAPTURE
                                                  name → email → phone
                                                  (email/phone read back)
                                                             │ all confirmed
                                                             ▼
                                                          BOOKING
                                                  offer slots → choose → book
                                                  (declined → offer alternatives)
                                                             │ booked
                                                             ▼
                                                       CONFIRMATION  (read back, goodbye)
```

### Per-Phase Ownership

The eight phases (`CallPhase`). On scripted phases the spoken turn comes from
`compute_prompt()` and the LLM is skipped; only ROUTING and INFORMATION run the LLM.

| Phase | Driver | What happens | Tools available to LLM |
|-------|--------|--------------|------------------------|
| GREETING | Scripted | Greeting spoken on connect | none |
| ROUTING | **LLM** | Classify intent + legal area | `route_call`, `request_handoff` |
| QUALIFICATION | Scripted | Area-confirm + matter_type, then matter_details (or insurance for traffic) | `request_handoff` |
| INFORMATION | **LLM** | Answer general questions about the area | `route_call`, `request_handoff` |
| CAPTURE | Scripted | name → email → phone, with read-back confirmations | `request_handoff` |
| BOOKING | Scripted | Offer slots, match choice, book (deterministic) | `request_handoff` |
| CONFIRMATION | Scripted | Read back the booked slot / callback | none |
| ESCALATION | LLM / Scripted | Callback capture is scripted; out-of-scope escalation is LLM-driven | `capture_caller_details`, `confirm_caller_detail`, `request_handoff` |

`state.awaiting` (one of `area, matter_type, matter_details, insurance, name, name_confirm,
email, email_confirm, phone, phone_confirm, slot, callback_time`) records what the last
scripted question asked for; the reply parser in `manager.add_user_message` dispatches on it.

### Routing disambiguation — never stall in ROUTING

ROUTING is the one LLM-driven phase, so it is also the one place the agent can go
off the rails. A regression call exposed this: the opening *"Mein **Mietvertrag**
wurde **gekündigt** und ich will Frau Hofmann sprechen"* matched **two** areas
(tenancy via *Mietvertrag*, employment via *gekündigt*), so the keyword router
abstained — and the local LLM never called `route_call`. The call **stalled in
ROUTING with `legal_area` unknown**, where the LLM is free-form, and it
**hallucinated the whole call**: a fake appointment ("übermorgen 10:00") and
"Telefonnummer ist notiert" — while the deterministic capture/booking code (which
runs only in CAPTURE/BOOKING) never executed, so **nothing was persisted** (no
name, email, phone entity, or booking row).

The rule that fixes this class of bug:

- **If keyword routing matches ≥2 areas, the agent MUST ask a deterministic
  follow-up question to disambiguate** — it does not guess and does not hand the
  turn to the LLM. `_try_auto_route` records the candidates in `state.area_options`;
  `compute_prompt` then speaks a scripted `disambiguate_area` question
  ("Geht es eher um Arbeitsrecht oder Mietrecht?", `awaiting="area"`), and
  `_handle_area_choice` maps the reply (incl. spoken area names like *Mietrecht*)
  to a single area in code. The call leaves ROUTING deterministically without
  depending on the 7B calling a tool.
- **Safety net:** while no booking is confirmed, the false-booking guard
  (`processors._guard_false_booking`) also blocks **future-tense** fabrications
  like "… wird … einen Termin … buchen", not just "… ist gebucht" — so even if the
  LLM did wander, it cannot voice a booking that did not happen. The genuine offer
  "ich vereinbare gerne einen Beratungstermin" is preserved.

### How a Turn Drives State

```
Caller: "I was unfairly dismissed from my job"
  │
  ▼
manager.add_user_message():
  intent UNKNOWN → keyword fallback (or LLM route_call) sets
  intent=book_consultation, legal_area=employment
  advance_phase() → QUALIFICATION (matter_type missing)
  │
  ▼
manager.next_prompt() → compute_prompt():
  QUALIFICATION + no matter_type → speak SCRIPTED `employment_confirm`,
  set state.awaiting = "matter_type"   (LLM is skipped this turn)
  │
  ▼
Caller: "Yes, a dismissal"
  │
  ▼
add_user_message(): awaiting=="matter_type" → keyword-map → matter_type="dismissal"
  advance_phase() → QUALIFICATION (matter_details missing)
  next_prompt() → SCRIPTED `employment_details`, awaiting="matter_details"
  … → CAPTURE (name → email → phone) → BOOKING (offer/choose/book) → CONFIRMATION
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
    │ Deterministic│  phase: CAPTURE, state.awaiting == "email"
    │ parser       │  _parse_email("… fadejeff at gmail") → "fadejeff@gmail.com"
    │ (manager)    │  1. regex validates email format → OK
    │              │  2. stores entity unconfirmed (email always read back)
    │              │  3. advance_phase() → still CAPTURE (unconfirmed field)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ next_prompt()│  email present + unconfirmed → SCRIPTED confirm_email,
    │ (script.py)  │  awaiting = "email_confirm". The LLM is skipped this turn.
    │              │  "Ich habe notiert: fadejeff@gmail.com — ist das korrekt?"
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ TTS          │  speaks the read-back; reply parsed against "email_confirm"
    └──────────────┘
    (low STT confidence or an unparseable reply → ask_email_not_understood, re-ask)
```

## Provider Abstraction

The pipeline depends on three vendor-agnostic factories (`create_stt`/`create_llm`/
`create_tts`), never on a concrete vendor SDK. Each modality keeps a
**name → builder registry**, so the provider is chosen by env var and adding a
vendor is one registry entry — no caller changes. Vendor SDK imports live inside
each builder, so an unused provider's dependency is never imported.

### Registry (backend/app/providers/)

```
providers/
├── base.py   # ConversationAware protocol (services that take the live conversation)
├── stt.py    # _BUILDERS = {"whisper": ..., "deepgram": ...};  create_stt(settings)
├── llm.py    # _BUILDERS = {"ollama": ...,  "openai": ...};    create_llm(settings)
└── tts.py    # _BUILDERS = {"piper": ...,   "elevenlabs": ...}; create_tts(settings)
```

```python
# stt.py
_BUILDERS: dict[str, SttBuilder] = {"whisper": _build_whisper, "deepgram": _build_deepgram}

def create_stt(settings):
    try:
        return _BUILDERS[settings.stt_provider](settings)
    except KeyError:
        raise ValueError(f"Unknown STT provider {settings.stt_provider!r}") from None
```

The returned services are Pipecat-native — they plug directly into the pipeline and
handle streaming, audio conversion, and frame passing internally. Swapping models is
a one-line `.env` change.

**`ConversationAware`** is a `runtime_checkable` `Protocol` (`set_conversation(...)`).
The orchestrator wires the live conversation into any service that implements it
(local Whisper biases its decoder toward the awaited field); cloud services that
don't implement it are skipped via `isinstance` — no vendor-specific branching in
the pipeline.

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
| State | In-memory dict | Redis (state is already serializable; add a Redis-backed store) |
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
| Caller asks for a human | `policy.is_explicit_handoff_request` (deterministic) or LLM `request_handoff` | `callback_requested=True` (named-person callback) |
| Out-of-scope legal area | `route_call(legal_area="unknown")` | Sets `escalation_requested=True` in route.py |
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

The `request_handoff` tool (and the deterministic handoff path) already records the escalation reason, target person, and all collected state — this is the payload that would be sent to the receiving human agent.

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

**Inter-sentence pause** (`LocalPiperTTSService`)
- Pipecat feeds Piper one sentence per call, and Piper emits no trailing silence, so
  consecutive sentences run together ("…der Kanzlei.Ich nehme…"). The local Piper
  subclass appends ~180ms of silence per synthesized sentence (`PIPER_SENTENCE_PAUSE_MS`).
  Email "." is expanded to " Punkt " before TTS, so addresses are never split — the pause
  is per sentence, not per period. Cloud TTS (ElevenLabs) renders prosody natively.

Production improvements would add:
- Date/time formatting (`2026-06-09T14:00` → `Dienstag, 9. Juni um 14 Uhr`)
- Name spelling normalization (NATO alphabet for confirmation)
- Legal term pronunciation hints

## Honest Tradeoffs

### Local stack limitations
- **STT latency**: faster-whisper processes complete segments (~300-500ms), not streaming. Cloud Deepgram streams interim results during speech.
- **TTS quality**: Piper is functional but noticeably synthetic vs ElevenLabs or Kokoro.
- **End-to-end latency**: ~1-2s locally vs ~500-800ms with cloud stack. LLM inference is the bottleneck.

### Open-ended slots: keyword-first, LLM-assisted
Two slots are genuinely open-ended — **spoken email** and **matter_type** — and keyword/regex is brittle there (it works when the caller echoes an offered option, but misses free phrasings like "ich wurde aus meiner Wohnung geworfen"). For both, the deterministic parse runs first (local default, free, instant) and an **LLM rescue** only fires when it misses: `email_extract.make_email_extractor` reassembles a noisy spoken address, and `matter_classify.make_matter_classifier` maps a free-phrased matter to one of the area's labels. The rescue is implicitly on for the cloud (OpenAI) provider and opt-in on local via `LLM_ASSIST=true`. Safety nets keep the call moving even with no rescue: email skips after 3 misses, and an unrecognised matter is recorded as `"other"` after 2 — and the verbatim answer is always kept in `matter_summary`, so triage never loses information. This is the deliberate split: deterministic for the flow and gates; LLM only for the two slots where language understanding genuinely beats pattern-matching.

### Design tradeoffs
- **Hybrid state machine**: More engineering effort than pure LLM-driven, but provides deterministic, testable flow control for business-critical transitions. The LLM cannot skip fields, bypass confirmation, or forget to escalate.
- **Pipecat framework**: Handles VAD, turn-taking, interruptions, and streaming TTS out of the box. We focus on business logic (tools, prompts, state) instead of audio plumbing.
- **Sentence-level TTS**: Pipecat splits LLM output at sentence boundaries and feeds each to TTS immediately.
- **State as projection**: `next_phase()` computes the correct phase from accumulated data, not from the previous phase. This makes it idempotent and robust to out-of-order tool calls — important when a caller gives name + email + legal issue in a single sentence.

### Conversation state: scope and limits
Within a call, everything captured is held in an explicit, typed `ConversationState`
(`conversation/state.py`): each field is an `ExtractedEntity` with `value`,
`confidence`, `confirmed`, and `source_turn`, alongside `caller_intent`, `legal_area`,
`phase`, `awaiting`, `target_person`, `offered_slots`, `booking_confirmed`, and the
message history. The next question is computed from this state (`next_prompt()` /
`build_system_prompt` injects the missing/unconfirmed fields), so the agent never
re-asks what it already knows — the conversational-continuity principle implemented as
a structured state object rather than raw prompt-chaining. The row is also upserted to
SQLite every turn (`_persist`), so a dropped call keeps its partial record.

Honest scope note (deliberate "ready but not wired" decisions):
- **Per-call, in-memory.** State lives in one `ConversationManager` per WebSocket; it is
  not shared across processes. `to_dict()`/`to_json()` is the serialization seam — moving
  it to a **Redis**-backed store builds on that seam rather than a rewrite.
- **No cross-session memory.** A returning caller starts fresh; there is no caller-history
  lookup by phone number. Cross-session recall (recognise a repeat caller, pull prior
  matter/contact details) is the natural next production step and would build on the same
  `callers` table that already persists every call.
- For this prototype that scope is intentional — per-call state fully satisfies the
  user stories; durable/shared/cross-session state is a scaling concern, not a
  correctness one.