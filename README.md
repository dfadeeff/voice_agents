# Voice AI Agent for Law Firms

Inbound voice agent that handles calls end-to-end: greeting, routing by legal area (employment/tenancy), entity capture with confidence handling, consultation booking, and human escalation.

This prototype follows the production-standard cascaded STT → LLM → TTS architecture, but uses local-first components so the project can run without API keys. In production, I would swap in streaming STT, streaming TTS, WebRTC/SIP transport, scalable state storage, and full observability.

### What works in the prototype

- Browser-based local voice call (mic → agent → speaker)
- Full STT → LLM → TTS streaming pipeline
- Employment/tenancy routing with unknown-area escalation
- Structured contact capture (name, email, phone) with confidence-based confirmation
- SQLite calendar with slot checking and alternatives for unavailable times
- Human callback/handoff path (caller request, out-of-scope area, repeated misunderstandings)
- Legal advice boundary enforcement — agent refuses to assess cases
- Call transcript logging to JSON
- TTS preprocessing for phone numbers (digit-by-digit) and emails (spoken form)

### Takehome story coverage

| Required story | Prototype behavior | Concrete evidence |
|---|---|---|
| Routing across law types | Routes employment and tenancy issues, then asks one area-specific qualification question | `conversation/flow.py`, `tools/route.py`, scenario tests |
| Booking a consultation | Requires confirmed contact details, checks SQLite availability, offers alternatives, and only confirms a successful booking | `tools/booking.py`, `services/calendar.py`, unavailable-slot tests |
| Reliable detail capture | Local Whisper supplies segment confidence; low-confidence names require confirmation; email and phone always require read-back confirmation | `pipeline/local_whisper.py`, `tools/extraction.py` |
| Knowing when to hand off | Explicit person requests and unsupported/complex cases enter escalation deterministically. The local demo records a callback request and context; it does not pretend to transfer live | `conversation/policy.py`, `tools/handoff.py` |

### What is intentionally simplified

- Local STT (faster-whisper) processes complete utterances, not streaming interim results
- Piper TTS voice is functional but noticeably synthetic
- No real phone carrier required — browser mic/speaker only (Twilio adapter exists but needs cloud deployment)
- Single-process backend (no horizontal scaling)
- Local LLM tool calling quality depends on model — best with Qwen family

| Mode | Purpose | Stack |
|---|---|---|
| **Local demo** (default) | Zero-key evaluation | faster-whisper + Ollama/Qwen + Piper |
| **Best demo quality** | Better voice and tool reliability | Whisper/Deepgram + GPT-4o-mini + ElevenLabs |
| **Production** | Scale, reliability, observability | Streaming STT + low-latency LLM + streaming TTS + telephony + monitoring |

## Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** — local LLM runtime
- **A microphone** — for browser-based voice calls

### Install Ollama

**macOS:**
```bash
brew install ollama
```

**Or download directly:** [ollama.com/download](https://ollama.com/download)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server (keep it running in a separate terminal):
```bash
ollama serve
```

Pull the default model:
```bash
ollama pull qwen2.5:7b
```

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/dfadeeff/voice_agents.git && cd voice_agents

# 2. Install Python dependencies
pip install -e "backend/.[dev]"

# 3. Download models (Piper TTS voice + Ollama qwen2.5:7b + NLTK tokenizer)
cd backend && python3 scripts/download_models.py && cd ..

# 4. Seed the appointment calendar
cd backend && python3 scripts/seed_calendar.py && cd ..

# 5. Start the server
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser and click "Start Call".

Or use `make`:
```bash
make setup   # steps 2-4 above
make run     # step 5
```

## Choosing a Local Model

For a real-time phone call agent, the priority is **latency first, then tool calling quality, then reasoning**. A smaller model that reliably produces short, correct responses under roughly one second of perceived latency is often better than a larger model that creates several seconds of silence.

Set the model in `.env` via `OLLAMA_MODEL`:

| Model | Size | Latency | Tool calling | Recommended for |
|-------|------|---------|-------------|----------------|
| **`qwen2.5:7b`** | 4.7 GB | Fast (0.8s) | Good with guardrails | **Best local default** |
| `qwen3:4b` | 2.6 GB | Fastest (0.5s) | Excellent | Speed-constrained setups |
| `qwen3:8b` | 5.2 GB | Slow (4.3s) | Excellent | Not recommended (thinking tokens cause silence) |
| `llama3.2:3b` | 2.0 GB | Fastest | Fair | Speed-constrained devices |
| `llama3.1:8b` | 4.9 GB | Fast | Unreliable | Not recommended (often dumps JSON as text) |

**qwen2.5:7b is the default.** In local benchmarks it averages 0.8s latency — much faster than qwen3:8b because qwen3 emits thinking tokens that create dead air. Qwen 2.5 can still make tool or language mistakes, so business-critical transitions are code-controlled and a pre-TTS safety layer drops internal JSON/tool text. Run `make benchmark` to test on your machine.

In my local tests, **llama3.1:8b was unreliable** for this project's tool-calling flow: it sometimes emitted JSON as normal text or called tools with invalid arguments.

To switch models:
```bash
# Pull the model
ollama pull qwen2.5:7b

# Set in .env
OLLAMA_MODEL=qwen2.5:7b
```

Tool calling is enabled by default (`USE_TOOLS_LOCAL=true` in `.env`). This enables the full structured tool pipeline locally: intent classification, legal area routing, entity extraction with confidence scoring, availability checking, and booking. If a local model outputs JSON as speech or misuses tools, set `USE_TOOLS_LOCAL=false` to fall back to a purely conversational mode.

## Cloud Mode (Optional)

For the best quality (GPT-4o handles tool calling perfectly), switch providers via env vars — no code changes:

```bash
# .env — switch any or all providers
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your-key

LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini

TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your-key
```

When `LLM_PROVIDER=openai`, the full tool-calling pipeline activates automatically: intent classification, legal area routing, entity extraction with confidence scoring, availability checking, and consultation booking.

You can mix local and cloud freely — e.g. keep Whisper STT local but use OpenAI for LLM and ElevenLabs for TTS.

## What Happens When You Call

1. **Greeting** — the agent answers as a law firm receptionist
2. **Routing** — identifies your legal area (employment law or tenancy law) and adapts questions accordingly
3. **Capture** — collects your name, email, phone number. If STT confidence is low (noisy line, unusual name), the agent asks you to spell it back rather than guessing
4. **Booking** — checks the calendar for available consultation slots. If your preferred time is taken, offers alternatives
5. **Escalation** — if you ask for a human, the issue is out of scope, or the agent cannot understand you after 3 attempts, it records a callback/handoff request with context

## Legal Advice Boundary

The agent is a receptionist/intake assistant, not a lawyer. It does not assess legal merits, predict outcomes, recommend legal strategy, or provide legal advice. If the caller asks for legal advice ("Do I have a strong case?", "Will I win?"), the agent explains the boundary and offers to book a consultation or transfer to a human.

This is enforced at two levels:
- **System prompt**: every phase includes "NEVER give legal advice or opinions on cases"
- **Escalation tool**: the LLM can call `escalate_to_human` with reason `caller_frustrated` or `complex_situation` when the caller pushes for advice

## TTS Pronunciation

Raw structured data sounds wrong when read aloud by TTS. A preprocessing layer (`processors.py`) reformats agent text before it reaches the TTS engine:

| Raw text | TTS receives |
|---|---|
| `+49 151 9823 4567` | `plus 4, 9, 1, 5, 1, 9, 8, 2, 3, 4, 5, 6, 7` |
| `fadejeff@gmail.com` | `fadejeff at gmail dot com` |

This prevents phone numbers from being read as natural numbers ("nine million eight hundred...") and emails from being garbled.

## Call Logging

Every call is logged to `logs/{call_id}.json` with full transcription:

```json
{
  "call_id": "abc-123",
  "start_time": "2026-06-05T21:33:47Z",
  "end_time": "2026-06-05T21:34:09Z",
  "transcript": [
    {"role": "agent", "text": "Hello! Welcome to our law firm.", "timestamp": "..."},
    {"role": "user", "text": "I need help with an employment issue.", "timestamp": "..."},
    {"role": "agent", "text": "I'd be happy to help. Can you tell me more about your situation?", "timestamp": "..."}
  ]
}
```

Logs are excluded from git (`.gitignore`). In production, these would go to a database.

## Model Benchmarks

Model selection is data-driven. The benchmark script tests each model on 3 scenarios that matter for phone calls: natural greeting, greeting with tool schemas present, and structured tool calling after conversation context.

```bash
# Run benchmark on all installed models
make benchmark

# Or test specific models
cd backend && python3 scripts/benchmark_models.py qwen2.5:7b qwen3:4b llama3.1
```

Results are saved to `benchmarks/`:
- `benchmark_results.json` — machine-readable (for CI/CD pipelines)
- `benchmark_results.md` — markdown table (for PRs and documentation)

### CI/CD Integration

The benchmark runs automatically via GitHub Actions (`.github/workflows/benchmark.yml`):
- **On push to `main`** — when benchmark script or prompts change
- **Manual trigger** — `Actions → Model Benchmark → Run workflow` with optional model list
- Results are uploaded as artifacts and commented on the commit

### Latest Results

These results are from one local run and are hardware- and version-specific. Run `make benchmark` to reproduce on your machine.

| Model | Greeting | Greeting + Tools | Tool Call | Avg Latency | Verdict |
|-------|----------|-----------------|-----------|-------------|---------|
| qwen2.5:7b | PASS | PASS | PASS | 0.81s | **Recommended (default)** |
| qwen3:4b | PASS | PASS | PASS | 0.48s | Recommended (fastest) |
| qwen3:8b | PASS | PASS | PASS | 4.28s | Not recommended (thinking tokens) |
| llama3.2:3b | PASS | WARN | PASS | 0.44s | Usable (tool calling sometimes unreliable) |
| llama3.1:8b | PASS | PASS | FAIL | 0.89s | Not recommended (outputs JSON as text) |

## Stack

All local, all free:

| Layer | Technology | What it does |
|-------|-----------|--------------|
| **Orchestration** | [Pipecat](https://github.com/pipecat-ai/pipecat) | Pipeline wiring, VAD, turn-taking, interruptions, streaming |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + Pipecat | Speech-to-text with segment confidence and domain bias |
| **LLM** | [Ollama](https://ollama.com) + Qwen 2.5 | Conversation, tool calling, routing decisions |
| **TTS** | [Piper](https://github.com/rhasspy/piper) (via Pipecat) | Text-to-speech, German Eva voice |
| **VAD** | Silero (via Pipecat) | Voice activity detection, endpointing |
| **Transport** | WebSocket + protobuf | Browser mic/speaker over FastAPI WebSocket |
| **DB** | SQLite | Appointment calendar with 560 seeded slots |

## Project Structure

```
voice_agent/
├── backend/
│   └── app/
│       ├── main.py                 # FastAPI app, lifespan
│       ├── config.py               # Pydantic Settings (env-driven)
│       ├── api/
│       │   ├── ws.py               # WebSocket /ws/call/{id} (browser)
│       │   ├── twilio.py           # Twilio Media Streams (phone calls)
│       │   └── health.py           # GET /health, /ready
│       ├── pipeline/
│       │   ├── orchestrator.py     # Pipecat pipeline + tool registration
│       │   ├── local_whisper.py    # Local STT + confidence + domain bias
│       │   ├── local_piper.py      # Local TTS + sentence-pause shaping
│       │   └── processors.py       # Safety, transcript display + logging
│       ├── providers/              # Vendor-agnostic STT/LLM/TTS factories + registries
│       │   ├── base.py             # ConversationAware protocol
│       │   ├── stt.py / llm.py / tts.py  # name → builder registries
│       ├── conversation/
│       │   ├── state.py            # Serializable call state
│       │   ├── manager.py          # State management + deterministic policies
│       │   ├── policy.py           # Safety-critical handoff detection
│       │   └── prompts.py          # System prompt + law-area fragments
│       ├── tools/
│       │   ├── registry.py         # Tool name -> handler + JSON schema
│       │   ├── route.py            # route_call: intent + legal area
│       │   ├── extraction.py       # capture + explicit confirmation
│       │   ├── booking.py          # check_availability, book_consultation
│       │   └── handoff.py          # truthful callback/handoff request
│       ├── services/
│       │   └── calendar.py         # SQLite: slots, bookings
│       └── models/
│           └── schemas.py          # Dataclasses: CallPhase, LegalArea, etc.
├── frontend/
│   ├── index.html                  # Single page, no build step
│   ├── style.css
│   ├── app.js                      # WebSocket client, transcript UI
│   └── components/
│       ├── protobuf.js             # Pipecat frame encoder/decoder
│       └── audio.js                # AudioWorklet mic capture + playback
├── scripts/
│   ├── download_models.py          # Download Piper voice + pull Ollama model
│   ├── seed_calendar.py            # Populate 2 weeks of appointment slots
│   └── benchmark_models.py         # Model latency + tool calling benchmark
├── benchmarks/                     # Benchmark output (JSON + markdown)
├── logs/                           # Call transcripts (auto-created, gitignored)
├── .github/workflows/
│   └── benchmark.yml               # CI: model benchmark on push/manual
├── ARCHITECTURE.md                 # Detailed architecture + diagrams
├── Makefile
└── .env.example
```

## Additional Questions

See [QUESTIONS.md](QUESTIONS.md) for detailed answers to the four takehome questions:
1. Real-Time Latency & Pipeline Design
2. Turn-Taking, Interruptions & Audio Robustness
3. Iteration, Scaling & Health Tracking
4. Telephony, Warm Transfer & Failure Handling

Each answer references specific files and code paths in the repo.

## Design Decisions & Trade-offs

### Why Pipecat (not a custom pipeline)

**Decision:** Use Pipecat for all audio orchestration.

**Trade-off:** Adds a dependency (~50 transitive packages) but eliminates thousands of lines of custom audio code.

**Why:** Pipecat handles VAD (Silero), turn-taking, interruption/barge-in, streaming TTS sentence chunking, and transport abstraction out of the box. Building this from scratch would take weeks and produce a worse result. Our code only handles business logic — tools, prompts, and conversation state. When Pipecat's API changed between versions (e.g., `create_context_aggregator()` removed in 1.3.0, `PipelineRunner` deprecated for `WorkerRunner`), the migration was straightforward because our custom code is minimal.

### Why Qwen 3 (not llama3.1)

**Decision:** Default to `qwen2.5:7b` for the local LLM. Optimize for latency + tool calling, not raw intelligence.

**Trade-off:** qwen2.5:7b is 5x faster than qwen3:8b in local benchmarks (0.8s vs 4.3s) because qwen3 models generate thinking tokens that create dead air on the phone line.

**Why:** For a real-time voice agent, the ranking is: latency > tool calling reliability > reasoning depth. qwen2.5:7b provides reliable tool calling without the thinking-token overhead of qwen3. The Qwen family was specifically optimized for agentic/tool-calling use cases (per Qwen's model card and Docker's local tool-calling benchmarks).

In my local tests with Ollama's OpenAI-compatible API, llama3.1:8b showed these failure modes:
1. **Outputs raw JSON as text** — e.g., `{"name": "greet", "parameters": {}}` spoken aloud by TTS
2. **Calls wrong tools with garbage arguments** — e.g., `extract_caller_details` with `{"debug_mode": true}` on the first turn
3. **Skips greeting entirely** — calls a tool immediately with empty content instead of speaking first

These were observed via direct API testing (`curl` to Ollama's `/v1/chat/completions`). The behavior may vary with different Ollama versions or quantizations.

### Local vs Cloud LLM: tool toggle

**Decision:** Enable tool calling by default (`USE_TOOLS_LOCAL=true`). Disable with `USE_TOOLS_LOCAL=false` if the local model is unreliable with tools.

**Trade-off:** Local mode with tools enabled runs the full structured pipeline (routing, extraction, booking). With tools disabled, the agent still has a natural conversation but doesn't execute structured business logic. Cloud OpenAI mode always has the most reliable tool execution.

**Why:** A single flag in `ws.py` controls this:
```python
use_tools = settings.llm_provider != "ollama" or settings.use_tools_local
```
This flows into `orchestrator.py` which either registers tool handlers + tool-aware prompt, or uses a simpler conversational prompt with no tool schemas. All 6 tools and their handlers stay intact in the codebase — nothing is deleted, just not wired in. Switching between modes is a one-line `.env` change.

### Two system prompts

**Decision:** Maintain separate prompts for tool-calling mode (`SYSTEM_PROMPT_TOOLS`) and conversational mode (`SYSTEM_PROMPT_LOCAL`).

**Trade-off:** Two prompts to maintain instead of one.

**Why:** Small local models are extremely sensitive to prompt content. The tool-calling prompt references tool names and behavior (`"When extract_caller_details returns needs_confirmation..."`) — if the model can't actually use tools, this confuses it into outputting JSON as text. The local prompt is purely conversational with no tool references, so the model focuses on natural speech.

### Piper TTS (not Coqui, not Bark)

**Decision:** Use Piper for local text-to-speech.

**Trade-off:** Piper sounds robotic compared to ElevenLabs or even Bark, but it's fast and reliable.

**Why:** Piper is the only local TTS that Pipecat supports natively (built-in `PiperTTSService`). It runs in ~50ms per sentence, produces consistent output, and doesn't need a GPU. Bark and Coqui XTTS produce better-sounding speech but take 2-10x longer (unacceptable for real-time voice) and would require a custom Pipecat service integration — exactly the kind of custom code we're avoiding.

### faster-whisper STT (not Whisper.cpp, not Vosk)

**Decision:** Use faster-whisper for local speech-to-text.

**Trade-off:** Processes complete utterances rather than streaming interim results like cloud Deepgram. On the development laptop, an 8-second German sample took roughly 8.5 seconds with `medium`; `base` was faster but materially less accurate, while `large-v3` was more accurate but too slow for the demo.

**Why:** `LocalWhisperSTTService` extends Pipecat's faster-whisper integration with legal-domain hotwords, internal VAD trimming, and segment confidence. That confidence feeds the confirmation policy without asking the LLM to judge audio quality.

### Confidence-aware entity extraction

**Decision:** Use STT segment confidence plus field-specific policy to decide when to confirm captured details.

**Trade-off:** Adds complexity to the extraction flow (confidence threshold, confirmation loop) but prevents booking with wrong details.

**Why:** The hardest problem in voice agents is getting contact details right. Names below the confidence threshold require confirmation; email and phone always require explicit read-back confirmation. The LLM receives the tool instruction, but code decides whether confirmation is mandatory.

### Hybrid state machine + LLM (not pure LLM-driven)

**Decision:** Code controls conversation flow via a deterministic state machine (`flow.py`). The LLM handles language understanding and natural phrasing within per-phase constraints.

This is not a rigid phone tree and not a fully autonomous LLM agent. It is a workflow-controlled voice agent. The state manager owns business-critical transitions, while the LLM fills slots, classifies intent, routes legal areas, and phrases responses naturally.

**Trade-off:** More engineering effort than letting the LLM improvise, but provides testable, deterministic flow control for business-critical transitions.

| Decision | Owner |
|---|---|
| Caller intent classification | LLM proposes, state manager validates |
| Legal area routing | LLM classifies, code checks supported values |
| Which fields are required | Code (`REQUIRED_FIELDS` in flow.py) |
| Whether a field is confirmed | STT confidence + regex validation + code policy |
| Slot availability | Calendar tool (SQLite query) |
| Whether booking is allowed | Code (all fields confirmed) |
| When to escalate | Code policy, with LLM signal |
| What words to say | LLM (within phase-specific prompt constraints) |

**Why:** For a law-firm intake call, transitions like "did we confirm the email?", "is this slot available?", and "should we escalate?" must not depend on the LLM "deciding what feels right." The state machine (`next_phase()`) projects accumulated state to the correct phase. Each phase gives the LLM a narrow task via a phase-specific system prompt — not the full call flow. Tool handlers update state and trigger `advance_phase()`, which recomputes the phase and updates the prompt. The LLM never sees the overall flow; it only sees its current task. This makes the agent predictable while still sounding natural — the LLM phrases responses freely, but code decides what to ask and when to move on.

### Provider abstraction via env vars

**Decision:** Swap STT/LLM/TTS providers by changing environment variables, no code changes.

**Trade-off:** Requires maintaining a builder per vendor. But the pipeline depends only on the `create_stt/llm/tts` factories — adding a provider is one registry entry, with no caller changes.

**Why:** The project requirement is local-first with zero API keys, but production would use cloud providers. Each modality in `app/providers/` keeps a name → builder registry; the factory looks the provider up by env var (and raises a clear error on an unknown name):
```python
# app/providers/stt.py
_BUILDERS = {"whisper": _build_whisper, "deepgram": _build_deepgram}
# STT_PROVIDER=whisper → local Whisper;  STT_PROVIDER=deepgram → cloud Deepgram
```
You can mix freely — e.g., local Whisper STT + cloud OpenAI LLM + local Piper TTS. A service can opt into per-turn context by implementing the `ConversationAware` protocol (`base.py`); the orchestrator wires it in via `isinstance`, so there's no vendor branching in the pipeline.

### WebSocket text side-channel (not protobuf)

**Decision:** Send transcription text as JSON via `websocket.send_json()` alongside Pipecat's binary protobuf audio stream.

**Trade-off:** Two message formats on the same WebSocket (JSON text + protobuf binary) instead of one unified protocol.

**Why:** Pipecat's `ProtobufFrameSerializer` uses exact `type()` checks (not `isinstance`) and only serializes `TextFrame` and `AudioRawFrame`. Our `AggregatedTextFrame` (which carries the LLM's response) extends `TextFrame` but doesn't match the exact type check, so it never reaches the frontend via protobuf. Similarly, `TranscriptionFrame` is consumed by the `UserAggregator` before it reaches the output transport. Custom processors (`TranscriptProcessor`, `AgentTextProcessor`) intercept these frames and send them as JSON directly. The frontend handles both: `typeof event.data === "string"` for JSON, `instanceof ArrayBuffer` for audio.

### Call logging to JSON files

**Decision:** Write call transcripts to `logs/{call_id}.json` files.

**Trade-off:** Not queryable like a database. But simple, zero-dependency, and easy to inspect.

**Why:** For a demo/take-home, JSON files are sufficient to verify calls are being recorded and transcribed. The `CallLogger` class accumulates entries during the call and writes them on disconnect. In production, these would go to a database (the SQLite infrastructure is already there for calendar data). Logs are gitignored.

## Privacy

This prototype stores transcripts locally in `logs/` for debugging and evaluation. In production, legal intake calls may contain sensitive personal data (names, contact details, legal situations), so transcripts and recordings should be encrypted at rest, access-controlled, retained only as long as necessary, and redacted where possible. Logs are gitignored and never include secrets or API keys.

## Production Telephony

In production, the agent connects to real phone lines via Twilio Media Streams (adapter exists in `api/twilio.py`). The Pipecat pipeline is transport-agnostic — the same business logic handles both browser WebSocket and Twilio calls.

**Warm transfer design**: On escalation, the agent would:
1. Create a short handoff summary (caller name, legal area, issue description, collected details)
2. Dial the human recipient
3. Play or display the summary to the human
4. Bridge the caller only after the human accepts

**Failure handling**:
- Human doesn't pick up → return to caller, offer voicemail or callback
- Human line busy → try backup recipient or schedule callback
- Bridge drops → keep caller connected, apologize, retry or collect callback number
- SIP errors (486 Busy, 408 Timeout, 480 Unavailable) → mark transfer status, try backup route

## Evaluation and Health Tracking

**Offline evaluation**: A suite of scenario tests (`tests/test_scenarios.py`) covers the core user stories: routing, booking, low-confidence capture, unavailable slots, handoff, and legal-advice boundary. Each scenario has an expected final state and assertions on state machine behavior.

**Online monitoring** (production): I would track:
- Call completion rate and booking conversion rate
- Escalation rate and fallback rate
- P50/P95 time-to-first-audio per component (STT, LLM, TTS)
- STT confidence distribution
- Tool-call failure rate
- Legal-advice boundary trigger count
- Repeated-question rate (signal of poor understanding)

Low-confidence, failed, or escalated calls would be sampled for human review and used to update prompts, validators, and workflow rules.

## Troubleshooting

**"Ollama not found"** — Make sure `ollama serve` is running in a separate terminal.

**`ollama pull` fails with "connection reset by peer"** — This is an IPv6 connectivity issue with Cloudflare's CDN. Fix: disable IPv6 temporarily (`sudo networksetup -setv6off Wi-Fi`), pull the model, then re-enable (`sudo networksetup -setv6automatic Wi-Fi`). Also: never run multiple `ollama pull` commands simultaneously — they compete for bandwidth and both fail.

**Slow first response** — The first call loads Whisper + Ollama models into memory. Subsequent calls are faster.

**Microphone not working** — Chrome requires HTTPS for mic access on non-localhost origins. On localhost it works over HTTP.

**Agent outputs JSON instead of speaking** — Your local model may not support tool calling reliably. Switch to qwen2.5:7b (`OLLAMA_MODEL=qwen2.5:7b` in `.env`), or set `USE_TOOLS_LOCAL=false` for conversational mode.

**NLTK SSL warning** — Harmless. Appears on macOS when Python's SSL certificates aren't installed. Run: `/Applications/Python 3.11/Install Certificates.command`

## Honest Limitations

The prototype is production-shaped, not production-grade. It runs locally with zero API keys to make evaluation easy, but production would require the upgrades described in the table at the top.

- **STT latency**: faster-whisper processes complete utterances and can take several seconds on CPU. Cloud Deepgram would stream interim results during speech.
- **TTS quality**: Piper is functional but noticeably synthetic compared to ElevenLabs. This is the most obvious "not production" tell in a demo.
- **End-to-end latency**: hardware and utterance dependent; local Whisper is currently the largest latency contributor. A cloud streaming stack would be substantially faster.
- **Local tool calling**: Works best with Qwen models. llama3.1:8b was unreliable in my tests — outputs raw JSON text instead of using the tool calling API. Results may vary with different Ollama versions.
- **Single-process**: One Uvicorn worker handles all calls. Production would use multiple workers behind a load balancer, with Redis for shared state.
- **Confidence handling**: The prototype uses segment confidence plus field-specific confirmation policy. Production would use richer word/alternative confidence, email normalization, and repeated-confirmation analytics.
- **Human handoff**: The browser demo records a callback/handoff request and summary. It does not bridge a live phone call.

## Outlook

The current architecture (STT → LLM → TTS) is the industry standard for voice agents, but the field is evolving fast. Three directions worth watching:

### Real-time Audio LLMs

Models like [Ultravox](https://github.com/fixie-ai/ultravox) and Qwen-Audio accept raw audio as input alongside text, eliminating the STT stage entirely. The LLM "hears" the caller directly — it can pick up on tone, hesitation, and emphasis that STT flattens into text. For a law-firm intake agent, this means better detection of caller distress (an escalation signal) and more natural turn-taking since the model processes speech without waiting for a complete utterance.

**Architecture shift:** STT is removed. Audio frames go directly to the LLM, which outputs text for TTS. The pipeline shrinks from 3 stages to 2, cutting ~300ms of latency. Pipecat already supports Ultravox as a provider, so the migration path is swapping one service.

### Speech-to-Speech models

Models like [Moshi](https://github.com/kyutai-labs/moshi) and Qwen2.5-Omni generate audio output directly — no TTS stage. The model controls prosody, pacing, and emphasis natively. Both STT and TTS are eliminated, collapsing the pipeline to a single model.

**What this enables:** The agent could emphasize key details ("Your consultation is on **Tuesday at 2pm**"), pause naturally before important information, and match the caller's speaking pace. Current TTS (especially local Piper) sounds robotic precisely because it has no semantic understanding of what it's reading.

**Trade-off:** These models are early-stage. They can't yet match the quality of a tuned LLM (for reasoning) + ElevenLabs (for voice quality) combination. But the gap is closing fast.

### Transport: WebRTC over WebSocket

The current WebSocket transport adds buffering latency that WebRTC avoids. [Daily](https://www.daily.co/) and LiveKit provide WebRTC transports for Pipecat with sub-100ms audio delivery. For phone calls, Twilio Media Streams already uses WebSocket, but a Twilio → WebRTC bridge (via Daily) can reduce perceived latency.

### Smart turn detection

Current VAD (Voice Activity Detection) uses simple energy thresholds — it can't distinguish a mid-sentence pause from "I'm done talking." Models like [smart-turn](https://huggingface.co/livekit/smart-turn-v2) classify whether a pause is a turn boundary using linguistic context, reducing both premature interruptions and awkward silences. This is especially valuable for legal intake where callers often pause to think about sensitive details.
