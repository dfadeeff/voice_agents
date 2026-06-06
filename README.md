# Voice AI Agent for Law Firms

Inbound voice agent that handles calls end-to-end: greeting, routing by legal area (employment/tenancy), entity capture with confidence handling, consultation booking, and human escalation.

Runs locally with zero API keys. No cloud accounts needed.

## Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** — local LLM runtime (provides the "brain" of the agent)
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

After installing, start the Ollama server:
```bash
ollama serve
```
Leave this running in a separate terminal.

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/dfadeeff/voice_agents.git && cd voice_agents

# 2. Install Python dependencies
pip install -e "backend/.[dev]"

# 3. Download models (Piper TTS voice + Ollama LLM)
python3 scripts/download_models.py

# 4. Seed the appointment calendar
mkdir -p data && python3 scripts/seed_calendar.py

# 5. Start the server
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser and click "Start Call".

Or use `make`:
```bash
make setup   # steps 2-4 above
make run     # step 5
```

## Choosing a Local Model

The agent's quality depends heavily on the LLM. Set the model in `.env` via `OLLAMA_MODEL`:

| Model | Size | RAM needed | Tool calling | Conversation quality | Recommended for |
|-------|------|-----------|-------------|---------------------|----------------|
| `qwen2.5:7b` | 4.7 GB | 8 GB | Good | Good | Machines with 8-16 GB RAM |
| `qwen2.5:14b` | 9 GB | 16 GB | Very good | Very good | Machines with 16-32 GB RAM |
| `qwen2.5:32b` | 20 GB | 32 GB | Excellent | Excellent | Machines with 32+ GB RAM |
| `llama3.1:8b` | 4.9 GB | 8 GB | Poor | Good | Not recommended (dumps JSON as text) |

**Qwen 2.5 is strongly recommended** over llama3.1 for this use case. llama3.1:8b cannot reliably use function calling via Ollama's API — it outputs raw JSON as text instead of using the tool calling protocol. Qwen 2.5 handles tool calling correctly at every size.

To switch models:
```bash
# Pull the model
ollama pull qwen2.5:32b

# Set in .env
OLLAMA_MODEL=qwen2.5:32b
```

When using Qwen 2.5 (or any model with reliable tool calling), set `USE_TOOLS_LOCAL=true` in `.env` to enable the full tool-calling pipeline locally. Without this, the local agent runs in conversational mode (no structured tool calls).

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
5. **Escalation** — if you ask for a human, the issue is out of scope, or the agent can't understand you after 3 attempts, it hands off with context

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

## Stack

All local, all free:

| Layer | Technology | What it does |
|-------|-----------|--------------|
| **Orchestration** | [Pipecat](https://github.com/pipecat-ai/pipecat) | Pipeline wiring, VAD, turn-taking, interruptions, streaming |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (via Pipecat) | Speech-to-text with per-word confidence scores |
| **LLM** | [Ollama](https://ollama.com) + Qwen 2.5 | Conversation, tool calling, routing decisions |
| **TTS** | [Piper](https://github.com/rhasspy/piper) (via Pipecat) | Text-to-speech, en_US-lessac-medium voice |
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
│       │   ├── processors.py       # Transcript display + call logging
│       │   └── services.py         # STT/LLM/TTS factory from config
│       ├── conversation/
│       │   ├── state.py            # Serializable call state
│       │   ├── manager.py          # State management, confidence scoring
│       │   └── prompts.py          # System prompt + law-area fragments
│       ├── tools/
│       │   ├── registry.py         # Tool name -> handler + JSON schema
│       │   ├── intent.py           # classify_caller_intent
│       │   ├── routing.py          # classify_legal_area
│       │   ├── extraction.py       # extract_caller_details + confidence
│       │   ├── booking.py          # check_availability, book_consultation
│       │   └── escalation.py       # escalate_to_human
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
│   └── seed_calendar.py            # Populate 2 weeks of appointment slots
├── logs/                           # Call transcripts (auto-created, gitignored)
├── ARCHITECTURE.md                 # Detailed architecture + diagrams
├── Makefile
└── .env.example
```

## Design Decisions & Trade-offs

### Why Pipecat (not a custom pipeline)

**Decision:** Use Pipecat for all audio orchestration.

**Trade-off:** Adds a dependency (~50 transitive packages) but eliminates thousands of lines of custom audio code.

**Why:** Pipecat handles VAD (Silero), turn-taking, interruption/barge-in, streaming TTS sentence chunking, and transport abstraction out of the box. Building this from scratch would take weeks and produce a worse result. Our code only handles business logic — tools, prompts, and conversation state. When Pipecat's API changed between versions (e.g., `create_context_aggregator()` removed in 1.3.0, `PipelineRunner` deprecated for `WorkerRunner`), the migration was straightforward because our custom code is minimal.

### Why Qwen 2.5 (not llama3.1)

**Decision:** Default to `qwen2.5:7b` for the local LLM.

**Trade-off:** Qwen 2.5 is less widely known than llama3.1, but it's the same download size (4.7 GB vs 4.9 GB) and runs identically via Ollama.

**Why:** llama3.1:8b **cannot reliably do function calling** via Ollama's OpenAI-compatible API. When given tool schemas, it exhibits three failure modes:
1. **Outputs raw JSON as text** — e.g., `{"name": "greet", "parameters": {}}` spoken aloud by TTS instead of using the tool calling protocol
2. **Calls wrong tools with garbage arguments** — e.g., calls `extract_caller_details` with `{"debug_mode": true}` on the very first turn
3. **Skips greeting entirely** — calls a tool immediately with empty content instead of speaking first

This was verified via direct API testing (`curl` to Ollama's `/v1/chat/completions`). The issue is in the model's fine-tuning for tool use, not in our code or Pipecat. Qwen 2.5 handles the same tool schemas correctly at every size (7B, 14B, 32B).

### Local vs Cloud LLM: automatic tool toggle

**Decision:** Disable tool calling for Ollama by default, enable it when `USE_TOOLS_LOCAL=true` or when using OpenAI.

**Trade-off:** Local mode without tools still has a natural conversation (greeting, asking about legal issues, collecting details) but doesn't execute structured business logic (booking, calendar checks). With tools enabled (qwen2.5 or cloud), the agent does real booking and routing.

**Why:** A single flag in `ws.py` controls this:
```python
use_tools = settings.llm_provider != "ollama" or settings.use_tools_local
```
This flows into `orchestrator.py` which either registers tool handlers + tool-aware prompt, or uses a simpler conversational prompt with no tool schemas. All 6 tools and their handlers stay intact in the codebase — nothing is deleted, just not wired in. This means switching from local-conversational to local-with-tools or cloud-with-tools is a one-line `.env` change.

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

**Trade-off:** Processes complete utterances (300-500ms latency) rather than streaming interim results like cloud Deepgram.

**Why:** Pipecat has a built-in `WhisperSTTService` for faster-whisper. It provides per-word confidence scores, which feed directly into our confidence-aware entity extraction (names/emails/phones with low STT confidence get flagged for verbal confirmation). Vosk streams faster but doesn't report word-level confidence. Whisper.cpp would need a custom integration.

### Confidence-aware entity extraction

**Decision:** Use STT word-level confidence scores to decide when to ask callers to spell back names, emails, and phone numbers.

**Trade-off:** Adds complexity to the extraction flow (confidence threshold, confirmation loop) but prevents booking with wrong details.

**Why:** The hardest problem in voice agents: getting "Siobhan" right when the STT hears "Shivan." When faster-whisper reports confidence < 0.7 for any word in a name, email, or phone number, the `extract_caller_details` tool flags it. The LLM then asks the caller to spell it back ("That's S-I-O-B-H-A-N, is that right?") before proceeding. This works at the tool level — the LLM doesn't need to understand confidence scores; it just follows the tool's instruction.

### LLM-driven flow (not a rigid FSM)

**Decision:** Guide conversation via system prompt and tool availability, not a state machine.

**Trade-off:** Less predictable than a strict FSM — the LLM might ask things in a different order. But the conversation feels natural.

**Why:** A rigid FSM forces "step 1: greet, step 2: classify, step 3: route" which sounds robotic on a phone call. The LLM decides naturally when to route, when to ask for details, and when to book. Business rules are enforced at the tool level (e.g., `book_consultation` refuses if details aren't confirmed), so the LLM can't skip required steps even if it tries.

### Provider abstraction via env vars

**Decision:** Swap STT/LLM/TTS providers by changing environment variables, no code changes.

**Trade-off:** Requires maintaining factory functions (`services.py`) and provider-specific imports. But adding a new provider is 5 lines in the factory.

**Why:** The project requirement is local-first with zero API keys, but production would use cloud providers. `pipeline/services.py` creates the right Pipecat service based on config:
```
STT_PROVIDER=whisper → WhisperSTTService (local)
STT_PROVIDER=deepgram → DeepgramSTTService (cloud)
```
You can mix freely — e.g., local Whisper STT + cloud OpenAI LLM + local Piper TTS.

### WebSocket text side-channel (not protobuf)

**Decision:** Send transcription text as JSON via `websocket.send_json()` alongside Pipecat's binary protobuf audio stream.

**Trade-off:** Two message formats on the same WebSocket (JSON text + protobuf binary) instead of one unified protocol.

**Why:** Pipecat's `ProtobufFrameSerializer` uses exact `type()` checks (not `isinstance`) and only serializes `TextFrame` and `AudioRawFrame`. Our `AggregatedTextFrame` (which carries the LLM's response) extends `TextFrame` but doesn't match the exact type check, so it never reaches the frontend via protobuf. Similarly, `TranscriptionFrame` is consumed by the `UserAggregator` before it reaches the output transport. Custom processors (`TranscriptProcessor`, `AgentTextProcessor`) intercept these frames and send them as JSON directly. The frontend handles both: `typeof event.data === "string"` for JSON, `instanceof ArrayBuffer` for audio.

### Call logging to JSON files

**Decision:** Write call transcripts to `logs/{call_id}.json` files.

**Trade-off:** Not queryable like a database. But simple, zero-dependency, and easy to inspect.

**Why:** For a demo/take-home, JSON files are sufficient to verify calls are being recorded and transcribed. The `CallLogger` class accumulates entries during the call and writes them on disconnect. In production, these would go to a database (the SQLite infrastructure is already there for calendar data). Logs are gitignored.

## Troubleshooting

**"Ollama not found"** — Make sure `ollama serve` is running in a separate terminal.

**`ollama pull` fails with "connection reset by peer"** — This is an IPv6 connectivity issue with Cloudflare's CDN. Fix: disable IPv6 temporarily (`sudo networksetup -setv6off Wi-Fi`), pull the model, then re-enable (`sudo networksetup -setv6automatic Wi-Fi`). Also: never run multiple `ollama pull` commands simultaneously — they compete for bandwidth and both fail.

**Slow first response** — The first call loads Whisper + Ollama models into memory. Subsequent calls are faster.

**Microphone not working** — Chrome requires HTTPS for mic access on non-localhost origins. On localhost it works over HTTP.

**Agent outputs JSON instead of speaking** — You're using llama3.1. Switch to qwen2.5 (`OLLAMA_MODEL=qwen2.5:7b` in `.env`).

**NLTK SSL warning** — Harmless. Appears on macOS when Python's SSL certificates aren't installed. Run: `/Applications/Python 3.11/Install Certificates.command`

## Honest Limitations

- **STT latency**: faster-whisper processes complete utterances (~300-500ms), not streaming. Cloud Deepgram would stream interim results during speech.
- **TTS quality**: Piper is functional but noticeably synthetic compared to ElevenLabs. This is the most obvious "not production" tell in a demo.
- **End-to-end latency**: ~1-2s locally (STT + LLM inference + TTS) vs ~500-800ms with a cloud stack. The LLM inference is the bottleneck — larger models (32B) are slower but smarter.
- **Local tool calling**: Only works reliably with Qwen 2.5 models. llama3.1:8b outputs raw JSON text instead of using the tool calling API. This is a model limitation, not a code issue.
- **No streaming STT**: faster-whisper waits for a complete utterance before transcribing. The user sees nothing in the UI while speaking — the transcript appears all at once after they stop. Cloud Deepgram provides interim results that show words appearing as you speak.
- **Single-process**: One Uvicorn worker handles all calls. Under load, calls would queue. Production would use multiple workers behind a load balancer, with Redis for shared state.
