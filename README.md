# Voice AI Agent — Mitchell & Associates Law Firm

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
git clone <repo-url> && cd voice_agent

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

## What Happens When You Call

1. **Greeting** — the agent answers as a receptionist for Mitchell & Associates
2. **Routing** — identifies your legal area (employment law or tenancy law) and adapts questions accordingly
3. **Capture** — collects your name, email, phone number. If STT confidence is low (noisy line, unusual name), the agent asks you to spell it back rather than guessing
4. **Booking** — checks the calendar for available consultation slots. If your preferred time is taken, offers alternatives
5. **Escalation** — if you ask for a human, the issue is out of scope, or the agent can't understand you after 3 attempts, it hands off with context

## Stack

All local, all free:

| Layer | Technology | What it does |
|-------|-----------|--------------|
| **Orchestration** | [Pipecat](https://github.com/pipecat-ai/pipecat) | Pipeline wiring, VAD, turn-taking, interruptions, streaming |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (via Pipecat) | Speech-to-text with per-word confidence scores |
| **LLM** | [Ollama](https://ollama.com) + llama3.1 8B | Conversation, tool calling, routing decisions |
| **TTS** | [Piper](https://github.com/rhasspy/piper) (via Pipecat) | Text-to-speech, en_US-lessac-medium voice |
| **VAD** | Silero (via Pipecat) | Voice activity detection, endpointing |
| **Transport** | WebSocket + protobuf | Browser mic/speaker over FastAPI WebSocket |
| **DB** | SQLite | Appointment calendar with 560 seeded slots |

### Cloud Swap (optional)

Change env vars to use cloud providers — no code changes:

```bash
# .env
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your-key

LLM_PROVIDER=openai
OPENAI_API_KEY=your-key

TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your-key
```

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
│       │   └── services.py         # STT/LLM/TTS factory from config
│       ├── conversation/
│       │   ├── state.py            # Serializable call state
│       │   ├── manager.py          # State management, confidence scoring
│       │   └── prompts.py          # System prompt + law-area fragments
│       ├── tools/
│       │   ├── registry.py         # Tool name -> handler + JSON schema
│       │   ├── routing.py          # classify_legal_area
│       │   ├── extraction.py       # extract_caller_details + confidence
│       │   ├── booking.py          # check_availability, book_consultation
│       │   └── escalation.py       # escalate_to_human
│       ├── services/
│       │   └── calendar.py         # SQLite: slots, bookings, call logs
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
├── ARCHITECTURE.md                 # Detailed architecture + diagrams
├── Makefile
└── .env.example
```

## Key Design Decisions

### Pipecat for orchestration
Pipecat handles all the audio plumbing: VAD, turn-taking, interruption handling (barge-in), streaming TTS, transport abstraction. Our code only handles business logic — tools, prompts, and conversation state. This avoids reinventing audio pipeline management and gives us production-quality features (Silero VAD, sentence-level TTS streaming) for free.

### Confidence-aware entity extraction
The hard part of voice agents: getting names and emails right over a noisy line. When faster-whisper reports low per-word confidence (< 0.7) for a name, email, or phone number, the `extract_caller_details` tool flags it. The system prompt instructs the LLM to spell back names, read back emails letter-by-letter, and repeat phone numbers digit-by-digit before proceeding to booking.

### LLM-driven flow (not a rigid FSM)
The conversation flow is guided by the system prompt and tool availability, not a state machine. The LLM decides naturally when to route, when to capture details, and when to book. Call phases are tracked implicitly by which tools get called. This makes the conversation feel natural while the tools enforce the business rules (e.g., can't book without confirmed details).

### Provider abstraction via env vars
Swapping from local to cloud is a one-line env var change. The `pipeline/services.py` factory creates the right Pipecat service based on config. Same pipeline, same tools, different backend.

## Troubleshooting

**"Ollama not found"** — Make sure `ollama serve` is running in a separate terminal.

**Slow first response** — The first call loads Whisper + Ollama models into memory. Subsequent calls are faster.

**Microphone not working** — Chrome requires HTTPS for mic access on non-localhost origins. On localhost it works over HTTP.

**NLTK SSL warning** — Harmless. Appears on macOS when Python's SSL certificates aren't installed. Run: `/Applications/Python 3.11/Install Certificates.command`

## Honest Limitations

- **STT latency**: faster-whisper processes complete utterances (~300-500ms), not streaming. Cloud Deepgram would stream interim results during speech.
- **LLM tool calling**: Ollama llama3.1 8B occasionally misformats tool calls compared to GPT-4o. The system prompt compensates with explicit formatting instructions.
- **TTS quality**: Piper is functional but noticeably synthetic compared to ElevenLabs.
- **End-to-end latency**: ~1-2s locally (model loading + inference) vs ~500-800ms with cloud stack.
- **No persistent call logs**: Call state lives in memory per connection. Production would use Redis for state and Postgres for logs.
