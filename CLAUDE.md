# Voice AI Agent for Law Firms

## What this is
Inbound voice AI agent for law firms. Handles calls end-to-end: greeting, routing by legal area (employment/tenancy/traffic), entity capture with confidence handling, consultation booking, and human escalation.

## Architecture
See ARCHITECTURE.md for full details. Key decisions:
- **Pipecat** for pipeline orchestration (STT→LLM→TTS, VAD, interruptions, turn-taking)
- **Local-first stack**: faster-whisper + Ollama + Piper (zero API keys)
- **Cloud-swappable**: change env vars to use Deepgram/OpenAI/ElevenLabs
- **Twilio** transport for real phone calls (optional, cloud deployment)
- **FastAPI** backend with WebSocket transport for browser demo

## Stack
- Python 3.11+, FastAPI, Pipecat
- STT: faster-whisper (local) | Deepgram (cloud)
- LLM: Ollama qwen2.5:7b (local) | OpenAI (cloud)
- TTS: Piper (local) | Cartesia / ElevenLabs (cloud)
- DB: SQLite with aiosqlite
- Transport: Pipecat WebSocket | Twilio Media Streams

## Project layout
- `backend/app/` — FastAPI app
  - `pipeline/` — Pipecat pipeline assembly + custom FrameProcessors
  - `providers/` — Vendor-agnostic STT/LLM/TTS factories: name→builder registries (`create_stt/llm/tts`) + the `ConversationAware` protocol. Add a vendor = one registry entry, no caller changes.
  - `conversation/` — State management, the scripted spine (flow/script), per-field capture strategies (email/phone/insurance), parsers, prompts
  - `tools/` — LLM function calling tools (routing, extraction, booking, escalation)
  - `services/` — Business logic (calendar/booking service)
  - `api/` — HTTP + WebSocket endpoints
- `frontend/` — Browser UI (vanilla HTML/JS, no build step)
- `scripts/` — Setup and seed scripts

## Commands
```bash
make setup      # install deps, download models, seed DB
make run        # start backend + frontend
make seed       # re-seed calendar slots
make demo-call  # headless demo calls → demo/ (audio + transcripts, asserts outcomes)
```

## Key conventions
- Provider swapping via env vars (STT_PROVIDER, LLM_PROVIDER, TTS_PROVIDER)
- All tools registered in tools/registry.py using a registry pattern
- Conversation state is serializable (ready for Redis at scale)
- Pipecat handles audio plumbing; custom code handles business logic only

## Design principles (follow these)
Write clean, maintainable code that respects core OOP and software-design
principles. Before adding or changing code, prefer the design that upholds these;
when touching existing code that violates them, improve it rather than extend the
violation.
- **Encapsulation** — keep state private to the object that owns it. Don't reach
  into another object's internals (e.g. mutating `conversation.state.*` from a
  processor); expose intent-revealing methods instead. No leaking implementation
  details across module boundaries.
- **Abstraction** — depend on interfaces/ABCs, not concretes. Provider code talks
  to the abstraction (`providers/`), not to a specific vendor SDK. Callers should
  not need to know whether STT is Whisper or Deepgram.
- **Single Responsibility** — one class/function, one reason to change. Split
  god-objects and multi-purpose functions; a 900-line manager is a smell.
- **DRY / no dead code** — one source of truth; delete unused branches, fields,
  and parameters rather than leaving them "just in case".
- **Open/Closed & dependency inversion** — adding a provider or a phase should
  mean adding a class/registry entry, not editing a switch in five places.
- **Cohesion over coupling** — related logic lives together; modules expose small,
  stable surfaces. Avoid hidden temporal coupling (call order that silently
  matters).
- Favor pure, testable functions; keep side effects (DB, network, audio) at the
  edges. Name things for intent, not mechanism.