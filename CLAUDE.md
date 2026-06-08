# Voice AI Agent for Law Firms

## What this is
Inbound voice AI agent for law firms. Handles calls end-to-end: greeting, routing by legal area (employment/tenancy), entity capture with confidence handling, consultation booking, and human escalation.

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
- TTS: Piper (local) | ElevenLabs (cloud)
- DB: SQLite with aiosqlite
- Transport: Pipecat WebSocket | Twilio Media Streams

## Project layout
- `backend/app/` — FastAPI app
  - `pipeline/` — Pipecat pipeline setup and tool handlers
  - `providers/` — Provider abstraction (ABCs + implementations)
  - `conversation/` — State management, prompts
  - `tools/` — LLM function calling tools (routing, extraction, booking, escalation)
  - `services/` — Business logic (calendar/booking service)
  - `api/` — HTTP + WebSocket endpoints
- `frontend/` — Browser UI (vanilla HTML/JS, no build step)
- `scripts/` — Setup and seed scripts

## Commands
```bash
make setup    # install deps, download models, seed DB
make run      # start backend + frontend
make seed     # re-seed calendar slots
```

## Key conventions
- Provider swapping via env vars (STT_PROVIDER, LLM_PROVIDER, TTS_PROVIDER)
- All tools registered in tools/registry.py using a registry pattern
- Conversation state is serializable (ready for Redis at scale)
- Pipecat handles audio plumbing; custom code handles business logic only