# Latency results (per provider configuration)

Recorded with `make compare-latency ARGS="--record <label> <log>"`. TTFA is perceived latency (caller stops speaking → first agent audio); the rest are per-component TTFB. LLM ~0 ms = turns fast-pathed by the deterministic spine.

**Read TTFA as the headline metric** — it's measured from turn-end → first agent audio and is comparable across configs. Two caveats on the component columns:
- **LLM ~0 ms in every row**: the deterministic scripted spine fast-paths capture/booking, and these scripted demo calls don't hit free-form routing/INFORMATION turns, so the LLM is essentially never invoked. The local↔cloud delta is therefore STT + TTS; the cloud LLM's quality edge isn't exercised here.
- **STT TTFB inflates once per-turn endpoint tuning is on** (from `cloud: deepgram + gpt-4o`): dictation turns use a 2.0 s silence window, so STT TTFB on those turns reflects the *endpoint wait*, not the model. Deepgram's true speed shows on normal turns (~390–440 ms); the avg mixes both. Whisper (local) ~1819 ms is genuine processing.

| config | calls | TTFA | STT | LLM | TTS | recorded |
|---|---|---|---|---|---|---|
| cloud-deepgram-stt | 1 | 157 ms | Deepgram 390 ms | Ollama 0 ms | Piper 126 ms | 2026-06-09 22:19 |
| cloud-deepgram-stt (run 2) | 1 | 164 ms | Deepgram 455 ms | Ollama 0 ms | Piper 148 ms | 2026-06-09 23:00 |
| local (whisper+ollama+piper) | 1 | 208 ms | Whisper 1819 ms | Ollama 0 ms | Piper 124 ms | 2026-06-09 23:21 |
| cloud-deepgram-stt (run 3, VAD 1.2s) | 1 | 142 ms | Deepgram 444 ms | Ollama 0 ms | Piper 128 ms | 2026-06-09 23:21 |
| cloud: deepgram + gpt-4o (piper tts) | 1 | 231 ms | Deepgram 913 ms | OpenAI 0 ms | Piper 219 ms | 2026-06-10 00:40 |
