# Latency results (per provider configuration)

Recorded with `make compare-latency ARGS="--record <label> <log>"`. TTFA is perceived latency (caller stops speaking → first agent audio); the rest are per-component TTFB. LLM ~0 ms = turns fast-pathed by the deterministic spine.

| config | calls | TTFA | STT | LLM | TTS | recorded |
|---|---|---|---|---|---|---|
| cloud-deepgram-stt | 1 | 157 ms | Deepgram 390 ms | Ollama 0 ms | Piper 126 ms | 2026-06-09 22:19 |
