# Latency results (per provider configuration)

Recorded with `make compare-latency ARGS="--record <label> <log>"`. TTFA is perceived latency (caller stops speaking → first agent audio); the rest are per-component TTFB. LLM ~0 ms = turns fast-pathed by the deterministic spine.

| config | calls | TTFA | STT | LLM | TTS | recorded |
|---|---|---|---|---|---|---|
| cloud-deepgram-stt | 1 | 157 ms | Deepgram 390 ms | Ollama 0 ms | Piper 126 ms | 2026-06-09 22:19 |
| cloud-deepgram-stt (run 2) | 1 | 164 ms | Deepgram 455 ms | Ollama 0 ms | Piper 148 ms | 2026-06-09 23:00 |
| local (whisper+ollama+piper) | 1 | 208 ms | Whisper 1819 ms | Ollama 0 ms | Piper 124 ms | 2026-06-09 23:21 |
| cloud-deepgram-stt (run 3, VAD 1.2s) | 1 | 142 ms | Deepgram 444 ms | Ollama 0 ms | Piper 128 ms | 2026-06-09 23:21 |
