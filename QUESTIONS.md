# Additional Questions

Answers to the four questions from the takehome, referencing actual code in the repo.

---

## Question 1: Real-Time Latency & Pipeline Design

> How would you design the pipeline to keep end-to-end response latency low while preserving conversation quality?

### The latency budget

The critical metric is **Time-to-First-Audio (TTFA)** — the gap between the caller finishing their sentence and hearing the first word of the agent's response. Above ~1.5 seconds feels like dead air; below ~800ms feels conversational.

TTFA is the sum of three sequential stages:

```
TTFA = STT latency + LLM time-to-first-token + TTS time-to-first-byte
```

With the local stack:
- **STT** (faster-whisper, small model): ~300-500ms per utterance
- **LLM** (Ollama qwen2.5:7b): ~300-500ms TTFT depending on hardware
- **TTS** (Piper): ~50ms TTFB (lightweight CPU model)

Total: ~650ms-1s locally. With cloud providers (Deepgram + GPT-4o-mini + ElevenLabs), the Salesforce paper benchmarks this at ~755ms measured end-to-end.

### How the pipeline reduces latency

The key insight is **streaming overlap** — stages don't wait for the previous stage to fully complete.

**1. Sentence-level TTS streaming** (`pipeline/orchestrator.py`)

Pipecat's pipeline is a frame processor chain. The LLM streams tokens, Pipecat's built-in sentence aggregator accumulates them until a sentence boundary (period, question mark, exclamation), then immediately pushes that sentence to TTS. TTS starts synthesizing the first sentence while the LLM is still generating the second. This means:

```
Effective TTFA = STT + LLM(first sentence) + TTS(first sentence)
```

Not `STT + LLM(full response) + TTS(full response)`. For a two-sentence response, this can save 500ms+.

**2. Small model selection** (`config.py:25`, `providers/llm.py`)

I default to qwen2.5:7b (4.7GB) rather than a larger model. For a receptionist conversation, the model needs to do three things well: follow the phase-specific system prompt, call the right tool with correct arguments, and phrase short responses naturally. qwen2.5:7b does all three — it was specifically trained for function calling — at ~0.8s latency. A 14B or 70B model might reason better in edge cases, but the extra seconds of inference latency make the call feel broken. I avoid qwen3:8b because it emits thinking tokens that create dead air on the phone line.

The model is configurable via `OLLAMA_MODEL` in `.env` — qwen3:4b is ~0.3s faster if you are latency-constrained and accept slightly less reliable tool calling.

**3. Phase-constrained prompts** (`prompts.py`, `build_system_prompt`)

Each phase gives the LLM a narrow task with 2-3 sentences of instructions, not the full call flow. This keeps system prompts short, which reduces prefill time. The LLM also produces shorter outputs because it has a focused task ("Ask for the caller's name") rather than an open-ended one ("Handle this call").

**4. VAD endpointing** (`pipeline/orchestrator.py`, `config.py`)

Silero VAD with an 800ms silence threshold (`silence_timeout_ms=800`) triggers the STT as soon as the caller pauses. This is a tradeoff:
- Too short (300ms): interrupts the caller mid-sentence, wasting a turn
- Too long (1500ms): adds dead air after every utterance
- 800ms: close to the Salesforce paper's recommendation, catches natural sentence-ending pauses without cutting off mid-thought

### What I would change for production

1. **Streaming STT**: faster-whisper processes complete utterances. Deepgram Nova streams interim results during speech — the LLM can start processing before the caller finishes, shaving 200-300ms. Switch with `STT_PROVIDER=deepgram` (`providers/stt.py`).

2. **GPU inference or vLLM**: Ollama on CPU is the local-demo bottleneck. In production I'd serve the model via vLLM on a GPU with PagedAttention for efficient batching, or use a cloud LLM (GPT-4o-mini via `LLM_PROVIDER=openai`, `providers/llm.py`).

3. **Streaming TTS**: Piper generates complete audio synchronously. ElevenLabs streams audio chunks as they're synthesized (~220ms TTFB). Switch with `TTS_PROVIDER=elevenlabs` (`providers/tts.py`).

4. **Smart turn detection**: Replace the fixed 800ms silence threshold with a model like [smart-turn](https://huggingface.co/livekit/smart-turn-v2) that uses linguistic context to decide if a pause is a turn boundary. Particularly valuable for legal intake where callers pause to think about sensitive details.

---

## Question 2: Turn-Taking, Interruptions & Audio Robustness

> How does your agent decide when the caller has finished speaking, and how does it handle being interrupted?

### Endpointing: when has the caller finished?

The pipeline uses **Silero VAD** (Voice Activity Detection) via Pipecat's VAD analyzer (`pipeline/orchestrator.py`):

```python
vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
```

Silero VAD is a 2MB neural model that classifies 32ms audio chunks as speech or silence in <1ms on CPU. The VAD processor maintains a state machine:

```
IDLE ─speech──> LISTENING ─silence 800ms──> PROCESSING ─LLM+TTS──> SPEAKING ─done──> IDLE
```

When the VAD detects 800ms of continuous silence (`config.py`: `silence_timeout_ms=800`), it marks the utterance as complete and pushes the accumulated audio to STT. The threshold is configurable — a tradeoff between responsiveness (shorter = faster replies) and robustness (longer = fewer mid-sentence interruptions). While the agent is awaiting a dictated field (phone, email, insurance number), the window widens per-turn to `silence_timeout_dictation_ms=2000`, because callers read numbers out in bursts with pauses that a short window would chop into separate turns.

The VAD threshold (`config.py`: `vad_threshold=0.5`) controls how aggressively speech is detected. 0.5 is the default — lower values detect quieter speech but increase false positives from background noise.

### Barge-in: when the caller talks over the agent

Pipecat handles barge-in natively. When the VAD detects new speech while the agent is speaking:

1. **TTS playback stops** — Pipecat cancels the current audio output and flushes the TTS buffer
2. **LLM generation stops** — if the LLM is still streaming, that generation is abandoned
3. **Pipeline resets to listening** — the new speech is captured and processed as a fresh turn

This is critical for a receptionist agent. If the agent says "Let me check availability for Tuesday, June 10th at—" and the caller says "Actually, Wednesday would be better," the agent shouldn't finish the Tuesday sentence. The pipeline drops it and processes the interruption.

The `AgentTextProcessor` (`pipeline/processors.py`) sits between the LLM and TTS. It captures the agent's text for the transcript even if the caller interrupts mid-sentence, so the call log reflects what was actually said.

### Echo cancellation

The browser client requests echo cancellation via the MediaStream API (`frontend/components/audio.js:42-47`):

```javascript
mediaStream = await navigator.mediaDevices.getUserMedia({
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
  },
});
```

This prevents the agent's own TTS output (playing through speakers) from being picked up by the mic and fed back into the pipeline as "user speech." Without this, the agent would hear itself speaking and try to respond to its own output.

For Twilio calls (`api/twilio.py:56-65`), the phone network handles echo cancellation at the carrier level. The `TwilioFrameSerializer` handles mulaw 8kHz <-> PCM 16kHz conversion transparently.

### What I would improve

1. **Smart turn detection**: The fixed silence threshold is a blunt instrument. A caller saying "I was dismissed... [400ms pause] ...from my job last week" would get cut off at the pause if the threshold were lower. A learned model like smart-turn-v2 uses linguistic context to distinguish mid-sentence pauses from turn boundaries.

2. **Partial transcript feedback**: With streaming STT (Deepgram), the frontend could show interim results as the caller speaks, so they see their words appearing in real time. Currently the transcript appears all at once after the utterance is complete.

3. **Server-side echo gate**: For noisy environments or speaker phone usage, I'd add a server-side gain reduction on the mic input while the agent is speaking, as a second layer beyond browser-level echo cancellation.

---

## Question 3: Iteration, Scaling & Health Tracking

> How would you take this rough prototype to something callers find genuinely useful, and how would you monitor its health once it's live?

### From prototype to production

The prototype is designed so production upgrades are config changes, not rewrites:

| Concern | Prototype | Production | How to switch |
|---|---|---|---|
| STT | faster-whisper (batch) | Deepgram Nova (streaming) | `STT_PROVIDER=deepgram` |
| LLM | Ollama qwen2.5:7b (CPU) | GPT-4o-mini or vLLM (GPU) | `LLM_PROVIDER=openai` |
| TTS | Piper (robotic) | ElevenLabs (natural) | `TTS_PROVIDER=elevenlabs` |
| State | In-memory dict | Redis | Set `REDIS_URL` |
| DB | SQLite | Postgres | Change `DB_URL` |
| Transport | Browser WebSocket | Twilio Media Streams | Deploy + configure Twilio |
| Concurrency | Single process | Multiple workers + LB | Standard deployment |
| Logging | JSON files | Structured logging + tracing | Add OpenTelemetry |

The provider abstraction (`app/providers/`) means each factory function returns a Pipecat service that plugs into the same pipeline. The business logic (tools, state machine, prompts) doesn't change.

### Scaling architecture

The current architecture has one Uvicorn worker with an explicit per-process capacity gate (`api/session.py:call_capacity`, shared by the browser and Twilio transports; `MAX_CONCURRENT_CALLS=2` locally because shared Whisper serializes on CPU — excess callers are rejected immediately rather than silently queued). For production:

1. **Horizontal scaling**: Run multiple Uvicorn workers behind a sticky-session load balancer. Each WebSocket connection is long-lived, so sticky sessions ensure a call stays on one worker.

2. **Shared state**: Move `ConversationState` from in-memory to Redis (add a `redis_url` setting). The state dataclass is already serializable (`state.py`: `to_dict()`, `to_json()`), designed for this migration.

3. **Database**: Swap SQLite for Postgres by changing `DB_URL`. The SQLAlchemy models and async queries work with both.

4. **Health endpoints** (`api/health.py`): `/health` returns basic liveness, `/ready` checks database connectivity and tool registry — a load balancer can use these for routing.

### Monitoring and health tracking

**Per-call metrics I would track:**

| Metric | What it reveals | Source |
|---|---|---|
| TTFA (time-to-first-audio) | Latency per turn | Timestamp deltas in pipeline |
| Call completion rate | Are callers reaching booking/farewell? | `ConversationState.phase` at disconnect |
| Booking conversion rate | Business value | `booking_confirmed` flag |
| Escalation rate | Agent limitations | `escalation_requested` flag |
| STT confidence distribution | Audio quality / model accuracy | `get_word_confidence_for_value()` |
| Tool call failure rate | LLM reliability | Tool handler error logs |
| Phase progression | Where do calls stall? | Phase transition logs |
| Repeated-question rate | Understanding failures | Turn count vs. phase advancement |
| Barge-in frequency | Turn-taking quality | VAD interruption events |

**Alerting thresholds:**
- TTFA P95 > 2s → LLM or network degradation
- Escalation rate > 30% → prompt or routing issue
- Tool call failure rate > 5% → model or schema regression
- Booking rate drops 20% week-over-week → investigate call recordings

**Offline evaluation:**

The repo includes a scenario-level test suite (`tests/test_scenarios.py`, part of 440+ tests) that verifies the state machine and tool pipeline for realistic call flows: routing per area, unknown-area escalation, human handoff, low-confidence email (rejection → spelling → skip), unavailable slot, full booking happy path. These run in CI and catch regressions in flow logic.

One level up, `make demo-call` (`scripts/demo_call.py`) runs four full calls headlessly through the *real audio pipeline*: caller utterances are synthesized with a second TTS voice, transcribed by the same faster-whisper model with the live hotword biasing, driven through the real conversation manager, and answered by the agent voice — with each scenario asserting its outcome (booking confirmed, callback recorded). The recordings and transcripts land in `demo/`. This is the audio-level regression net the unit tests can't provide: on its first run it caught a callback-flow bug (a low-STT-confidence name looped the name question instead of reading it back) that 400+ unit tests had missed, because only real audio produces low-confidence transcripts.

For LLM behavior evaluation, I would maintain a bank of recorded call transcripts with expected outcomes and run them through the pipeline periodically. Calls where the agent escalated, failed to extract entities, or triggered the legal-advice boundary would be sampled for human review and used to refine prompts.

### Iteration process

1. **Weekly prompt tuning**: Review escalated and failed calls, identify patterns, update phase prompts
2. **Model upgrades**: When a better local model releases, run `make benchmark` to compare latency and tool calling reliability before switching
3. **Scenario expansion**: Each bug or edge case becomes a new scenario test
4. **A/B testing**: For prompt changes, route a percentage of calls to the new prompt and compare booking/escalation rates

---

## Question 4: Telephony, Warm Transfer & Failure Handling

> How would you connect this agent to telephony to dial out and perform a warm transfer to a human? What happens when that transfer call fails?

### Telephony connection

The agent already has a Twilio adapter (`api/twilio.py`). The connection flow:

```
Inbound call → Twilio → POST /twilio/voice (webhook)
                              ↓
                         TwiML response: <Connect><Stream url="wss://server/twilio/stream"/></Connect>
                              ↓
                    Twilio opens WebSocket to /twilio/stream
                              ↓
                    TwilioFrameSerializer handles mulaw 8kHz ↔ PCM 16kHz
                              ↓
                    Same Pipecat pipeline processes the call
```

The pipeline doesn't know or care whether audio comes from a browser or Twilio — both use `FastAPIWebsocketTransport` with different serializers (`api/ws.py` uses `ProtobufFrameSerializer`, `api/twilio.py` uses `TwilioFrameSerializer`). All business logic, tools, and state management work identically across both transports.

### Warm transfer design

When the agent decides to escalate (via the `request_handoff` tool or automatic triggers in `flow.py`), the `request_handoff` handler (`tools/handoff.py:27-51`) builds a context bundle:

```python
context_for_human = {
    "reason": args.get("reason", "unknown"),       # why we're escalating
    "summary": args.get("summary", ""),             # LLM's description
    "caller_details": {k: v.value for k, v in ctx.state.entities.items()},
    "legal_area": ctx.state.legal_area.value,
    "turn_count": ctx.state.turn_count,
    "call_id": ctx.state.call_id,
}
```

In production, the warm transfer would work like this:

1. **Agent tells the caller**: "I'm going to connect you with someone who can help directly. One moment please."
2. **Create an outbound call** to the human recipient using Twilio's REST API:
   ```python
   client.calls.create(
       to=lawyer_phone,
       from_=twilio_number,
       url=f"https://server/twilio/transfer-briefing/{call_id}",
   )
   ```
3. **Brief the human**: The briefing webhook plays a TTS summary: "Incoming transfer: John Smith, employment law, unfair dismissal. Caller has been on the line for 3 minutes. Accept?"
4. **Human accepts** → Twilio bridges the two legs using `<Conference>` or `<Dial>`. The caller hears the human; the agent drops off.
5. **Human rejects or doesn't answer** → return to the caller (see failure handling below).

The key difference from a cold transfer: the human gets context *before* being connected, so the caller doesn't have to repeat their story.

### Failure handling at each layer

**SIP / signalling level:**

| SIP Response | Meaning | Agent reaction | Caller experience |
|---|---|---|---|
| `200 OK` | Human answered | Bridge the call | "I'm connecting you now." |
| `486 Busy Here` | Human is on another call | Try backup recipient or queue | "They're currently on another call. Let me try someone else." |
| `408 Request Timeout` | Human didn't pick up (30s) | Try backup, then offer callback | "I wasn't able to reach them right now. Can I take your number and have someone call you back?" |
| `480 Temporarily Unavailable` | Human's phone is off/unreachable | Try backup route | Same as 408 |
| `503 Service Unavailable` | Twilio or carrier issue | Log error, offer callback | "I'm having trouble connecting. Let me take your details and someone will call you back shortly." |

**Media layer:**

| Failure | Detection | Agent reaction |
|---|---|---|
| Audio stream drops mid-bridge | Twilio `stream` event: `stop` | Keep caller leg alive, attempt reconnect, or collect callback number |
| One-way audio (caller can't hear human) | Silence detection on the bridged leg | Drop bridge, apologize, retry |
| Poor audio quality on transfer | Human reports via DTMF or callback | Log for investigation, offer alternative contact |

**Application layer:**

| Failure | Detection | Agent reaction |
|---|---|---|
| No backup recipients configured | Empty routing table | Offer callback with estimated response time |
| All recipients busy | All transfers return 486 | "Our team is currently assisting other callers. I've noted your details — someone will call you back within [X] minutes." |
| Transfer succeeds but human drops quickly | Short call duration on bridged leg | Follow up with human; may indicate accidental disconnect |

### Implementation approach

I would implement the transfer using Twilio's `<Conference>` primitive rather than `<Dial>`:

1. Place the caller into a named conference room (on hold with music or reassurance)
2. Dial the human into the same conference
3. Play the briefing to the human while they're in the conference but muted to the caller
4. When the human presses `1` to accept, unmute both legs
5. The agent's WebSocket connection can drop — Twilio manages the conference independently

This is more robust than `<Dial>` because:
- The caller stays connected even if the human leg fails
- Multiple transfer attempts can be made without dropping the caller
- The agent can "come back" to the caller if all transfers fail (rejoin the conference via WebSocket)

### What's implemented now vs. production

| Component | Prototype | Production |
|---|---|---|
| Escalation trigger logic | Implemented (`flow.py`, `tools/handoff.py`) | Same |
| Context bundle for human | Implemented (`context_for_human`) | Same + send via API |
| Twilio transport adapter | Implemented (`api/twilio.py`) | Same |
| Outbound transfer call | Not implemented | Twilio REST API + Conference |
| Briefing webhook | Not implemented | TTS reads `context_for_human` |
| Failure handling | Documented (this file) | SIP status handlers + retry logic |
| Monitoring | Call logs | Transfer outcome tracking + latency |
