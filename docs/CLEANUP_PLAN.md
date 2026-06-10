# Clean-up plan

Verified against the code on `main`. Ordered by leverage ÷ effort. Each phase is a
standalone PR that keeps `pytest` green; later phases assume earlier ones.

The split is real: `providers/` is textbook (registries, lazy vendor imports,
`ConversationAware` protocol). The domain core (`manager.py`, `state.py`, `tools/`)
violates the encapsulation / SRP / DRY rules that CLAUDE.md itself states. That
stated-vs-actual gap is the priority because a reviewer reads CLAUDE.md first.

---

## Phase 0 — Correctness bugs (do first; small, high-visibility)

**0a. Fix the Twilio path + extract a shared session factory.**
`twilio.py:83` builds `ConversationManager(call_id=call_id)` — no `lang`, no
`calendar`, no `use_tools`, no `_save_caller`, no semaphore. Result: `calendar=None`
→ `_try_book` is a no-op → `compute_prompt` hits `no_slots` → every phone caller is
told there are no appointments; teardown never persists the caller.
- Add `app/api/session.py: build_conversation(app, settings)` and
  `run_call_session(transport, websocket/ws, settings)` used by **both** `ws.py`
  and `twilio.py`: provider creation, manager wiring (`lang`, `calendar`),
  `use_tools`, the concurrency gate, and the final caller save.
- Net effect: fixes Twilio *and* removes the duplicated setup/teardown.
- Effort: **M**. Risk: low (Twilio is untested locally; cover with a unit test that
  asserts the Twilio branch builds a manager with a calendar).

**0b. CORS.** `main.py:64` `allow_origins=["*"]` on a legal-PII service. Restrict to
the configured frontend origin(s) via a setting; note it in the README privacy
section. Effort: **S**.

---

## Phase 1 — Encapsulation: privatize `ConversationState` behind intent methods

The most-cited CLAUDE.md rule ("don't mutate `conversation.state.*` from a
processor") is broken in `tools/booking.py`, `tools/route.py`, `pipeline/processors.py`,
`pipeline/local_whisper.py`, and `api/ws.py`.

- Rename `ConversationState` access to private (`manager._state`) and add the missing
  intents on `ConversationManager`:
  - `confirm_booking(slot)` (replaces `booking.py:147-149`'s three assignments + `advance_phase`)
  - `record_escalation()` (replaces `route.py:56-57`)
  - `record_offered_slots(slots)` (replaces `booking.py:67,82`)
  - `awaiting_field() -> str | None`, `is_booking_confirmed() -> bool`
    (for `processors.py:257,548,569,719` and `local_whisper.py:67` — kills the
    double-`getattr`)
  - `snapshot_for_persistence() -> CallerRecord` (for `ws._save_caller`)
- Tools receive a narrow facade, not raw `ctx.state`.
- Effort: **M**. Risk: medium (touches tools + processors); mechanical, fully
  covered by existing tests. Do **after** Phase 0 so the new save path lands once.

---

## Phase 2 — DRY: one source of truth (fast, greppable wins)

- New `app/conversation/validation.py`: single `EMAIL_VALID_RE` and
  `LOW_CONFIDENCE_THRESHOLD`. Replace the 4 email regexes
  (`manager._EMAIL_VALID_RE`, `extraction.EMAIL_RE`, `processors._EMAIL_RE`,
  `email_extract._EMAIL_RE`) and the 2 threshold defs (`manager:23`,
  `extraction:92`). Keep `processors._TTS_EMAIL_RE` (different job).
- Extract `_compile_tool_name_re(tool_names)` used by both `PreTTSSanitizer.__init__`
  (`processors.py:163`) and `AgentTextProcessor.__init__` (`:692`).
- Single `compute_outcome(state) -> str` (one place: `flow.py` or the manager),
  used by both `manager._persist` (`:735-741`) and the teardown save (`ws.py:40-44`).
  Folds naturally into Phase 0's shared teardown.
- Effort: **S–M**. Risk: low.

---

## Phase 3 — SRP: extract per-field capture strategies (largest, highest-quality)

`manager.py` (1392 lines) holds routing tables, email grammar, the spelling parser,
Levenshtein anchoring, insurance accumulation, slot matching, deterministic booking,
and persistence. `state.py` carries 7 email fields + 3 insurance fields — two missing
objects.

- Define a `FieldCapture` strategy (parse / confirm / retry / buffer policy) and
  implement `EmailCapture`, `PhoneCapture`, `InsuranceCapture`, `NameCapture`. Each
  owns its buffer + attempt counters (collapses the 10 `email_*`/`insurance_*`
  fields into `EmailCaptureSession` / `InsuranceCaptureSession`).
- Replace the `_try_capture_*` chain in `add_user_message` with a dispatch on a
  registry keyed by `awaiting`. Adding a field = adding a class (open/closed).
- Move pure parsers into focused modules: `email_parse.py` (grammar + spelling +
  `_anchor_email`/`_levenshtein`), reuse `phone.py`, `time_parse.py` (hour words +
  `_parse_requested_time` + slot matching). Manager shrinks to dispatch + state.
- Effort: **L**. Risk: medium — but the scenario/flow tests pin behaviour, so
  refactor under green tests. This is the change that most directly answers the
  "manager is a smell" critique.

---

## Phase 4 — Open/Closed for legal areas (deeper; optional)

Adding an area today means editing `_MATTER_KEYWORDS`, `_AREA_DISAMBIG_KEYWORDS`,
both locale files, `MATTER_TYPES`, the seed script, **and** `flow.py:93`'s hard-coded
`if state.legal_area == LegalArea.TRAFFIC` insurance branch (and the mirror in
`compute_prompt`).

- Introduce an `AreaDescriptor` (name, keywords, qualification slots, required
  fields). `next_phase`/`compute_prompt` read the descriptor instead of branching on
  `TRAFFIC`. Adding an area = adding a descriptor + locale strings.
- Effort: **L**. Risk: medium. Do last; biggest design payoff, least immediate need.

---

## Phase 5 — Honest claims (cheap, high credibility)

- **Concurrency:** `max_concurrent_calls=10` is not real — sync `sqlite3`
  (`upsert_caller_sync`, `available_slots_sync`) blocks the event loop, the shared
  Whisper model serialises on CPU, and `ws.py:75` `sem._value` is a private peek with
  a check-then-accept race. Either default to `1` with a README note on the
  Redis/worker-pool shape for N, or wrap sync DB calls in `asyncio.to_thread` and use
  the semaphore as a context manager only (drop the `_value` peek). Effort: **S**.
- **Benchmark CI:** gate the 4.7 GB-model job to `workflow_dispatch` (flaky/slow on a
  hosted runner with a 30-min timeout). Fix or stop committing the benchmark JSON
  whose `greeting_with_tools` PASS has garbled detail text. Effort: **S**.

---

## Related but NOT clean-up (behaviour change — track separately)

**Off-script turns.** While `awaiting == "email"`, "Warum brauchen Sie meine
E-Mail?" is parsed as a failed attempt and burns a retry. A cheap intent check
(`?`/`warum`/`wieso` regex, or one-shot LLM classify) → brief LLM answer → re-issue
the scripted ask without consuming an attempt. Highest-impact *UX* fix; name it in
the video as a known limitation with a designed fix. Effort: **M**.

---

## If time is short — minimum set before submission

1. **Phase 0** (Twilio + CORS) — fixes a real bug that undercuts the abstraction story.
2. **Phase 1** (encapsulation) — closes the most-cited CLAUDE.md gap.
3. **Phase 2** (DRY) — fast, and reviewers grep for exactly these.

Phases 3–4 are the "strong → exceptional" work; if they can't land in time, name the
strategy-registry / area-descriptor shapes in the video as designed debt.
