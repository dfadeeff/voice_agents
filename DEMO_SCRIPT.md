# Demo Script

Five short calls for the video — one per user story plus a handoff variant. The
agent runs in **German** by default (the receptionist "Claudia"); English glosses
are in parentheses. Say the caller lines out loud; watch the server log lines
quoted under each step to narrate what the code is doing.

Run it:
```bash
ollama serve            # terminal 1
make run                # terminal 2
open http://localhost:8000
```

What to watch for in the logs:
- `Phase advanced: greeting → qualification → capture → booking → confirmation`
- `FAST PATH: …` — a scripted turn (LLM skipped; deterministic spine)
- `Deterministic capture: …` / `Deterministic booking: slot N booked`

---

## Demo 1 — Happy path (Story 1 routing + Story 2 booking) ✅ your current script

Caller:
1. "Ich hatte einen Autounfall." (I had a car accident)
2. "Verkehrsunfall." (traffic accident — answers the area-confirm + matter type)
3. "Versicherungsnummer F 6 2 3 1 4 7 5 8." (insurance number)
4. "Mein Name ist Moritz Lang."
5. "Ja, das ist korrekt." (confirm the name read-back)
6. "Die E-Mail-Adresse ist lang at gmail punkt com."
7. "Ja, das ist korrekt." (confirm the email read-back)
8. "0151 597 386 94." (phone)
9. "Ja, ist korrekt." (confirm the phone read-back)
10. "10. Juni um 10 Uhr." (pick a slot)

Proves: auto-routing to **traffic**, area-specific question (insurance, not a
generic script), name/email/phone capture with read-back, a real DB booking.
Point at: `Auto-route: traffic` → scripted `traffic_confirm` → `Deterministic
booking: slot N booked` → "Ihr Termin ist gebucht…". Then show the DB row:
```bash
cd backend && sqlite3 -header -column data/voice_agent.db \
 "SELECT s.date,s.time,s.lawyer_name,b.caller_name,b.caller_email,b.caller_phone \
  FROM bookings b JOIN slots s ON s.id=b.slot_id ORDER BY b.id DESC LIMIT 1;"
```

## Demo 2 — Different law type (Story 1: it branches)

Caller:
1. "Mein Vermieter zahlt meine Kaution nicht zurück." (landlord won't return deposit)
2. "Ja, es geht um die Kaution." (deposit — matter type)
3. "Haben wir schriftlich angemahnt, ja." (answers the **tenancy-specific** follow-up)
4. … then name / email / phone / slot as in Demo 1.

Proves: a **different** area (tenancy) asks **different** questions — tenancy asks
"haben Sie es schriftlich angezeigt?" where traffic asked for an insurance number.
That's the branching conversation design, not one-size-fits-all.

## Demo 3 — Unavailable / declined slot (Story 2 second half)

Run Demo 1 up to the slot offer, then:
- At "Welcher passt Ihnen?" say: **"Die passen mir nicht, haben Sie etwas anderes?"**
- The agent offers a **different** batch of times; pick one.

Proves: `_SLOT_DECLINE_RE` → re-offer excluding the declined `(date, time)` pairs
(so it never re-offers the same time via another lawyer's slot). Point at the two
different `FAST PATH: Ich kann Ihnen folgende Termine anbieten: …` lines.
Tip: slots are real DB reads; to make consumption visible, the seed has limited
times per day, so repeated declines roll the offer forward.

## Demo 4 — Low-confidence / unsure capture (Story 3)

Run to the email step, then deliberately make it hard:
1. (email step) say just **"Lang M"** — not a parseable address.
   → agent: "Ich habe die E-Mail-Adresse leider nicht verstanden. Können Sie sie bitte nochmal sagen?"
2. say **"Lang M at gmail punkt com"** clearly → agent reads it back: "Ich habe notiert: langm@gmail.com — ist das korrekt?"
3. "Nein." → it drops the value and re-asks. Then say it correctly and confirm.

Also note (happens naturally): the **name** read-back fires on low STT confidence
("name='…' (low conf 0.73 → confirm)"). And after 3 unparseable email attempts the
agent **skips email** and moves to phone (a phone number is enough to book).

Proves: the unsure path — re-prompt, spell/read back, confirm, and graceful skip —
not just the happy path.

## Demo 5 — Handoff (Story 4), three variants

**5a — caller asks for a specific lawyer → booked WITH that lawyer (the headline):**
1. "Ich möchte mit Frau Hoffmann sprechen." (I'd like to speak with Ms Hoffmann)
   → agent: can't transfer live, but will arrange a consultation with Frau Hoffmann;
   asks what it's about.
2. "Es geht um einen Autounfall." (Frau Hoffmann is a traffic lawyer — pick a matter she
   handles so her slots exist), then proceed through the normal capture.
3. At the slot offer, **only Frau Hoffmann's slots** are offered, and the confirmation
   reads "Ihr Termin mit Frau Hoffmann ist gebucht: …".
   Point at: `target_person` set, `_available_slots` filters by lawyer (graceful fallback
   to any lawyer if she's full), `booking_done_person` names her. The DB `bookings` row
   joins to her `slots.lawyer_name`.

**5b — explicit callback (the other path):**
1. "Bitte rufen Sie mich zurück." (please call me back)
   → recorded as a **callback**: name → phone → preferred time → "… meldet sich bei Ihnen."
   (`wants_callback` → `callback_requested`). This is distinct from asking for a lawyer.

**5c — out of scope:**
1. "Ich brauche Hilfe bei einer Scheidung." (divorce — family law, not handled)
   → `route_call(legal_area="unknown")` sets `escalation_requested=True` → ESCALATION;
   the agent explains the firm handles employment/tenancy/traffic and offers to pass it on.
   Note: "Scheidung" matches no routing keyword, so this path depends on the **LLM**
   classifying it as `unknown` in the ROUTING phase — reliable on cloud, occasionally
   needs a clearer cue on the local 7B. **5a and 5b are deterministic**; lead with 5a
   (book-with-lawyer) and use 5c as the "out-of-scope" illustration.

Proves: when it triggers (explicit request, out-of-scope area, or 3 consecutive
misunderstandings) and that the agent **never claims a live transfer**
(`_guard_impossible_handoff` blocks "ich verbinde Sie").

---

## Honest limitations to mention on camera
- **Spoken email** over local Whisper is the weak field — regex now handles clean
  dictation; heavy STT noise is mitigated by read-back + re-ask + 3-try skip, and a
  cloud LLM rescue (`email_extract.py`) when an OpenAI key is configured.
- **STT is non-streaming** (faster-whisper, ~1.8s/turn locally); Piper TTS is
  functional but synthetic. Both swap to cloud (Deepgram/ElevenLabs) via env.
- In the **scripted phases** an off-script remark (e.g. asking for legal advice mid-
  capture) is answered by re-asking the current field — the deliberate
  reliability-over-flexibility trade-off for a local 7B. The legal-advice boundary
  is LLM-enforced in the ROUTING/INFORMATION phases.
