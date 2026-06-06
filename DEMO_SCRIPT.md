# Demo Script

Four call scenarios for the video walkthrough. Each shows a different capability.

## Demo 1: Employment Booking (Happy Path)

**What it shows**: Full end-to-end flow — greeting, routing, capture, booking, confirmation.

**Caller lines**:
1. "Hi, I was dismissed from my job last week and I think it was unfair."
2. "I'd like to book a consultation."
3. "My name is John Smith."
4. "My email is john dot smith at gmail dot com."
5. "My phone number is 0, 7, 9, 1, 2, 3, 4, 5, 6, 7."
6. "Tuesday morning would work."
7. "The first slot sounds good."

**Expected agent behavior**:
- Greets warmly, acknowledges the dismissal with empathy
- Calls `classify_caller_intent(intent="book_consultation")`
- Calls `classify_legal_area(legal_area="employment")`
- Asks for name, email, phone one at a time
- Reads back email and phone for confirmation
- Calls `check_availability` → presents options
- Calls `book_consultation` → confirms booking details

**What to point out**:
- Server logs show phase transitions: GREETING → INTENT_DETECTION → ROUTING → CAPTURE → BOOKING → CONFIRMATION
- Tool calls visible in logs
- Phone number read back digit-by-digit (TTS preprocessing)

---

## Demo 2: Tenancy — Unavailable Slot

**What it shows**: Legal area routing to tenancy + unavailable slot handling.

**Caller lines**:
1. "My landlord is keeping my deposit and I want to do something about it."
2. "I'd like to speak with a lawyer about this."
3. (Provide name, email, phone when asked)
4. "How about next Sunday?" (deliberately pick a date with no slots)
5. "OK, what about the first available option?"

**Expected agent behavior**:
- Routes to tenancy law
- Asks tenancy-specific follow-up questions
- Collects and confirms details
- `check_availability` returns no slots → agent offers alternatives
- Books an alternative slot

---

## Demo 3: Low-Confidence Email Confirmation

**What it shows**: STT confidence handling — agent asks to spell back when uncertain.

**Caller lines**:
1. "I need help with a workplace discrimination issue."
2. "I'd like to book a consultation."
3. "My name is Siobhan McNally." (unusual name — likely low STT confidence)
4. Spell back: "S-I-O-B-H-A-N M-C-N-A-L-L-Y"
5. "My email is siobhan dot mcnally at outlook dot com."
6. (Confirm when agent reads it back)

**Expected agent behavior**:
- `extract_caller_details` flags "Siobhan McNally" with low confidence
- Agent asks caller to spell it back
- On confirmation, entity marked as confirmed
- Agent proceeds to next field

**What to point out**:
- Tool response includes `needs_confirmation` with confidence score
- State machine stays in CAPTURE until all fields confirmed

---

## Demo 4: Human Handoff + Legal Advice Boundary

**What it shows**: Out-of-scope area escalation + legal advice refusal.

**Caller lines**:
1. "I'm going through a divorce and need help with custody of my children."
2. (Wait for agent to explain the firm doesn't handle family law)
3. "Do you think I have a strong case?"
4. "OK, can I speak to someone who can help?"

**Expected agent behavior**:
- Calls `classify_legal_area(legal_area="unknown")`
- Agent explains politely that the firm handles employment and tenancy law
- Escalation triggered — agent offers to connect with someone who can help
- When asked for legal advice, agent declines: "I can't assess the strength of your case, but I can connect you with someone who can help."
- Calls `escalate_to_human(reason="out_of_scope_legal_area")`

**What to point out**:
- `routing.py` sets `escalation_requested=True` on unknown area
- Legal advice boundary enforced by system prompt
- `context_for_human` includes all collected state for the human agent

---

## Running the Demo

```bash
# Terminal 1: start Ollama
ollama serve

# Terminal 2: start the agent
make run

# Open browser
open http://localhost:8000
```

Watch server logs for phase transitions and tool calls:
```
INFO: Phase: greeting → intent_detection
INFO: Tool call: classify_caller_intent({'intent': 'book_consultation'})
INFO: Phase advanced: intent_detection → routing
```
