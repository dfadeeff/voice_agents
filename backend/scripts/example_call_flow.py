"""Example call flow demonstrating what the voice agent should achieve.

Based on the Jupus "Telefon-KI Marie" reference recording, this script
simulates two realistic German law firm intake calls:

1. EXISTING CLIENT — caller wants to speak to a specific lawyer about a
   pending case (Klageschrift). Agent can't connect directly but captures
   the request, contact info, and case reference for callback.

2. NEW CLIENT — employment law caller (unfair dismissal). Full intake flow:
   greeting -> intent -> routing -> intake follow-ups -> contact capture ->
   conflict check (employer + insurance) -> additional info -> booking.

Run: python scripts/example_call_flow.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.conversation.manager import ConversationManager  # noqa: E402
from app.conversation.summary import generate_call_summary  # noqa: E402
from app.services.calendar import CalendarService  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402

DB = "data/voice_agent.db"


def header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def marie(text: str) -> None:
    print(f"  Marie:   {text}")


def caller(text: str) -> None:
    print(f"  Caller:  {text}")


def system(text: str) -> None:
    print(f"  [{text}]")


async def scenario_existing_client(registry) -> None:
    """Scenario 1: Existing client wants callback from specific lawyer.

    Reference: Jupus Marie demo recording
    - Caller: Amelie Sommer, existing client (Mandantin)
    - Wants to speak to Herr Richter about a Klageschrift
    - Marie can't transfer but captures info for callback
    - Collects: name, phone (from display), email, case reference (Aktenzeichen)
    """
    header("SCENARIO 1: Existing Client — Callback Request")

    ctx = ConversationManager(call_id="example-existing-client", lang="de")

    # --- Greeting ---
    marie(
        "Guten Tag, hier ist die Anrufannahme der Kanzlei. "
        "Ich nehme gerne Ihr Anliegen auf und leite es an unser Team weiter. "
        "Wobei können wir Ihnen behilflich sein?"
    )
    system(f"phase: {ctx.state.phase.value}")
    print()

    # --- Caller asks for specific lawyer ---
    caller("Hallo, Amelie Sommer hier. Ich hätte gerne einmal den Herrn Richter gesprochen, bitte.")
    ctx.add_user_message(
        "Hallo, Amelie Sommer hier. Ich hätte gerne einmal den Herrn Richter gesprochen, bitte."
    )
    system(f"phase: {ctx.state.phase.value}")
    print()

    # --- Marie explains she can't transfer, asks about the matter ---
    marie(
        "Es tut mir leid, Frau Sommer, aber ich kann Sie nicht "
        "direkt zu Herrn Richter verbinden. Ich nehme jedoch gerne "
        "Ihr Anliegen und Ihre Informationen auf und leite sie weiter, "
        "sodass sich die Experten schnellstmöglich bei Ihnen melden können. "
        "Worum geht es in Ihrem Anliegen?"
    )

    # Agent classifies: caller wants to reach a specific person -> escalation path
    # But first, understand the situation
    await registry.execute("classify_caller_intent", {"intent": "book_consultation"}, ctx)
    system(f"intent: {ctx.state.caller_intent.value} -> phase: {ctx.state.phase.value}")
    print()

    # --- Caller explains ---
    caller(
        "Ich bin Mandantin bei Ihnen und Herr Richter hatte mir "
        "eine Klageschrift zugesendet, zu der ich jetzt ein paar Rückfragen habe."
    )
    ctx.add_user_message(
        "Ich bin Mandantin bei Ihnen und Herr Richter hatte mir "
        "eine Klageschrift zugesendet, zu der ich Rückfragen habe."
    )
    print()

    # Route and capture intake
    await registry.execute("classify_legal_area", {"legal_area": "employment"}, ctx)
    await registry.execute(
        "complete_intake",
        {"summary": "Bestandsmandat, Rückfragen zur zugesendeten Klageschrift von Herrn Richter"},
        ctx,
    )
    system(f"intake complete -> phase: {ctx.state.phase.value}")
    print()

    # --- Marie asks for contact info ---
    marie(
        "Vielen Dank für die Rückmeldung, Frau Sommer. "
        "Ich werde Ihr Anliegen sofort weiterleiten. "
        "Können Sie mir bitte Ihre Telefonnummer und E-Mail-Adresse geben, "
        "damit wir Sie erreichen können?"
    )

    # --- Caller gives contact ---
    caller(
        "Gerne. Die Telefonnummer ist die, die Sie auf dem Display sehen "
        "und meine E-Mail-Adresse ist amelie.sommer@gmail.com."
    )
    ctx.add_user_message(
        "Die Telefonnummer ist die auf dem Display "
        "und meine E-Mail-Adresse ist amelie.sommer@gmail.com."
    )
    await registry.execute("extract_caller_details", {"name": "Amelie Sommer"}, ctx)
    await registry.execute("extract_caller_details", {"email": "amelie.sommer@gmail.com"}, ctx)
    await registry.execute("extract_caller_details", {"phone": "+4915112345678"}, ctx)
    for f in ("name", "email", "phone"):
        ctx.confirm_entity(f)
    system(f"contact captured -> phase: {ctx.state.phase.value}")
    print()

    # --- Conflict check (existing client, skip insurance question) ---
    ctx.state.employer_name = "Bestandsmandat"
    ctx.state.has_legal_insurance = None
    # For existing clients, skip the insurance question
    ctx.state.has_legal_insurance = True
    ctx.advance_phase()
    system(f"conflict check skipped (existing client) -> phase: {ctx.state.phase.value}")

    # --- Additional info: Aktenzeichen ---
    marie("Darf ich außerdem noch nach dem Aktenzeichen für Ihren Fall fragen?")
    caller("Ja, das ist die 44 von 24.")
    ctx.add_user_message("Das Aktenzeichen ist 44 von 24.")
    await registry.execute("record_additional_info", {"notes": "Aktenzeichen: 44/24"}, ctx)
    system(f"additional info -> phase: {ctx.state.phase.value}")
    print()

    # --- For existing client requesting callback, escalate rather than book ---
    marie(
        "Danke, Frau Sommer. Ich werde alle Informationen an unser Team weiterleiten. "
        "Herr Richter oder ein Mitglied des Teams wird sich so schnell wie möglich "
        "bei Ihnen melden."
    )
    print()

    marie("Haben Sie noch weitere Fragen oder Anliegen, die ich notieren soll?")
    caller("Nein, danke. Das war's.")
    print()

    # Escalate for human callback
    await registry.execute(
        "escalate_to_human",
        {
            "reason": "existing_client_callback",
            "summary": (
                "Bestandsmandat Amelie Sommer, Rückfragen zur Klageschrift, "
                "Az 44/24, bittet um Rückruf von Herrn Richter"
            ),
        },
        ctx,
    )
    system(f"escalated for callback -> phase: {ctx.state.phase.value}")
    print()

    summary = generate_call_summary(ctx.state)
    print("  Call Summary:")
    print(f"    Caller:  {summary['caller_name']}")
    print(f"    Email:   {summary['email']}")
    print(f"    Phone:   {summary['phone']}")
    print(f"    Area:    {summary['legal_area']}")
    print(f"    Issue:   {summary['issue_summary']}")
    print(f"    Notes:   {summary['additional_notes']}")
    print(f"    Outcome: escalation -> {summary['escalation']}")


async def scenario_new_client_booking(registry) -> None:
    """Scenario 2: New client with employment issue, full booking flow.

    Demonstrates the complete Marie-inspired intake:
    - Warm greeting, formal "Sie"
    - Intent detection (book consultation)
    - Legal area routing (employment / Arbeitsrecht)
    - Intake follow-ups (what happened, timeline, documents)
    - Contact capture with spelling confirmation
    - Conflict check (employer name + Rechtsschutzversicherung)
    - Additional info catch-all
    - Availability check + booking
    - Confirmation with all details
    """
    header("SCENARIO 2: New Client — Employment Law, Full Booking")

    ctx = ConversationManager(call_id="example-new-client", lang="de")

    # --- 1. Greeting ---
    marie(
        "Guten Tag, hier ist die Anrufannahme der Kanzlei. "
        "Ich nehme gerne Ihr Anliegen auf und leite es an unser Team weiter. "
        "Sie können mit mir in ganzen Sätzen sprechen. "
        "Wobei können wir Ihnen behilflich sein?"
    )
    system(f"phase: {ctx.state.phase.value}")
    print()

    # --- 2. Caller describes situation ---
    caller(
        "Hallo, ich habe ein Problem mit meinem Arbeitgeber. "
        "Ich wurde letzte Woche gekündigt und ich glaube, "
        "das war nicht rechtmäßig."
    )
    ctx.add_user_message(
        "Ich habe ein Problem mit meinem Arbeitgeber. "
        "Ich wurde letzte Woche gekündigt und glaube das war nicht rechtmäßig."
    )
    system(f"phase: {ctx.state.phase.value}")
    print()

    # --- 3. Intent classification ---
    marie(
        "Das tut mir leid zu hören. Ich kann Ihr Anliegen aufnehmen "
        "und an unsere Experten weiterleiten. Möchten Sie einen "
        "Beratungstermin mit einem unserer Anwälte vereinbaren?"
    )
    caller("Ja, das wäre gut.")
    ctx.add_user_message("Ja, das wäre gut.")
    await registry.execute("classify_caller_intent", {"intent": "book_consultation"}, ctx)
    system(f"intent: {ctx.state.caller_intent.value} -> phase: {ctx.state.phase.value}")
    print()

    # --- 4. Legal area routing ---
    await registry.execute("classify_legal_area", {"legal_area": "employment"}, ctx)
    system(f"routed to: {ctx.state.legal_area.value} -> phase: {ctx.state.phase.value}")
    print()

    # --- 5. Intake: situation-specific follow-ups ---
    marie(
        "Verstanden, es geht um eine Kündigung. Damit unsere Anwälte "
        "sich optimal vorbereiten können, hätte ich noch ein paar Fragen. "
        "Wie lange waren Sie bei dem Unternehmen beschäftigt?"
    )
    caller("Seit fünf Jahren, also seit 2021.")
    ctx.add_user_message("Seit fünf Jahren, seit 2021.")
    print()

    marie("Wurden Ihnen Gründe für die Kündigung genannt?")
    caller(
        "Nein, es hieß nur, meine Stelle würde wegfallen. "
        "Aber ein Kollege hat mir erzählt, dass sie jemand Neues einstellen."
    )
    ctx.add_user_message(
        "Nein, es hieß nur meine Stelle würde wegfallen. "
        "Aber ein Kollege sagte sie stellen jemand Neues ein."
    )
    print()

    marie("Haben Sie die Kündigung schriftlich erhalten?")
    caller("Ja, per Brief letzte Woche Mittwoch.")
    ctx.add_user_message("Ja, per Brief letzte Woche Mittwoch.")
    print()

    await registry.execute(
        "complete_intake",
        {
            "summary": (
                "Kündigung nach 5 Jahren Betriebszugehörigkeit (seit 2021). "
                "Begründung: Stelle falle weg, aber Hinweise dass Neueinstellung geplant. "
                "Kündigung schriftlich erhalten, letzte Woche Mittwoch. "
                "Mögliche Kündigungsschutzklage, 3-Wochen-Frist beachten."
            )
        },
        ctx,
    )
    system(f"intake complete -> phase: {ctx.state.phase.value}")
    print()

    # --- 6. Contact capture ---
    marie(
        "Vielen Dank für die Schilderung. Ich brauche jetzt Ihre "
        "Kontaktdaten, damit wir uns bei Ihnen melden können. "
        "Wie ist Ihr vollständiger Name?"
    )
    caller("Thomas Wilhelm Becker.")
    ctx.add_user_message("Thomas Wilhelm Becker")
    await registry.execute("extract_caller_details", {"name": "Thomas Wilhelm Becker"}, ctx)
    print()

    marie("Danke. Das ist T-H-O-M-A-S W-I-L-H-E-L-M B-E-C-K-E-R, richtig?")
    caller("Ja, genau.")
    ctx.confirm_entity("name")
    print()

    marie("Können Sie mir Ihre E-Mail-Adresse geben? Bitte buchstabieren Sie sie zur Sicherheit.")
    caller("T-H-O-M-A-S Punkt B-E-C-K-E-R at G-M-X Punkt D-E.")
    ctx.add_user_message("thomas punkt becker at gmx punkt de")
    await registry.execute("extract_caller_details", {"email": "thomas.becker@gmx.de"}, ctx)
    print()

    marie("thomas Punkt becker at gmx Punkt de — ist das korrekt?")
    caller("Ja.")
    ctx.confirm_entity("email")
    print()

    marie("Und Ihre Telefonnummer, bitte Ziffer für Ziffer?")
    caller("0, 1, 7, 6, 3, 4, 5, 6, 7, 8, 9, 0.")
    ctx.add_user_message("0 1 7 6 3 4 5 6 7 8 9 0")
    await registry.execute("extract_caller_details", {"phone": "+4917634567890"}, ctx)
    print()

    marie("Ich wiederhole: 0, 1, 7, 6, 3, 4, 5, 6, 7, 8, 9, 0. Stimmt das?")
    caller("Ja, richtig.")
    ctx.confirm_entity("phone")
    system(f"all confirmed -> phase: {ctx.state.phase.value}")
    print()

    # --- 7. Conflict check ---
    marie(
        "Bei welchem Unternehmen waren Sie beschäftigt? "
        "Ich frage, damit wir einen möglichen Interessenkonflikt "
        "ausschließen können."
    )
    caller("Bei der Müller und Partner GmbH.")
    ctx.add_user_message("Bei der Müller und Partner GmbH.")
    print()

    marie("Haben Sie eine Rechtsschutzversicherung?")
    caller("Ja, bei der ARAG.")
    ctx.add_user_message("Ja, bei der ARAG.")
    await registry.execute(
        "record_conflict_info",
        {"employer_name": "Müller und Partner GmbH", "has_legal_insurance": True},
        ctx,
    )
    system(f"conflict check -> phase: {ctx.state.phase.value}")
    print()

    # --- 8. Additional info ---
    marie("Gibt es sonst noch etwas, das Sie uns mitteilen möchten?")
    caller("Ja, ich bin schwerbehindert, Grad 50. Ich weiß nicht, ob das eine Rolle spielt.")
    ctx.add_user_message("Ich bin schwerbehindert, Grad 50.")
    await registry.execute(
        "record_additional_info",
        {"notes": "Schwerbehindert (GdB 50) — besonderer Kündigungsschutz prüfen"},
        ctx,
    )
    system(f"additional info -> phase: {ctx.state.phase.value}")
    print()

    # --- 9. Booking ---
    marie(
        "Das ist eine wichtige Information, vielen Dank. "
        "Wann würde Ihnen ein Beratungstermin passen?"
    )
    caller("Am besten so schnell wie möglich, vielleicht Montag oder Dienstag?")
    ctx.add_user_message("So schnell wie möglich, Montag oder Dienstag.")
    print()

    result = await registry.execute(
        "check_availability",
        {"date": "2026-06-09", "legal_area": "employment", "time_preference": "morning"},
        ctx,
    )

    if result.get("available") and result["slots"]:
        slot = result["slots"][0]
        marie(
            f"Am Montag, den 9. Juni, hätten wir einen Termin um {slot['time']} Uhr "
            f"bei {slot['lawyer']}. Passt Ihnen das?"
        )
        caller("Ja, das passt perfekt.")
        ctx.add_user_message("Ja, das passt.")
        print()

        result = await registry.execute(
            "book_consultation",
            {"slot_id": slot["id"], "caller_name": "Thomas Wilhelm Becker"},
            ctx,
        )
        system(f"booked -> phase: {ctx.state.phase.value}")
    print()

    # --- 10. Confirmation ---
    summary = generate_call_summary(ctx.state)
    booking = summary.get("booking", {})
    marie(
        f"Wunderbar, Ihr Termin ist bestätigt: "
        f"{booking.get('date', 'N/A')} um {booking.get('time', 'N/A')} Uhr "
        f"bei {booking.get('lawyer', 'N/A')}. "
        f"Ich habe alle Informationen aufgenommen und leite sie an unser Team weiter. "
        f"Vielen Dank für Ihren Anruf, Herr Becker, und alles Gute!"
    )
    caller("Vielen Dank, auf Wiederhören!")
    print()

    print("  " + "-" * 56)
    print("  CALL SUMMARY")
    print("  " + "-" * 56)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


async def main() -> None:
    calendar = CalendarService(DB)
    await calendar.init_db()
    registry = build_default_registry(calendar)

    await scenario_existing_client(registry)
    await scenario_new_client_booking(registry)

    header("KEY BEHAVIORS DEMONSTRATED")
    print(
        """\
  From the Jupus Marie reference recording:

  1. WARM GREETING — "Guten Tag, hier ist die Anrufannahme der Kanzlei..."
     Agent introduces itself, explains it will forward the request,
     and invites the caller to speak naturally.

  2. CAN'T TRANSFER — When caller asks for a specific person, Marie
     politely explains she can't connect directly but will take the
     request and forward it. Never promises things she can't do.

  3. FORMAL GERMAN — Uses "Sie" throughout. Addresses caller by last
     name ("Frau Sommer"). Professional but warm tone.

  4. SITUATION CAPTURE — Asks follow-up questions based on the legal
     area: timeline, reasons given, documentation, special circumstances.

  5. CONTACT SPELLING — Asks caller to spell email letter by letter.
     Reads back phone numbers digit by digit. Confirms everything.

  6. CONFLICT CHECK — Asks for employer name (to check for conflicts
     of interest) and whether caller has Rechtsschutzversicherung.

  7. CASE REFERENCE — For existing clients, asks for Aktenzeichen.
     This maps to our additional_notes field.

  8. CATCH-ALL — "Haben Sie noch weitere Fragen oder Anliegen?"
     Always gives the caller one more chance to add info.

  9. HANDOFF — "Ich leite alle Informationen an unser Team weiter.
     Herr Richter wird sich schnellstmöglich bei Ihnen melden."
     Clear expectation setting about what happens next.

  10. NO LEGAL ADVICE — Marie never opines on the case merits.
      Only captures facts and routes to the right lawyer.\
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
