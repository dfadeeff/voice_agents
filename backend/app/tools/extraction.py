import re

from app.conversation.manager import ConversationManager
from app.conversation.phone import normalize_phone_text
from app.tools.registry import ToolRegistry

CAPTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Caller's full name as heard.",
        },
        "email": {
            "type": "string",
            "description": "Caller's email address as heard.",
        },
        "phone": {
            "type": "string",
            "description": "Caller's phone number as heard.",
        },
        "matter_type": {
            "type": "string",
            "description": (
                "Type of legal matter within the area of law. "
                "Employment: 'dismissal', 'warning', 'wages', 'contract', 'other'. "
                "Tenancy: 'eviction', 'deposit', 'rent_increase', 'repairs', 'other'. "
                "Traffic: 'accident', 'damage', 'insurance', 'other'."
            ),
        },
        "matter_details": {
            "type": "string",
            "description": (
                "Follow-up details about the matter. "
                "Employment: whether there is a deadline "
                "(e.g. 3-week dismissal protection period). "
                "Tenancy: whether the issue was reported in writing. "
                "Traffic: whether police were involved or "
                "there is a reference/claim number."
            ),
        },
        "case_reference": {
            "type": "string",
            "description": (
                "Existing case reference number (Aktenzeichen) "
                "if the caller is an existing client."
            ),
        },
        "insurance_number": {
            "type": "string",
            "description": (
                "Insurance claim or policy number "
                "(Versicherungsnummer/Schadensnummer) for traffic cases."
            ),
        },
        "preferred_date": {
            "type": "string",
            "description": "Preferred consultation date (YYYY-MM-DD or natural language).",
        },
        "preferred_time": {
            "type": "string",
            "description": "Preferred time (morning/afternoon or specific time like 14:00).",
        },
    },
}

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "field": {
            "type": "string",
            "enum": ["name", "email", "phone"],
            "description": "Which field is being confirmed.",
        },
        "confirmed_value": {
            "type": "string",
            "description": (
                "The value being confirmed, or the corrected value if the caller made a correction."
            ),
        },
        "status": {
            "type": "string",
            "enum": ["accepted", "corrected"],
            "description": (
                "'accepted' if the caller confirmed the read-back was correct, "
                "'corrected' if they provided a different value."
            ),
        },
    },
    "required": ["field", "confirmed_value", "status"],
}

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[\d\s\-+()]{7,20}$")

ALWAYS_CONFIRM = ("email", "phone")
LOW_CONFIDENCE_THRESHOLD = 0.75


def _validate_format(field_name: str, value: str) -> str | None:
    if field_name == "email" and not EMAIL_RE.match(value):
        return f"'{value}' does not look like a valid email. Ask the caller to repeat it."
    if field_name == "phone" and not PHONE_RE.match(value.replace(" ", "")):
        return f"'{value}' does not look like a valid phone number. Ask the caller to repeat it."
    return None


async def capture_caller_details(args: dict, ctx: ConversationManager) -> dict:
    stored = []
    needs_confirmation = []
    format_errors = []

    for field_name, value in args.items():
        if not value or not isinstance(value, str):
            continue

        if field_name == "phone":
            normalized = normalize_phone_text(value)
            if normalized:
                value = normalized

        validation_error = _validate_format(field_name, value)
        if validation_error:
            format_errors.append({"field": field_name, "error": validation_error})
            continue

        confidence = ctx.state.last_transcription_confidence
        if confidence is None:
            confidence = 0.9
        ctx.store_entity(field_name, value, confidence)
        stored.append(field_name)

        low_confidence_contact = (
            field_name in ("name", "email", "phone") and confidence < LOW_CONFIDENCE_THRESHOLD
        )
        if field_name in ALWAYS_CONFIRM or low_confidence_contact:
            if field_name == "email":
                suggestion = f"Spell back the email address letter by letter: '{value}'"
            elif field_name == "phone":
                suggestion = f"Read back the phone number digit by digit: '{value}'"
            else:
                suggestion = f"Read back the caller's name and ask them to confirm: '{value}'"
            needs_confirmation.append(
                {
                    "field": field_name,
                    "value": value,
                    "confidence": confidence,
                    "suggestion": suggestion,
                }
            )
        else:
            ctx.confirm_entity(field_name)

    if format_errors:
        ctx.record_misunderstanding()
    elif stored:
        ctx.reset_misunderstanding_streak()

    result = {
        "stored": stored,
        "needs_confirmation": needs_confirmation,
        "all_confirmed": len(needs_confirmation) == 0 and len(format_errors) == 0,
    }
    if format_errors:
        result["format_errors"] = format_errors
    return result


async def confirm_caller_detail(args: dict, ctx: ConversationManager) -> dict:
    field_name = args.get("field", "")
    confirmed_value = args.get("confirmed_value", "")
    status = args.get("status", "")

    if not field_name or not confirmed_value or status not in ("accepted", "corrected"):
        return {"status": "error", "message": "field, confirmed_value, and status are required."}

    validation_error = _validate_format(field_name, confirmed_value)
    if validation_error:
        ctx.record_misunderstanding()
        return {"status": "invalid", "error": validation_error}

    if status == "accepted":
        existing = ctx.state.entities.get(field_name)
        if not existing:
            return {"status": "error", "message": f"No stored value for '{field_name}' to confirm."}
        ctx.confirm_entity(field_name)
    else:
        ctx.update_and_confirm_entity(field_name, confirmed_value)

    ctx.reset_misunderstanding_streak()
    return {
        "status": "confirmed",
        "field": field_name,
        "value": confirmed_value,
        "was_corrected": status == "corrected",
    }


def register_extraction_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="capture_caller_details",
        fn=capture_caller_details,
        description=(
            "Store caller details extracted from the conversation. "
            "Email and phone are stored but require explicit confirmation — "
            "read them back and use confirm_caller_detail after the caller responds. "
            "Name and matter_type are confirmed automatically."
        ),
        parameters=CAPTURE_SCHEMA,
    )
    registry.register(
        name="confirm_caller_detail",
        fn=confirm_caller_detail,
        description=(
            "Confirm or correct a previously captured field after reading it back to the caller. "
            "Use 'accepted' when the caller confirms the value is correct. "
            "Use 'corrected' with the new value when the caller provides a correction."
        ),
        parameters=CONFIRM_SCHEMA,
    )
