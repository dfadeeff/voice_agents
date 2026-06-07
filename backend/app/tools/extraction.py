import re

from app.conversation.manager import ConversationManager
from app.tools.registry import ToolRegistry

SCHEMA = {
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
        "preferred_date": {
            "type": "string",
            "description": "Preferred consultation date (YYYY-MM-DD or natural language).",
        },
        "preferred_time": {
            "type": "string",
            "description": "Preferred time (morning/afternoon or specific time like 2pm).",
        },
        "matter_description": {
            "type": "string",
            "description": "Brief description of the legal matter.",
        },
    },
}

CONFIDENCE_THRESHOLD = 0.7

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[\d\s\-+()]{7,20}$")


def _validate_format(field_name: str, value: str) -> str | None:
    if field_name == "email" and not EMAIL_RE.match(value):
        return f"'{value}' does not look like a valid email. Ask the caller to repeat it."
    if field_name == "phone" and not PHONE_RE.match(value.replace(" ", "")):
        return f"'{value}' does not look like a valid phone number. Ask the caller to repeat it."
    return None


async def extract_caller_details(args: dict, ctx: ConversationManager) -> dict:
    stored = []
    needs_confirmation = []
    format_errors = []

    for field_name, value in args.items():
        if not value or not isinstance(value, str):
            continue

        validation_error = _validate_format(field_name, value)
        if validation_error:
            format_errors.append({"field": field_name, "error": validation_error})
            continue

        confidence = ctx.get_word_confidence_for_value(value)
        ctx.store_entity(field_name, value, confidence)
        stored.append(field_name)

        if confidence < CONFIDENCE_THRESHOLD and field_name in ("name", "email", "phone"):
            if field_name == "name":
                suggestion = f"Please confirm the caller's name by spelling it back: '{value}'"
            elif field_name == "email":
                suggestion = f"Please read back the email address letter by letter: '{value}'"
            else:
                suggestion = f"Please repeat the phone number digit by digit: '{value}'"

            needs_confirmation.append(
                {
                    "field": field_name,
                    "value": value,
                    "confidence": round(confidence, 2),
                    "suggestion": suggestion,
                }
            )
        elif field_name in ("name", "email", "phone"):
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


def register_extraction_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="extract_caller_details",
        fn=extract_caller_details,
        description=(
            "Store caller details extracted from conversation. "
            "The system will check STT confidence and format validity, "
            "and tell you which fields need verbal confirmation. "
            "You MUST confirm any low-confidence fields before booking."
        ),
        parameters=SCHEMA,
    )
