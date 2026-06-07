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

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[\d\s\-+()]{7,20}$")

ALWAYS_CONFIRM = ("email", "phone")


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

        existing = ctx.state.entities.get(field_name)
        if existing and existing.value == value and not existing.confirmed:
            ctx.confirm_entity(field_name)
            stored.append(field_name)
            continue

        ctx.store_entity(field_name, value, 0.9)
        stored.append(field_name)

        if field_name in ALWAYS_CONFIRM:
            if field_name == "email":
                suggestion = f"Please read back the email address letter by letter: '{value}'"
            else:
                suggestion = f"Please repeat the phone number digit by digit: '{value}'"
            needs_confirmation.append(
                {"field": field_name, "value": value, "suggestion": suggestion}
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


def register_extraction_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="extract_caller_details",
        fn=extract_caller_details,
        description=(
            "Store caller details extracted from conversation. "
            "Email and phone always require verbal confirmation — "
            "read them back and call this tool again with the same value "
            "after the caller confirms. Names are confirmed automatically."
        ),
        parameters=SCHEMA,
    )
