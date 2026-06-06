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


async def extract_caller_details(
    args: dict, ctx: ConversationManager
) -> dict:
    stored = []
    needs_confirmation = []

    for field_name, value in args.items():
        if not value or not isinstance(value, str):
            continue

        confidence = ctx.get_word_confidence_for_value(value)
        entity = ctx.store_entity(field_name, value, confidence)
        stored.append(field_name)

        if confidence < CONFIDENCE_THRESHOLD and field_name in (
            "name",
            "email",
            "phone",
        ):
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

    return {
        "stored": stored,
        "needs_confirmation": needs_confirmation,
        "all_confirmed": len(needs_confirmation) == 0,
    }


def register_extraction_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="extract_caller_details",
        fn=extract_caller_details,
        description=(
            "Store caller details extracted from conversation. "
            "The system will check STT confidence and tell you which fields need verbal confirmation. "
            "You MUST confirm any low-confidence fields before booking."
        ),
        parameters=SCHEMA,
    )