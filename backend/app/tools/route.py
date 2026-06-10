from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent, LegalArea
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["general_info", "book_consultation"],
            "description": (
                "What the caller wants: 'general_info' if they want to know "
                "whether the firm can help, 'book_consultation' if they want "
                "to schedule a meeting with a lawyer."
            ),
        },
        "legal_area": {
            "type": "string",
            "enum": ["employment", "tenancy", "traffic", "unknown"],
            "description": (
                "The area of law: employment (dismissal, wages, contracts), "
                "tenancy (rent, landlord, deposit), traffic (accidents, vehicle damage), "
                "or unknown if it doesn't fit."
            ),
        },
        "matter_summary": {
            "type": "string",
            "description": "Brief one-sentence summary of the caller's legal issue.",
        },
    },
    "required": ["intent", "legal_area"],
}


async def route_call(args: dict, ctx: ConversationManager) -> dict:
    intent_str = args.get("intent", "")
    area_str = args.get("legal_area", "unknown")
    summary = args.get("matter_summary", "")

    if not intent_str:
        return {"status": "need_more_info", "message": "Could not determine caller intent."}

    try:
        intent = CallerIntent(intent_str)
    except ValueError:
        intent = CallerIntent.UNKNOWN

    try:
        area = LegalArea(area_str)
    except ValueError:
        area = LegalArea.UNKNOWN

    ctx.set_route(intent, area, summary or None)

    if area == LegalArea.UNKNOWN:
        ctx.record_escalation(reason="out_of_scope_area")
        return {
            "status": "unknown_area",
            "intent": intent.value,
            "legal_area": area.value,
        }

    return {
        "status": "routed",
        "intent": intent.value,
        "legal_area": area.value,
    }


def register_route_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="route_call",
        fn=route_call,
        description=(
            "Determine the caller's intent and legal area in one step. "
            "Call this once you understand what the caller needs and what "
            "type of legal issue they have."
        ),
        parameters=SCHEMA,
    )
