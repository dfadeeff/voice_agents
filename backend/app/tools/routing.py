from app.conversation.manager import ConversationManager
from app.models.schemas import LegalArea
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "legal_area": {
            "type": "string",
            "enum": ["employment", "tenancy", "unknown"],
            "description": "The area of law the caller's issue falls under.",
        },
    },
    "required": ["legal_area"],
}


async def classify_legal_area(args: dict, ctx: ConversationManager) -> dict:
    area_str = args.get("legal_area", "unknown")
    try:
        area = LegalArea(area_str)
    except ValueError:
        area = LegalArea.UNKNOWN

    ctx.set_legal_area(area)

    if area == LegalArea.UNKNOWN:
        return {"status": "unknown_area", "legal_area": area.value}

    return {"status": "routed", "legal_area": area.value}


def register_routing_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="classify_legal_area",
        fn=classify_legal_area,
        description=(
            "Classify the caller's legal issue into an area of law."
            " Call once you have enough context to determine the area."
        ),
        parameters=SCHEMA,
    )
