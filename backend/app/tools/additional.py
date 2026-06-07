from app.conversation.manager import ConversationManager
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "string",
            "description": (
                "Any additional information the caller wants to share, "
                "or empty string if nothing to add."
            ),
        },
    },
    "required": ["notes"],
}


async def record_additional_info(args: dict, ctx: ConversationManager) -> dict:
    notes = args.get("notes", "")
    if notes is None:
        notes = ""
    ctx.state.additional_notes = notes.strip()
    ctx.advance_phase()
    return {"status": "recorded", "notes": ctx.state.additional_notes}


def register_additional_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="record_additional_info",
        fn=record_additional_info,
        description=(
            "Record any additional information the caller wants to share before booking. "
            "Pass an empty string if the caller has nothing to add."
        ),
        parameters=SCHEMA,
    )
