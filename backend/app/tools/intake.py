from app.conversation.manager import ConversationManager
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Brief summary of the caller's situation and legal issue.",
        },
    },
    "required": ["summary"],
}


async def complete_intake(args: dict, ctx: ConversationManager) -> dict:
    summary = args.get("summary", "").strip()
    if not summary:
        return {"status": "need_more_info", "message": "Please provide a summary of the situation."}
    ctx.state.intake_complete = True
    ctx.store_entity("matter_description", summary, 1.0)
    ctx.confirm_entity("matter_description")
    return {"status": "intake_complete", "summary": summary}


def register_intake_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="complete_intake",
        fn=complete_intake,
        description=(
            "Mark the intake as complete after gathering enough details about the caller's "
            "situation. Provide a brief summary of their issue."
        ),
        parameters=SCHEMA,
    )
