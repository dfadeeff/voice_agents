from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["general_info", "book_consultation"],
            "description": (
                "What the caller wants: 'general_info' if they want to know"
                " whether the firm can help, 'book_consultation' if they want"
                " to schedule a meeting with a lawyer."
            ),
        },
    },
    "required": ["intent"],
}


async def classify_caller_intent(args: dict, ctx: ConversationManager) -> dict:
    intent_str = args.get("intent", "")
    if not intent_str:
        return {"status": "need_more_info"}
    try:
        intent = CallerIntent(intent_str)
    except ValueError:
        intent = CallerIntent.UNKNOWN

    ctx.set_intent(intent)
    return {"status": "classified", "intent": intent.value}


def register_intent_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="classify_caller_intent",
        fn=classify_caller_intent,
        description=(
            "Classify what the caller wants: general information about whether the firm "
            "can help, or to book a consultation. Call this early in the conversation "
            "once you understand the caller's purpose."
        ),
        parameters=SCHEMA,
    )
