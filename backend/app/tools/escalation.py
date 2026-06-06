from app.conversation.manager import ConversationManager
from app.models.schemas import CallPhase
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "enum": [
                "caller_requested_human",
                "out_of_scope_legal_area",
                "repeated_misunderstanding",
                "complex_situation",
                "caller_frustrated",
            ],
            "description": "Why the call is being escalated.",
        },
        "summary": {
            "type": "string",
            "description": "Brief context summary for the human agent receiving the transfer.",
        },
    },
    "required": ["reason"],
}


async def escalate_to_human(
    args: dict, ctx: ConversationManager
) -> dict:
    ctx.state.escalation_requested = True
    ctx.state.phase = CallPhase.ESCALATION

    context_for_human = {
        "reason": args.get("reason", "unknown"),
        "summary": args.get("summary", ""),
        "caller_details": {
            k: v.value for k, v in ctx.state.entities.items()
        },
        "legal_area": ctx.state.legal_area.value,
        "turn_count": ctx.state.turn_count,
        "call_id": ctx.state.call_id,
    }

    return {
        "status": "escalating",
        "message": (
            "I'm going to connect you with a member of our team"
            " who can help you directly. Please hold for just a moment."
        ),
        "context_for_human": context_for_human,
    }


def register_escalation_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="escalate_to_human",
        fn=escalate_to_human,
        description=(
            "Transfer the call to a human agent. Use when: "
            "(1) caller explicitly asks for a person, "
            "(2) the legal area is not employment or tenancy, "
            "(3) you've failed to understand the caller 3+ times, "
            "(4) the situation is too complex for automated handling, "
            "(5) the caller is frustrated or distressed."
        ),
        parameters=SCHEMA,
    )