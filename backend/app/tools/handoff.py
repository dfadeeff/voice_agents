from app.conversation.manager import ConversationManager
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
            "description": "Why the call is being handed off.",
        },
        "summary": {
            "type": "string",
            "description": "Brief context summary for the team member receiving the handoff.",
        },
    },
    "required": ["reason"],
}


async def request_handoff(args: dict, ctx: ConversationManager) -> dict:
    ctx.state.escalation_requested = True
    ctx.state.escalation_reason = args.get("reason", "unknown")
    ctx.state.escalation_summary = args.get("summary", "")
    ctx.advance_phase()

    context_for_human = {
        "reason": args.get("reason", "unknown"),
        "summary": args.get("summary", ""),
        "caller_details": {k: v.value for k, v in ctx.state.entities.items()},
        "legal_area": ctx.state.legal_area.value,
        "matter_summary": ctx.state.matter_summary,
        "turn_count": ctx.state.turn_count,
        "call_id": ctx.state.call_id,
    }

    return {
        "status": "handing_off",
        "context_for_human": context_for_human,
    }


def register_handoff_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="request_handoff",
        fn=request_handoff,
        description=(
            "Hand the call off to a team member. Use when: "
            "(1) the caller explicitly asks for a person, "
            "(2) the legal area is not supported (employment, tenancy, traffic), "
            "(3) you've failed to understand the caller multiple times, "
            "(4) the situation is too complex for automated handling."
        ),
        parameters=SCHEMA,
    )
