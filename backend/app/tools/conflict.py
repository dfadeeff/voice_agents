from app.conversation.manager import ConversationManager
from app.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "employer_name": {
            "type": "string",
            "description": "Name of the caller's employer or the counterparty.",
        },
        "has_legal_insurance": {
            "type": "boolean",
            "description": (
                "Whether the caller has legal expenses insurance (Rechtsschutzversicherung)."
            ),
        },
    },
    "required": ["employer_name", "has_legal_insurance"],
}


async def record_conflict_info(args: dict, ctx: ConversationManager) -> dict:
    employer = args.get("employer_name", "").strip()
    insurance = args.get("has_legal_insurance")
    if not employer or insurance is None:
        return {
            "status": "need_more_info",
            "message": "Please provide both employer name and insurance status.",
        }
    ctx.state.employer_name = employer
    ctx.state.has_legal_insurance = bool(insurance)
    ctx.advance_phase()
    return {
        "status": "recorded",
        "employer_name": employer,
        "has_legal_insurance": ctx.state.has_legal_insurance,
    }


def register_conflict_tools(registry: ToolRegistry) -> None:
    registry.register(
        name="record_conflict_info",
        fn=record_conflict_info,
        description=(
            "Record the caller's employer (for conflict-of-interest check) and whether "
            "they have legal expenses insurance. Ask for both before calling this tool."
        ),
        parameters=SCHEMA,
    )
