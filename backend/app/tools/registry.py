from collections.abc import Callable
from typing import Any

from app.conversation.manager import ConversationManager


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable, dict]] = {}

    def register(
        self, name: str, fn: Callable, description: str, parameters: dict
    ) -> None:
        schema = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._tools[name] = (fn, schema)

    def get_schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": schema}
            for _, (_, schema) in self._tools.items()
        ]

    async def execute(
        self, name: str, arguments: dict, ctx: ConversationManager
    ) -> dict:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        fn, _ = self._tools[name]
        return await fn(arguments, ctx)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


def build_default_registry(calendar_service: Any) -> ToolRegistry:
    from app.tools.booking import register_booking_tools
    from app.tools.escalation import register_escalation_tools
    from app.tools.extraction import register_extraction_tools
    from app.tools.intent import register_intent_tools
    from app.tools.routing import register_routing_tools

    registry = ToolRegistry()
    register_intent_tools(registry)
    register_routing_tools(registry)
    register_extraction_tools(registry)
    register_booking_tools(registry, calendar_service)
    register_escalation_tools(registry)
    return registry