"""Tests for the tool registry pattern."""

import pytest
from app.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        reg.register("test_tool", lambda a, c: {}, "A test", {"type": "object"})
        assert "test_tool" in reg.list_tools()

    def test_get_schemas_format(self):
        reg = ToolRegistry()
        reg.register("my_tool", lambda a, c: {}, "Does stuff", {"type": "object", "properties": {}})
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "my_tool"
        assert schemas[0]["function"]["description"] == "Does stuff"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, conversation):
        reg = ToolRegistry()
        result = await reg.execute("nonexistent", {}, conversation)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_calls_handler(self, conversation):
        reg = ToolRegistry()

        async def handler(args, ctx):
            return {"echo": args.get("msg")}

        reg.register("echo", handler, "Echo", {"type": "object"})
        result = await reg.execute("echo", {"msg": "hello"}, conversation)
        assert result == {"echo": "hello"}


class TestBuildDefaultRegistry:
    def test_all_tools_registered(self, registry):
        tools = registry.list_tools()
        assert "classify_caller_intent" in tools
        assert "classify_legal_area" in tools
        assert "extract_caller_details" in tools
        assert "check_availability" in tools
        assert "book_consultation" in tools
        assert "escalate_to_human" in tools
        assert len(tools) == 6

    def test_schemas_are_valid_openai_format(self, registry):
        schemas = registry.get_schemas()
        for schema in schemas:
            assert schema["type"] == "function"
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
