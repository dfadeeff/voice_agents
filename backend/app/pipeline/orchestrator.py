"""
Pipecat pipeline factory.

Pipecat handles: VAD, turn-taking, interruptions, streaming TTS, transport.
We handle: system prompt, tool registration, conversation state.
"""

import json
import logging

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.workers.runner import WorkerRunner

from app.conversation.manager import ConversationManager
from app.conversation.prompts import SYSTEM_PROMPT_LOCAL, SYSTEM_PROMPT_TOOLS
from app.pipeline.processors import AgentTextProcessor, CallLogger, TranscriptProcessor
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _build_tools_schema(registry: ToolRegistry) -> ToolsSchema:
    schemas = []
    for raw in registry.get_schemas():
        fn = raw["function"]
        params = fn.get("parameters", {})
        schemas.append(FunctionSchema(
            name=fn["name"],
            description=fn["description"],
            properties=params.get("properties", {}),
            required=params.get("required", []),
        ))
    return ToolsSchema(standard_tools=schemas)


def register_tools_on_llm(llm_service, registry: ToolRegistry, conversation: ConversationManager):
    """Register our tool handlers with Pipecat's LLM service."""
    for tool_name in registry.list_tools():
        def _make_handler(name):
            async def handler(params):
                args = params.arguments
                result = await registry.execute(name, dict(args), conversation)
                await params.result_callback(json.dumps(result))
            return handler

        llm_service.register_function(tool_name, _make_handler(tool_name))


async def create_pipeline(
    stt_service,
    llm_service,
    tts_service,
    transport,
    websocket,
    conversation: ConversationManager,
    tools: ToolRegistry,
    use_tools: bool = True,
) -> tuple[PipelineTask, WorkerRunner]:
    """Create a Pipecat pipeline wired with our tools."""

    if use_tools:
        register_tools_on_llm(llm_service, tools, conversation)
        system_prompt = SYSTEM_PROMPT_TOOLS
        tools_schema = _build_tools_schema(tools)
    else:
        system_prompt = SYSTEM_PROMPT_LOCAL
        tools_schema = NOT_GIVEN

    context = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "[A new caller has connected]"},
        ],
        tools=tools_schema,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    call_logger = CallLogger(conversation.state.call_id)
    transcript_proc = TranscriptProcessor(websocket, call_logger)
    agent_text_proc = AgentTextProcessor(websocket, call_logger)

    pipeline = Pipeline([
        transport.input(),
        stt_service,
        transcript_proc,
        context_aggregator.user(),
        llm_service,
        tts_service,
        agent_text_proc,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams())
    runner = WorkerRunner()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, websocket):
        await task.queue_frame(LLMContextFrame(context=context))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        call_logger.save()

    return task, runner
