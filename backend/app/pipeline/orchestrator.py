"""
Pipecat pipeline factory.

Pipecat handles: VAD, turn-taking, interruptions, streaming TTS, transport.
We handle: state machine, system prompt per phase, tool registration + phase guards.
"""

import json
import logging

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.workers.runner import WorkerRunner

from app.conversation.locales import get_locale
from app.conversation.manager import ConversationManager
from app.conversation.prompts import build_system_prompt
from app.pipeline.processors import (
    AgentTextProcessor,
    CallLogger,
    MetricsProcessor,
    TranscriptProcessor,
)
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _build_tools_index(registry: ToolRegistry) -> dict[str, FunctionSchema]:
    """Build a name→FunctionSchema index from the registry."""
    index: dict[str, FunctionSchema] = {}
    for raw in registry.get_schemas():
        fn = raw["function"]
        params = fn.get("parameters", {})
        index[fn["name"]] = FunctionSchema(
            name=fn["name"],
            description=fn["description"],
            properties=params.get("properties", {}),
            required=params.get("required", []),
        )
    return index


def _build_tools_schema_from_names(
    names: list[str], index: dict[str, FunctionSchema]
) -> ToolsSchema | object:
    """Build a ToolsSchema containing only the named tools. Returns NOT_GIVEN if empty."""
    schemas = [index[n] for n in names if n in index]
    if not schemas:
        return NOT_GIVEN
    return ToolsSchema(standard_tools=schemas)


def register_tools_on_llm(llm_service, registry: ToolRegistry, conversation: ConversationManager):
    """Register tool handlers with phase guards."""
    for tool_name in registry.list_tools():

        def _make_handler(name):
            async def handler(params):
                allowed = conversation.get_available_tools()
                if name not in allowed:
                    logger.warning(
                        "Tool %s blocked (phase=%s, allowed=%s)",
                        name,
                        conversation.state.phase.value,
                        allowed,
                    )
                    result = {
                        "error": "This action is not available right now.",
                        "current_phase": conversation.state.phase.value,
                    }
                    await params.result_callback(json.dumps(result))
                    return

                args = params.arguments
                args_dict = dict(args)
                logger.info("Tool call: %s(%s)", name, args_dict)
                tool_call_id = getattr(params, "tool_call_id", f"tc_{name}")
                conversation.add_tool_call(
                    "",
                    [
                        {
                            "id": tool_call_id,
                            "function": {"name": name, "arguments": json.dumps(args_dict)},
                        }
                    ],
                )
                result = await registry.execute(name, args_dict, conversation)
                logger.info("Tool result: %s → %s", name, result)
                conversation.add_tool_result(tool_call_id, name, result)
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

    lang = conversation.lang
    if use_tools:
        register_tools_on_llm(llm_service, tools, conversation)
        system_prompt = build_system_prompt(conversation.state, lang)
        tools_index = _build_tools_index(tools)
        initial_tools = conversation.get_available_tools()
        tools_schema = _build_tools_schema_from_names(initial_tools, tools_index)
    else:
        system_prompt = get_locale(lang).SYSTEM_PROMPT_LOCAL
        tools_index = {}
        tools_schema = NOT_GIVEN

    context = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "[A new caller has connected]"},
        ],
        tools=tools_schema,
    )

    if use_tools:
        conversation.set_llm_context(context)
        conversation.set_tools_builder(
            lambda names: _build_tools_schema_from_names(names, tools_index)
        )

    context_aggregator = LLMContextAggregatorPair(context)

    call_logger = CallLogger(conversation.state.call_id)
    metrics_proc = MetricsProcessor(call_logger)
    transcript_proc = TranscriptProcessor(websocket, call_logger, conversation)
    agent_text_proc = AgentTextProcessor(websocket, call_logger, lang=lang)

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt_service,
            transcript_proc,
            context_aggregator.user(),
            llm_service,
            tts_service,
            agent_text_proc,
            transport.output(),
            context_aggregator.assistant(),
            metrics_proc,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    runner = WorkerRunner()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, websocket):
        await task.queue_frame(LLMContextFrame(context=context))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        call_logger.save()

    return task, runner
