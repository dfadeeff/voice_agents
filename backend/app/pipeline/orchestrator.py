"""
Pipecat pipeline factory.

Pipecat handles: VAD, turn-taking, interruptions, streaming TTS, transport.
We handle: state machine, system prompt per phase, tool registration + phase guards.
"""

import asyncio
import json
import logging

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext, NotGiven
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.workers.runner import WorkerRunner

from app.config import Settings
from app.conversation.email_extract import make_email_extractor
from app.conversation.locales import get_locale
from app.conversation.manager import ConversationManager
from app.conversation.matter_classify import make_matter_classifier
from app.conversation.prompts import build_system_prompt
from app.pipeline.processors import (
    AgentTextProcessor,
    CallLogger,
    FillerInjector,
    MetricsProcessor,
    PreTTSSanitizer,
    TranscriptProcessor,
)
from app.providers import ConversationAware
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
) -> ToolsSchema | NotGiven:
    """Build a ToolsSchema containing only the named tools. Returns NOT_GIVEN if empty."""
    schemas = [index[n] for n in names if n in index]
    if not schemas:
        return NOT_GIVEN
    return ToolsSchema(standard_tools=schemas)


def register_tools_on_llm(llm_service, registry: ToolRegistry, conversation: ConversationManager):
    """Register tool handlers with phase guards.

    Tool calls are cancelled when the caller interrupts. Pipecat treats
    cancel_on_interruption=False as an asynchronous tool: the LLM continues
    immediately and injects the result later, which can produce stale speech
    after the caller has already moved on.
    """
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

        llm_service.register_function(
            tool_name,
            _make_handler(tool_name),
            cancel_on_interruption=True,
        )


async def create_pipeline(
    stt_service,
    llm_service,
    tts_service,
    transport,
    websocket,
    conversation: ConversationManager,
    tools: ToolRegistry,
    use_tools: bool = True,
) -> tuple[PipelineWorker, WorkerRunner]:
    """Create a Pipecat pipeline wired with our tools."""

    lang = conversation.lang
    # Conversation-aware services (e.g. local STT biasing its decoder toward the
    # field the agent just asked for) get the live conversation; vendors that
    # don't implement the contract are simply skipped.
    if isinstance(stt_service, ConversationAware):
        stt_service.set_conversation(conversation)
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
        messages=[{"role": "system", "content": system_prompt}],
        tools=tools_schema,
    )

    if use_tools:
        conversation.set_llm_context(context)
        conversation.set_tools_builder(
            lambda names: _build_tools_schema_from_names(names, tools_index)
        )

    settings = Settings()
    # Optional LLM rescue for the open-ended slots (cloud-first; keyword/regex stays
    # the local default).
    conversation.set_email_extractor(make_email_extractor(settings))
    conversation.set_matter_classifier(make_matter_classifier(settings))
    vad_params = VADParams(
        confidence=settings.vad_threshold,
        stop_secs=settings.silence_timeout_ms / 1000.0,
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
        ),
    )

    tool_names = tools.list_tools() if use_tools else []

    call_logger = CallLogger(conversation.state.call_id)
    metrics_proc = MetricsProcessor(call_logger)
    filler_injector = FillerInjector(
        conversation=conversation,
        delay_s=settings.filler_delay_ms / 1000,
    )
    transcript_proc = TranscriptProcessor(
        websocket, call_logger, conversation, filler_injector=filler_injector
    )
    pre_tts_sanitizer = PreTTSSanitizer(lang=lang, tool_names=tool_names, conversation=conversation)
    agent_text_proc = AgentTextProcessor(
        websocket, call_logger, lang=lang, tool_names=tool_names, conversation=conversation
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt_service,
            transcript_proc,
            context_aggregator.user(),
            llm_service,
            filler_injector,
            pre_tts_sanitizer,
            tts_service,
            agent_text_proc,
            transport.output(),
            context_aggregator.assistant(),
            metrics_proc,
        ]
    )

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    runner = WorkerRunner()

    locale = get_locale(lang)
    fast_paths = getattr(locale, "FAST_PATH_RESPONSES", {})
    greeting_template = fast_paths.get("greeting")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, websocket):
        await asyncio.sleep(0.2)
        if greeting_template:
            logger.info("FAST PATH greeting (skipping LLM)")
            call_logger.log("agent", greeting_template)
            await task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        [{"role": "user", "content": "[A new caller has connected]"}],
                        run_llm=False,
                    ),
                    LLMFullResponseStartFrame(),
                    TextFrame(text=greeting_template),
                    LLMFullResponseEndFrame(),
                ]
            )
        else:
            logger.info("Client connected — triggering greeting via LLM")
            await task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        [{"role": "user", "content": "[A new caller has connected]"}],
                        run_llm=True,
                    )
                ]
            )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        call_logger.save()
        await task.queue_frame(EndFrame())

    return task, runner
