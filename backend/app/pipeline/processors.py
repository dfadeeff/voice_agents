"""Custom Pipecat processors for the voice agent pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.aggregators.sentence import match_endofsentence
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.conversation.locales import get_locale
from app.models.schemas import CallPhase

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"(?<!\w)[+]?[\d][\d\s\-]{3,}[\d](?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
# An alphanumeric reference (insurance/claim number) — has both a letter and a
# digit, 5+ chars. Spelled out char-by-char so TTS doesn't read "F62415723" as a
# giant number. Pure-digit strings are handled by _PHONE_RE instead.
_REF_CODE_RE = re.compile(r"\b(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,}\b")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CJK_RE = re.compile(r"[⺀-鿿豈-﫿︰-﹏\U00020000-\U0002FA1F]+")


_FALSE_BOOKING_RE = re.compile(
    r"\b(termin\s+\w*\s*(ist\s+|wurde\s+)?(gebucht|bestätigt|reserviert|vereinbart)|"
    r"ich\s+habe\s+.*?termin.*?(gebucht|bestätigt|reserviert|vereinbart)|"
    # Future-tense fabrication: "(Frau X) wird … einen Termin … buchen".
    r"wird\s+.*?\btermin\b.*?\b(buchen|gebucht|reservier\w*|vereinbar\w*)|"
    r"termin\s+steht)\b",
    re.IGNORECASE,
)
_JSON_LEAK_RE = re.compile(r"\{\s*\"?(intent|legal_area|reason|summary|area|field_name)\"?\s*:")
_TECHNICAL_LEAK_RE = re.compile(
    r"\b(legal[_\s]?area|classify|route[_\s]?call|capture[_\s]?caller|"
    r"request[_\s]?handoff|confirm[_\s]?caller|matter[_\s]?summary|"
    r"book[_\s]?consultation|check[_\s]?availability|"
    r"intent[_\s]?detection|caller[_\s]?intent)\b",
    re.IGNORECASE,
)
_INTERNAL_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_INTERNAL_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)
_INTERNAL_ONLY_RE = re.compile(r"^\s*(?:\{.*\}|\[.*\])\s*[.!?]?\s*$", re.DOTALL)

# A lowercase snake_case identifier (2+ underscore-joined parts) optionally
# trailed by a call payload like {…} or (…). Natural German/English speech never
# contains snake_case, so any such token is a leaked (or garbled) tool/arg name —
# e.g. "roring_caller_details {...}" (a mangled capture_caller_details). The
# lookbehind/lookahead protect email local-parts like "fade_jeff@gmail.com".
_SNAKE_IDENT_RE = re.compile(
    r"(?<![\w@.])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w@])(?:\s*[({][^)}]*[)}]?)?"
)
# An unfilled template placeholder such as [preferred_date], {name}, <feld>.
# Real JSON args ({"k": "v"}) start with a quote, so they don't match here.
_PLACEHOLDER_RE = re.compile(r"[\[{<]\s*[a-z][a-z0-9 _]*[\]}>]")
# A serialized tool call leaked into spoken text — e.g. '... "name": "", "arguments":'.
# These keys never occur in real speech, so the whole sentence is dropped.
_TOOLCALL_TEXT_RE = re.compile(
    r"\"(?:name|arguments|parameters|function|tool_call_id|tool_calls|role|content)\"\s*:",
    re.IGNORECASE,
)
# A sentence has speakable content only if it contains a real word (2+ letters).
_HAS_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]{2,}")
_FILLER_ONLY_RE = re.compile(
    r"^[äöüaeiouh]+[.!?,\s…]*$" r"|^(?:äh|ähm|eh|ehm|hm+|mh+m?|oh|öh|uh|uhm|ah|aha)[.!?,\s…]*$",
    re.IGNORECASE,
)
_TTS_EMAIL_RE = re.compile(r"\b\w+\s+at\s+\w+(?:\s+(?:Punkt|dot)\s+\w+)+")


def _reverse_email_tts(text: str) -> str:
    """Reverse TTS email expansion for visual display.

    'langfeld at gmail Punkt com' → 'langfeld@gmail.com'
    """

    def _rebuild(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace(" at ", "@", 1)
        s = s.replace(" Punkt ", ".").replace(" dot ", ".")
        return s

    return _TTS_EMAIL_RE.sub(_rebuild, text)


_IMPOSSIBLE_HANDOFF_DE_RE = re.compile(
    r"\b(?:ich\s+)?(?:verbinde\s+sie|stelle\s+sie\s+durch|leite\s+sie\s+weiter)\b",
    re.IGNORECASE,
)
_IMPOSSIBLE_HANDOFF_EN_RE = re.compile(
    r"\b(?:i(?:'ll| will)?\s+)?(?:connect|transfer)\s+you\b",
    re.IGNORECASE,
)


def _guard_false_booking(text: str, booking_confirmed: bool = False) -> str:
    if booking_confirmed:
        return text
    if _FALSE_BOOKING_RE.search(text):
        logger.warning("Blocked false booking language: %r", text)
        return "Gerne. Ich nehme Ihren Terminwunsch auf und leite ihn an das Kanzleiteam weiter."
    return text


def _guard_impossible_handoff(text: str, lang: str = "de") -> str:
    pattern = _IMPOSSIBLE_HANDOFF_DE_RE if lang == "de" else _IMPOSSIBLE_HANDOFF_EN_RE
    if pattern.search(text):
        logger.warning("Blocked impossible live-transfer claim: %r", text)
        if lang == "de":
            return "Ich kann Ihren Rückrufwunsch aufnehmen und an das Kanzleiteam weitergeben."
        return "I can record your callback request and pass it to the team."
    return text


class PreTTSSanitizer(FrameProcessor):
    """Sits between LLM and TTS. Sanitizes complete sentences before synthesis.

    Tool names and JSON often span several streamed LLM text chunks. Buffering
    to the same sentence boundary used by TTS prevents split internal text from
    reaching speech synthesis while adding no latency beyond TTS aggregation.
    """

    def __init__(
        self,
        lang: str = "de",
        tool_names: list[str] | None = None,
        conversation: ConversationManager | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._lang = lang
        self._in_think = False
        self._buffer = ""
        self._conversation = conversation

        if tool_names:
            patterns = []
            for n in tool_names:
                parts = re.split(r"[_\s]+", n)
                patterns.append(r"[\s_\-]*".join(re.escape(p) for p in parts))
            self._tool_re = re.compile(
                r"-?\s*(?:" + "|".join(patterns) + r")\s*(?:\{[^}]*\}?)?",
                re.IGNORECASE,
            )
        else:
            self._tool_re = None

    def _strip_think(self, text: str) -> str:
        result: list[str] = []
        i = 0
        while i < len(text):
            if self._in_think:
                end = text.find("</think>", i)
                if end >= 0:
                    self._in_think = False
                    i = end + 8
                else:
                    return "".join(result)
            else:
                start = text.find("<think>", i)
                if start >= 0:
                    result.append(text[i:start])
                    self._in_think = True
                    i = start + 7
                else:
                    result.append(text[i:])
                    break
        return "".join(result)

    def _sanitize(self, text: str) -> str:
        original = text
        if _INTERNAL_ONLY_RE.match(text):
            logger.warning("Dropped internal-only LLM output before TTS: %r", text)
            return ""
        # An unfilled template placeholder ([preferred_date], {name}) means the
        # LLM emitted a template it never filled. Stripping the token leaves
        # broken grammar ("Am liebsten am  um?"), so drop the whole sentence.
        if _PLACEHOLDER_RE.search(text):
            logger.warning("Dropped sentence with unfilled placeholder before TTS: %r", text)
            return ""
        if _TOOLCALL_TEXT_RE.search(text):
            logger.warning("Dropped leaked tool-call JSON before TTS: %r", text)
            return ""
        guarded = _guard_impossible_handoff(text, self._lang)
        if guarded != text:
            return _tts_preprocess(guarded, self._lang)
        text = _CJK_RE.sub("", text)
        text = _INTERNAL_JSON_OBJECT_RE.sub("", text)
        text = _INTERNAL_JSON_ARRAY_RE.sub("", text)
        if self._tool_re:
            text = self._tool_re.sub("", text)
        text = _SNAKE_IDENT_RE.sub("", text)
        text = _JSON_LEAK_RE.sub("", text)
        text = _TECHNICAL_LEAK_RE.sub("", text)
        # Scars left by stripped tool calls: empty brackets and orphan dashes
        # (e.g. "…8, 9()-Ich werde…" → "…8, 9 Ich werde…").
        text = re.sub(r"[(){}\[\]]+", " ", text)
        text = re.sub(r"(^|\s)[-–—]+\s*", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[\-–—.,\s]+", "", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", text)
        # If stripping internal tokens left no real word, it was all leakage.
        if text and not _HAS_WORD_RE.search(text):
            logger.warning("Dropped non-speakable residue before TTS: %r", original)
            return ""
        text = _guard_false_booking(text, self._booking_confirmed)
        text = _tts_preprocess(text, self._lang)
        if text != original.strip():
            logger.warning("Pre-TTS sanitized: %r → %r", original, text)
        return text

    async def _push_sanitized(self, text: str, direction: FrameDirection) -> None:
        sanitized = self._sanitize(text)
        if sanitized:
            await self.push_frame(TextFrame(text=sanitized), direction)

    async def _flush_sentences(self, direction: FrameDirection) -> None:
        while sentence_end := match_endofsentence(self._buffer):
            sentence = self._buffer[:sentence_end]
            self._buffer = self._buffer[sentence_end:]
            await self._push_sanitized(sentence, direction)

    @property
    def _booking_confirmed(self) -> bool:
        if self._conversation is None:
            return False
        return self._conversation.state.booking_confirmed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, TextFrame)
            and not isinstance(
                frame, TranscriptionFrame | InterimTranscriptionFrame | AggregatedTextFrame
            )
        ):
            self._buffer += self._strip_think(frame.text)
            self._buffer = re.sub(r"([.!?])(?=[A-ZÄÖÜ])", r"\1 ", self._buffer)
            await self._flush_sentences(direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._buffer = ""
            self._in_think = False

        if isinstance(frame, LLMFullResponseEndFrame) and self._buffer:
            remaining = self._buffer
            self._buffer = ""
            await self._push_sanitized(remaining, direction)

        await self.push_frame(frame, direction)


def _tts_preprocess(text: str, lang: str = "de") -> str:
    """Make agent text more TTS-friendly.

    - Expand phone-like digit sequences to space-separated digits so TTS
      reads them individually instead of as large numbers.
    - Expand email addresses to spoken form.
    """

    def _expand_phone(m: re.Match) -> str:
        digits = re.sub(r"[^\d]", "", m.group(0))
        prefix = "plus " if m.group(0).startswith("+") else ""
        return prefix + ", ".join(digits)

    def _expand_email(m: re.Match) -> str:
        email = m.group(0)
        if lang == "de":
            return email.replace("@", " at ").replace(".", " Punkt ")
        return email.replace("@", " at ").replace(".", " dot ")

    text = _THINK_RE.sub("", text).strip()
    text = _CJK_RE.sub("", text).strip()
    if lang == "de":
        text = re.sub(r"\bDr\.\s*", "Doktor ", text)
        text = re.sub(r"\bProf\.\s*", "Professor ", text)
        text = re.sub(r"\b(?:Mrs|Ms)\.?\s+", "Frau ", text)
        text = re.sub(r"\bMr\.?\s+", "Herr ", text)
    text = _EMAIL_RE.sub(_expand_email, text)
    text = _REF_CODE_RE.sub(lambda m: ", ".join(m.group(0)), text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " ", text)
    text = _PHONE_RE.sub(_expand_phone, text)
    text = re.sub(r"(\d)\.$", r"\1", text)
    return text


LOGS_DIR = Path("logs")


class CallLogger:
    """Accumulates call transcript entries and writes them to a JSON file on save()."""

    def __init__(self, call_id: str):
        self.call_id = call_id
        self.start_time = datetime.now(UTC).isoformat()
        self.entries: list[dict] = []
        self.ttfa_samples: list[float] = []
        self.last_user_speech_end: float | None = None
        self.component_ttfb: dict[str, list[float]] = {}

    def record_user_speech_end(self) -> None:
        self.last_user_speech_end = time.monotonic()

    def record_first_agent_chunk(self) -> None:
        if self.last_user_speech_end is not None:
            ttfa = time.monotonic() - self.last_user_speech_end
            self.ttfa_samples.append(ttfa)
            logger.info("TTFA: %.0fms", ttfa * 1000)
            self.last_user_speech_end = None

    def record_component_ttfb(self, component: str, ttfb_ms: float) -> None:
        self.component_ttfb.setdefault(component, []).append(ttfb_ms)
        logger.info("%s TTFB: %.0fms", component, ttfb_ms)

    def log(self, role: str, text: str):
        self.entries.append(
            {
                "role": role,
                "text": text,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def save(self):
        if not self.entries:
            return
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        metrics = {}
        if self.ttfa_samples:
            sorted_samples = sorted(self.ttfa_samples)
            metrics["ttfa_avg_ms"] = round(sum(sorted_samples) / len(sorted_samples) * 1000)
            metrics["ttfa_p50_ms"] = round(sorted_samples[len(sorted_samples) // 2] * 1000)
            metrics["ttfa_samples"] = len(sorted_samples)
            logger.info(
                "Call %s TTFA: avg=%dms p50=%dms (%d samples)",
                self.call_id,
                metrics["ttfa_avg_ms"],
                metrics["ttfa_p50_ms"],
                metrics["ttfa_samples"],
            )

        if self.component_ttfb:
            components = {}
            for name, samples in self.component_ttfb.items():
                sorted_s = sorted(samples)
                comp = {
                    "avg_ms": round(sum(sorted_s) / len(sorted_s)),
                    "p50_ms": round(sorted_s[len(sorted_s) // 2]),
                    "min_ms": round(min(sorted_s)),
                    "max_ms": round(max(sorted_s)),
                    "samples": len(sorted_s),
                }
                components[name] = comp
                logger.info(
                    "Call %s %s: avg=%dms p50=%dms min=%dms max=%dms (%d samples)",
                    self.call_id,
                    name,
                    comp["avg_ms"],
                    comp["p50_ms"],
                    comp["min_ms"],
                    comp["max_ms"],
                    comp["samples"],
                )
            metrics["component_ttfb"] = components

        log_data = {
            "call_id": self.call_id,
            "start_time": self.start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "transcript": self.entries,
            "metrics": metrics,
        }
        path = LOGS_DIR / f"{self.call_id}.json"
        path.write_text(json.dumps(log_data, indent=2))
        logger.info("Call log saved: %s (%d entries)", path, len(self.entries))


class MetricsProcessor(FrameProcessor):
    """Captures Pipecat's per-component TTFB metrics (STT, LLM, TTS)."""

    def __init__(self, call_logger: CallLogger, **kwargs):
        super().__init__(**kwargs)
        self._logger = call_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for datum in frame.data:
                if isinstance(datum, TTFBMetricsData):
                    self._logger.record_component_ttfb(datum.processor, datum.value * 1000)

        await self.push_frame(frame, direction)


class FillerInjector(FrameProcessor):
    """Injects a filler utterance if the LLM takes too long to respond.

    Sits between the LLM and PreTTSSanitizer. TranscriptProcessor starts
    the timer on each user turn; if the LLM emits LLMFullResponseStartFrame
    before the timer fires, the filler is cancelled.
    """

    def __init__(
        self,
        conversation: ConversationManager,
        delay_s: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._conversation = conversation
        self._delay_s = delay_s
        self._filler_index = 0
        self._filler_task: asyncio.Task[None] | None = None

    def _get_filler(self) -> str:
        locale = get_locale(self._conversation.lang)
        fillers = getattr(locale, "FILLERS", ["Mhm."])
        filler = fillers[self._filler_index % len(fillers)]
        self._filler_index += 1
        return filler

    def start_filler_timer(self) -> None:
        self._cancel_filler()
        if self._delay_s > 0:
            self._filler_task = asyncio.get_event_loop().create_task(self._delayed_filler())

    def _cancel_filler(self) -> None:
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            self._filler_task = None

    async def _delayed_filler(self) -> None:
        try:
            await asyncio.sleep(self._delay_s)
            filler = self._get_filler()
            logger.info("FILLER: %s", filler)
            await self.push_frame(TextFrame(text=filler), FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._cancel_filler()

        if isinstance(frame, InterruptionFrame):
            self._cancel_filler()

        await self.push_frame(frame, direction)


class TranscriptProcessor(FrameProcessor):
    """Sits between STT and UserAggregator. Sends user transcriptions to
    the frontend via the websocket (the output transport only serializes audio).
    Also advances the conversation state machine on each user turn."""

    # Fields the caller reads out as a number/email, with mid-utterance pauses —
    # these turns get a longer end-of-speech window so a pause doesn't split them.
    _DICTATION_AWAITING = frozenset({"email", "phone", "insurance", "insurance_confirm"})

    def __init__(
        self,
        websocket,
        call_logger: CallLogger,
        conversation: ConversationManager,
        filler_injector: FillerInjector | None = None,
        vad_analyzer=None,
        vad_params=None,
        dictation_stop_secs: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger
        self._conversation = conversation
        self._filler_injector = filler_injector
        # Per-turn endpoint tuning: widen the VAD stop window while the next reply
        # is a dictated number/email, restore it otherwise.
        self._vad = vad_analyzer
        self._vad_params = vad_params
        self._normal_stop_secs = vad_params.stop_secs if vad_params else None
        self._dictation_stop_secs = dictation_stop_secs
        # None until the first turn tunes it — the VAD is initialised wide (the
        # opening problem-description is free-form), so force the first apply.
        self._current_stop_secs = None
        # A split utterance ("Klein at Hotmail" | "Punkt de") arrives as two rapid
        # STT finals; each would re-fire the same scripted line, speaking it twice.
        # Suppress an identical fast-path repeated within this window.
        self._last_fast_path: str | None = None
        self._last_fast_path_at: float = 0.0

    def _display_transcript(self, text: str) -> str:
        """Tidy the caller transcript shown in the UI.

        When the caller was dictating their phone number, Whisper renders the
        spoken digits as noisy decimals ("01, 5.1, 5.6, 7.3, 8.6, 9.4"). We
        already normalize that to a real number for booking; show that clean
        number in the transcript too instead of the raw STT artifact. ``awaiting``
        still holds the previous question's slot at this point (it is recomputed
        later in next_prompt), so it tells us this turn was the phone dictation.
        """
        from app.conversation.phone import normalize_phone_text

        if self._conversation.state.awaiting == "phone":
            normalized = normalize_phone_text(text)
            if normalized and len(re.sub(r"\D", "", normalized)) >= 7:
                return normalized
        return text

    def _endpoint_secs_for(self, awaiting: str | None) -> float | None:
        """Target VAD stop window for the next reply. Wide while the caller speaks
        freely — dictating a number/email, OR describing their problem on the
        opening/routing/info turns (awaiting is None) where they pause to think.
        Snappy for the short scripted answers (name, yes/no confirms, slot choice)."""
        if self._normal_stop_secs is None:
            return None
        free_form = awaiting in self._DICTATION_AWAITING or awaiting is None
        return self._dictation_stop_secs if free_form else self._normal_stop_secs

    def _tune_endpoint(self) -> None:
        """Adjust the VAD end-of-speech window to match the field the agent just
        asked for (set after next_prompt has updated state.awaiting)."""
        if self._vad is None or self._vad_params is None:
            return
        target = self._endpoint_secs_for(self._conversation.state.awaiting)
        if target is None or target == self._current_stop_secs:
            return
        self._vad.set_params(self._vad_params.model_copy(update={"stop_secs": target}))
        self._current_stop_secs = target
        logger.info(
            "VAD stop window → %.1fs (awaiting=%s)", target, self._conversation.state.awaiting
        )

    def _is_duplicate_fast_path(self, line: str, now: float, window_s: float = 5.0) -> bool:
        """True when ``line`` is identical to the one just emitted within window_s
        (a split utterance re-firing the same scripted prompt). Records the line
        when it's not a duplicate."""
        if line == self._last_fast_path and now - self._last_fast_path_at < window_s:
            return True
        self._last_fast_path = line
        self._last_fast_path_at = now
        return False

    def _check_fast_path(self, old_phase: CallPhase, new_phase: CallPhase) -> str | None:
        # The data-collection spine is fully state-determined, so speak the next
        # question from a template and skip the LLM (it cannot drift/hallucinate
        # here). next_prompt() also records what the question asked for
        # (state.awaiting) so the next reply is parsed deterministically. Returns
        # None for ROUTING / INFORMATION, where the LLM legitimately drives.
        return self._conversation.next_prompt()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()

            if _FILLER_ONLY_RE.match(text):
                logger.info("Filtered filler transcription: %r", text)
                return

            result = frame.result if isinstance(frame.result, dict) else {}
            confidence = result.get("confidence")
            self._conversation.set_transcription_confidence(confidence)
            logger.info("USER: %s (confidence=%s)", text, confidence)
            self._logger.log("user", text)
            self._logger.record_user_speech_end()

            old_phase = self._conversation.state.phase
            self._conversation.add_user_message(text)
            # LLM rescue for the open-ended slots when the deterministic parse
            # missed (no-op unless configured — keyword/regex is the local default).
            await self._conversation.resolve_email_if_pending(text)
            await self._conversation.resolve_matter_if_pending(text)
            new_phase = self._conversation.state.phase
            if new_phase != old_phase:
                logger.info("Phase: %s → %s", old_phase.value, new_phase.value)

            try:
                await self._ws.send_json(
                    {
                        "type": "user_transcript",
                        "text": self._display_transcript(text),
                        "confidence": confidence,
                    }
                )
            except Exception:
                pass

            fast_path = self._check_fast_path(old_phase, new_phase)
            # next_prompt() (inside _check_fast_path) has set the next awaited field;
            # widen/restore the end-of-speech window to match it.
            self._tune_endpoint()
            if fast_path:
                if self._is_duplicate_fast_path(fast_path, time.monotonic()):
                    # Same line we just spoke (split-utterance double) — don't repeat it.
                    logger.info("Suppressed duplicate fast-path (split utterance): %s", fast_path)
                    return
                logger.info("FAST PATH: %s", fast_path)
                self._logger.log("agent", fast_path)
                await self.push_frame(
                    LLMMessagesAppendFrame(
                        [{"role": "user", "content": text}],
                        run_llm=False,
                    ),
                    direction,
                )
                await self.push_frame(LLMFullResponseStartFrame(), direction)
                await self.push_frame(TextFrame(text=fast_path), direction)
                await self.push_frame(LLMFullResponseEndFrame(), direction)
                return

            if self._filler_injector:
                self._filler_injector.start_filler_timer()

        await self.push_frame(frame, direction)


class AgentTextProcessor(FrameProcessor):
    """Sits after TTS. Streams the exact spoken text to the frontend and call log.

    Primary sanitization happens in PreTTSSanitizer (before TTS). This
    processor applies sentence-level safety nets to TTSTextFrame instances,
    which represent text that TTS actually synthesized. AggregatedTextFrame
    may include generated text that was interrupted or never spoken, so it is
    only used as a turn boundary.
    """

    def __init__(
        self,
        websocket,
        call_logger: CallLogger,
        lang: str = "de",
        tool_names: list[str] | None = None,
        conversation: ConversationManager | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws = websocket
        self._logger = call_logger
        self._lang = lang
        self._conversation = conversation
        self._first_chunk_this_turn = True
        self._turn_text_parts: list[str] = []
        if tool_names:
            patterns = []
            for n in tool_names:
                parts = re.split(r"[_\s]+", n)
                patterns.append(r"[\s_\-]*".join(re.escape(p) for p in parts))
            self._tool_re = re.compile(
                r"-?\s*(?:" + "|".join(patterns) + r")\s*(?:\{[^}]*\}?)?",
                re.IGNORECASE,
            )
        else:
            self._tool_re = None

    def _strip_tool_names(self, text: str) -> str:
        cleaned = text
        if self._tool_re is not None:
            cleaned = self._tool_re.sub("", cleaned)
        cleaned = _JSON_LEAK_RE.sub("", cleaned)
        if cleaned != text:
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—.,")
            logger.warning(
                "Stripped hallucinated tool/internal text from agent text: %r → %r",
                text,
                cleaned,
            )
        return cleaned.strip()

    @property
    def _booking_confirmed(self) -> bool:
        if self._conversation is None:
            return False
        return self._conversation.state.booking_confirmed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame) and frame.text:
            if self._first_chunk_this_turn:
                self._logger.record_first_agent_chunk()
                self._first_chunk_this_turn = False
            original = frame.text
            processed = original
            processed = self._strip_tool_names(processed)
            processed = _guard_false_booking(processed, self._booking_confirmed)
            if not processed:
                return
            if processed != original:
                logger.debug("Post-TTS text cleanup: %r → %r", original, processed)
                frame = TTSTextFrame(text=processed, aggregated_by=frame.aggregated_by)
            logger.info("AGENT SPOKEN: %s", processed)
            self._logger.log("agent", processed)
            self._turn_text_parts.append(processed)
            display_text = _reverse_email_tts(processed)
            try:
                await self._ws.send_json({"type": "agent_text", "text": display_text})
            except Exception:
                pass

        if (
            isinstance(frame, AggregatedTextFrame)
            and not isinstance(frame, TTSTextFrame)
            and frame.text
        ):
            if self._turn_text_parts and self._conversation:
                full_text = " ".join(self._turn_text_parts)
                self._conversation.add_assistant_message(full_text)
            self._turn_text_parts = []
            self._first_chunk_this_turn = True

        await self.push_frame(frame, direction)
