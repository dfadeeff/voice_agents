"""Headless demo calls: synthetic caller audio through the real STT → agent → TTS loop.

Each scenario synthesizes the caller's utterances with Piper (a different voice
than the agent), transcribes them with the same faster-whisper model and hotword
biasing the live pipeline uses, feeds the transcript through the real
ConversationManager, and speaks the agent's replies with the agent voice.
The caller is *reactive*: each reply is chosen by what the agent just asked for
(``state.awaiting``), so a mishearing leads to a natural re-ask/correction
instead of a derailed script — including rejecting a wrong email read-back,
exactly like a real caller. Output per scenario:

    demo/<scenario>.wav   — the full two-party call audio
    demo/<scenario>.md    — turn-by-turn transcript (what was said vs. what STT heard)

Because the caller audio passes through real STT, this doubles as an end-to-end
audio regression test: each scenario asserts its expected outcome (booking
confirmed, callback recorded, …) and the script exits non-zero on a miss.

If a local Ollama is reachable, the LLM email rescue is enabled (as in the live
pipeline); without it the demo still runs, regex-only.

Run via `make demo-call` (models must be downloaded first: `make models`).
"""

from __future__ import annotations

import asyncio
import struct
import sys
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.conversation.locales import get_locale  # noqa: E402
from app.conversation.manager import ConversationManager  # noqa: E402
from app.pipeline.local_whisper import _LEGAL_HOTWORDS, _PHONE_HOTWORDS  # noqa: E402
from app.pipeline.processors import _tts_preprocess  # noqa: E402
from app.services.calendar import CalendarService  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = BACKEND_ROOT.parent / "demo"

AGENT_VOICE = "de_DE-eva_k-x_low"
CALLER_VOICE = "de_DE-thorsten-medium"

TURN_GAP_MS = 450  # silence between turns in the stitched recording
MAX_TURNS = 18  # hard stop per call
MAX_REPEATS = 3  # times the caller re-gives the same datum before giving up

# A caller reply: fixed text, or a function of the manager (e.g. check the
# read-back against the intended value and confirm/deny accordingly).
Reply = str | Callable[[ConversationManager], str]


def confirm_or_reject(field_name: str, expected: str) -> Reply:
    """Reply to a read-back the way a real caller would: 'ja' when the agent
    heard the right value, 'nein' when it heard something else."""

    def _reply(manager: ConversationManager) -> str:
        entity = manager.state.entities.get(field_name)
        heard = (entity.value if entity else "").lower().replace(" ", "")
        if heard == expected.lower().replace(" ", ""):
            return "Ja, das ist richtig."
        return "Nein, das ist falsch."

    return _reply


@dataclass
class Scenario:
    name: str
    title: str
    opening: str
    # awaiting-field → queued caller replies; the last entry repeats (bounded)
    # when the agent re-asks. ``None`` queues replies for turns where nothing is
    # awaited (e.g. the closing after the booking confirmation).
    replies: dict[str | None, list[Reply]]
    check: Callable[[ConversationManager], list[str]]
    failures: list[str] = field(default_factory=list)


def _load_voice(name: str):
    from piper import PiperVoice

    model = BACKEND_ROOT / "models" / "piper" / f"{name}.onnx"
    if not model.exists():
        raise SystemExit(f"Piper voice missing: {model} — run `make models` first")
    return PiperVoice.load(str(model), config_path=str(model) + ".json")


def _synth(voice, text: str) -> tuple[bytes, int]:
    """Synthesize text → (int16 PCM bytes, sample_rate)."""
    chunks = [c.audio_int16_bytes for c in voice.synthesize(text)]
    return b"".join(chunks), voice.config.sample_rate


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Naive linear resample, good enough for stitching demo audio."""
    if src_rate == dst_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(len(samples) * dst_rate / src_rate)
    out = np.interp(
        np.linspace(0, len(samples) - 1, n_out), np.arange(len(samples)), samples
    ).astype(np.int16)
    return out.tobytes()


def _transcribe(
    model, pcm: bytes, rate: int, lang: str, awaiting: str | None
) -> tuple[str, float | None]:
    """Run caller audio through faster-whisper with the live pipeline's hotword
    biasing (legal vocabulary; digit words while a phone number is awaited;
    deliberately nothing for email — see local_whisper)."""
    audio = np.frombuffer(_resample(pcm, rate, 16000), dtype=np.int16)
    audio_f = audio.astype(np.float32) / 32768.0
    hotwords = _LEGAL_HOTWORDS
    if awaiting in ("phone", "phone_confirm"):
        hotwords += " " + _PHONE_HOTWORDS
    segments, _ = model.transcribe(
        audio_f,
        language=lang,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        hotwords=hotwords if lang == "de" else None,
    )
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    if not segments:
        return text, None
    confidence = float(np.exp(np.mean([s.avg_logprob for s in segments])))
    return text, confidence


def _make_email_extractor_if_available(settings) -> Callable | None:
    """Enable the LLM email rescue when a local Ollama responds, mirroring the
    live pipeline (llm_assist). Headless fallback: regex only."""
    try:
        import httpx

        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1.5).raise_for_status()
        from app.conversation.email_extract import make_email_extractor

        return make_email_extractor(settings)
    except Exception:
        return None


async def _seed_demo_calendar(db_path: str) -> CalendarService:
    """A small deterministic calendar: employment mornings only (so an afternoon
    request is reliably unavailable), plus a traffic slot."""
    import aiosqlite

    cal = CalendarService(db_path=db_path)
    await cal.init_db()
    d1 = (date.today() + timedelta(days=1)).isoformat()
    d2 = (date.today() + timedelta(days=2)).isoformat()
    rows = [
        (d1, "09:00", 30, "employment", "Sarah Mitchell"),
        (d1, "10:30", 30, "employment", "Sarah Mitchell"),
        (d2, "09:30", 30, "employment", "James Cooper"),
        (d1, "11:00", 30, "traffic", "Lisa Hoffmann"),
    ]
    async with aiosqlite.connect(db_path) as db:
        await db.executemany(
            "INSERT INTO slots (date, time, duration_minutes, legal_area, lawyer_name)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()
    return cal


def _booking_checks(m: ConversationManager) -> list[str]:
    return (
        []
        if m.state.booking_confirmed
        else [f"booking not confirmed (phase={m.state.phase.value})"]
    )


SCENARIOS = [
    Scenario(
        name="01_booking_happy_path",
        title="Happy path: employment dismissal → details → booked slot",
        opening="Guten Tag, ich habe gestern die Kündigung von meinem Arbeitgeber erhalten.",
        replies={
            "matter_type": ["Ja genau, es geht um die Kündigung."],
            "matter_details": ["Die Frist läuft in zwei Wochen ab, ich möchte dagegen vorgehen."],
            "name": ["Mein Name ist Anna Schmidt."],
            "name_confirm": [confirm_or_reject("name", "Anna Schmidt")],
            "email": ["anna punkt schmidt at gmail punkt com"],
            "email_confirm": [confirm_or_reject("email", "anna.schmidt@gmail.com")],
            "email_spell": ["A, N, N, A, Punkt, S, C, H, M, I, D, T."],
            "phone": ["null eins sieben null, zwei drei vier fünf, sechs sieben acht neun"],
            "phone_confirm": ["Ja, genau."],
            "slot": ["Die erste bitte."],
            None: ["Vielen Dank, auf Wiederhören!"],
        },
        check=lambda m: (
            _booking_checks(m)
            + ([] if "name" in m.state.entities else ["name not captured"])
            + ([] if "phone" in m.state.entities else ["phone not captured"])
        ),
    ),
    Scenario(
        name="02_unavailable_slot",
        title="Requested time not free → apology + alternatives → booked",
        opening="Hallo, mir wurde gekündigt und ich brauche einen Beratungstermin.",
        replies={
            "matter_type": ["Ja, es geht um die Kündigung."],
            "matter_details": ["Es geht um die Frist für die Kündigungsschutzklage."],
            "name": ["Ich heiße Peter Lang."],
            "name_confirm": [confirm_or_reject("name", "Peter Lang")],
            "email": ["Ich habe keine E-Mail-Adresse."],
            "phone": ["null eins fünf eins, zwei drei vier fünf, sechs sieben acht"],
            "phone_confirm": ["Ja, stimmt."],
            "slot": ["Geht es auch um 14 Uhr?", "Gut, dann nehme ich die erste."],
            None: ["Danke, auf Wiederhören!"],
        },
        check=lambda m: (
            _booking_checks(m) + ([] if m.state.email_skipped else ["email skip not recorded"])
        ),
    ),
    Scenario(
        name="03_email_uncertainty_spelling",
        title="Uncertainty path: email read-back rejected twice → spelling mode",
        opening="Guten Tag, ich habe eine Abmahnung von meinem Arbeitgeber bekommen.",
        replies={
            "matter_type": ["Ja, um die Abmahnung."],
            "matter_details": ["Ich halte die Abmahnung für unberechtigt."],
            "name": ["Mein Name ist Reiner Ritter."],
            "name_confirm": [confirm_or_reject("name", "Reiner Ritter")],
            "email": ["ritter at gmail punkt com"],
            # Simulated mishears: reject the read-back twice → spelling mode.
            "email_confirm": [
                "Nein, das ist falsch.",
                "Nein, das ist immer noch falsch.",
                "Ja, jetzt stimmt es.",
            ],
            "email_spell": ["R, I, T, T, E, R."],
            "phone": ["null eins sieben sechs, neun acht sieben sechs, fünf vier drei zwei"],
            "phone_confirm": ["Ja, richtig."],
            "slot": ["Die erste bitte."],
            None: ["Vielen Dank, auf Wiederhören!"],
        },
        check=lambda m: (
            (
                []
                if (em := m.state.entities.get("email")) and em.confirmed
                else ["email not confirmed via the spelling path"]
            )
            + ([] if m.state.email_spelling else ["spelling mode never engaged"])
        ),
    ),
    Scenario(
        name="04_callback_handoff",
        title="Human handoff: callback request for a named lawyer",
        opening=(
            "Guten Tag, können Sie mich bitte zurückrufen? Ich möchte mit Frau Hoffmann sprechen."
        ),
        replies={
            "name": ["Mein Name ist Sabine Becker."],
            "name_confirm": [confirm_or_reject("name", "Sabine Becker")],
            "phone": ["null eins sechs zwei, drei vier fünf sechs, sieben acht"],
            "phone_confirm": ["Ja, das stimmt."],
            "callback_time": ["Morgen Vormittag bitte."],
            None: ["Danke, auf Wiederhören!"],
        },
        check=lambda m: (
            ([] if m.state.callback_requested else ["callback not recorded"])
            + ([] if m.state.preferred_time else ["preferred callback time not captured"])
            + ([] if m.state.target_person else ["target person not captured"])
        ),
    ),
]


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def _silence(ms: int, rate: int) -> bytes:
    return struct.pack("<h", 0) * int(rate * ms / 1000)


class CallerScript:
    """Pops the next reply for the awaited field; the last one repeats (bounded)
    when the agent re-asks after a mishearing."""

    def __init__(self, replies: dict[str | None, list[Reply]]):
        self._queues = {k: list(v) for k, v in replies.items()}
        self._repeats: dict[str | None, int] = {}

    def next_reply(self, awaiting: str | None, manager: ConversationManager) -> str | None:
        queue = self._queues.get(awaiting)
        if not queue:
            return None
        if len(queue) > 1 or awaiting is None:
            # Consume; an exhausted None-queue ends the call (no goodbye loop).
            reply = queue.pop(0)
        else:
            self._repeats[awaiting] = self._repeats.get(awaiting, 0) + 1
            if self._repeats[awaiting] > MAX_REPEATS:
                return None
            reply = queue[0]
        return reply(manager) if callable(reply) else reply


async def run_scenario(scenario, settings, whisper, agent_voice, caller_voice, extractor) -> bool:
    lang = settings.language
    locale = get_locale(lang)
    greeting = locale.FAST_PATH_RESPONSES["greeting"]

    with tempfile.TemporaryDirectory() as tmp:
        calendar = await _seed_demo_calendar(f"{tmp}/demo.db")
        manager = ConversationManager(call_id=f"demo-{scenario.name}", lang=lang, calendar=calendar)
        if extractor is not None:
            manager.set_email_extractor(extractor)

        out_rate = agent_voice.config.sample_rate
        audio: list[bytes] = []
        transcript: list[str] = [f"# {scenario.title}\n"]
        script = CallerScript(scenario.replies)

        def speak_agent(line: str) -> None:
            pcm, rate = _synth(agent_voice, _tts_preprocess(line, lang))
            audio.append(_resample(pcm, rate, out_rate))
            audio.append(_silence(TURN_GAP_MS, out_rate))
            transcript.append(f"**Agent:** {line}\n")

        speak_agent(greeting)
        manager.add_assistant_message(greeting)
        caller_line: str | None = scenario.opening

        for _ in range(MAX_TURNS):
            if caller_line is None:
                break
            pcm, rate = _synth(caller_voice, caller_line)
            audio.append(_resample(pcm, rate, out_rate))
            audio.append(_silence(TURN_GAP_MS, out_rate))
            heard, confidence = _transcribe(whisper, pcm, rate, lang, manager.awaiting_field())
            conf_s = f"{confidence:.2f}" if confidence is not None else "–"
            note = "" if heard == caller_line else f" *(said: “{caller_line}”)*"
            transcript.append(f"**Caller** (STT {conf_s}): {heard}{note}\n")

            manager.set_transcription_confidence(confidence)
            manager.add_user_message(heard or caller_line)
            await manager.resolve_email_if_pending(heard or caller_line)
            line = manager.next_prompt()
            if line:
                speak_agent(line)
                manager.add_assistant_message(line)
            else:
                transcript.append(
                    "**Agent:** *(LLM-driven turn — nothing scripted pending; "
                    "omitted in the headless demo)*\n"
                )
            caller_line = script.next_reply(manager.awaiting_field(), manager)

        scenario.failures = scenario.check(manager)
        status = "PASS" if not scenario.failures else "FAIL: " + "; ".join(scenario.failures)
        transcript.append(f"\n**Outcome check:** {status}\n")

        DEMO_DIR.mkdir(exist_ok=True)
        _write_wav(DEMO_DIR / f"{scenario.name}.wav", b"".join(audio), out_rate)
        (DEMO_DIR / f"{scenario.name}.md").write_text("\n".join(transcript), encoding="utf-8")

    print(f"  {scenario.name}: {status}")
    return not scenario.failures


async def main() -> int:
    settings = Settings()
    print("Loading models (Piper ×2, faster-whisper)…")
    agent_voice = _load_voice(AGENT_VOICE)
    caller_voice = _load_voice(CALLER_VOICE)
    from faster_whisper import WhisperModel

    whisper = WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    extractor = _make_email_extractor_if_available(settings)
    print(f"LLM email rescue: {'on (Ollama reachable)' if extractor else 'off (regex only)'}")

    print(f"Running {len(SCENARIOS)} demo calls → {DEMO_DIR}/")
    ok = True
    for scenario in SCENARIOS:
        ok = (
            await run_scenario(scenario, settings, whisper, agent_voice, caller_voice, extractor)
            and ok
        )
    print("All scenarios passed." if ok else "Some scenarios FAILED — see demo/*.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
