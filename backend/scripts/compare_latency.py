#!/usr/bin/env python3
"""Compare per-component call latency between two provider configurations.

Each call the pipeline runs saves a JSON log to ``backend/logs/<call_id>.json``
with per-component TTFB and end-to-end TTFA (time from the caller finishing
speaking to the first agent audio). This harness reads two runs — e.g. one made
with the local ``.env`` and one with the cloud ``.env`` — and prints a
side-by-side latency diff.

Workflow:
    1. Run a scripted call on the local stack, note the printed call_id.
    2. Switch providers in .env (e.g. TTS_PROVIDER=cartesia), run the *same*
       script, note the new call_id.
    3. python scripts/compare_latency.py logs/<local>.json logs/<cloud>.json \
           --labels local cloud

Each argument may be a single log file, a directory (all *.json aggregated), or
a glob. With no arguments, the two most recent logs are compared.

This measures the real pipeline, so a couple of caveats show up honestly in the
output:
  * LLM TTFB is ~0ms on turns the deterministic scripted spine fast-paths (the
    LLM is skipped), so the LLM row reflects only the turns it actually ran.
  * Whisper reports a 0ms first sample (model warmup), which the min column shows.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Pipecat tags each service instance with a "#<n>" suffix; strip it for display.
ROLE_BY_MARKER = (("STT", "STT"), ("LLM", "LLM"), ("TTS", "TTS"))


def classify(component_name: str) -> str | None:
    """Map a Pipecat service class name to a pipeline role (STT/LLM/TTS)."""
    for marker, role in ROLE_BY_MARKER:
        if marker in component_name:
            return role
    return None


_VENDOR_SHORT = {
    "LocalWhisperSTTService": "Whisper",
    "DeepgramSTTService": "Deepgram",
    "OLLamaLLMService": "Ollama",
    "OpenAILLMService": "OpenAI",
    "LocalPiperTTSService": "Piper",
    "CartesiaTTSService": "Cartesia",
    "ElevenLabsTTSService": "ElevenLabs",
}


def vendor(component_name: str) -> str:
    """Short, human label for a component (class name minus the '#<n>' suffix)."""
    cls = component_name.split("#", 1)[0]
    return _VENDOR_SHORT.get(cls, cls)


@dataclass
class RoleStats:
    vendors: set[str] = field(default_factory=set)
    weighted_sum_ms: float = 0.0  # sum(avg_ms * samples) across logs
    samples: int = 0
    min_ms: float | None = None
    max_ms: float | None = None

    def add(self, avg_ms: float, samples: int, min_ms: float, max_ms: float, name: str) -> None:
        if samples <= 0:
            return
        self.vendors.add(name)
        self.weighted_sum_ms += avg_ms * samples
        self.samples += samples
        self.min_ms = min_ms if self.min_ms is None else min(self.min_ms, min_ms)
        self.max_ms = max_ms if self.max_ms is None else max(self.max_ms, max_ms)

    @property
    def avg_ms(self) -> float | None:
        return self.weighted_sum_ms / self.samples if self.samples else None


@dataclass
class RunSummary:
    calls: int = 0
    ttfa_weighted_sum_ms: float = 0.0
    ttfa_samples: int = 0
    roles: dict[str, RoleStats] = field(default_factory=dict)

    @property
    def ttfa_avg_ms(self) -> float | None:
        return self.ttfa_weighted_sum_ms / self.ttfa_samples if self.ttfa_samples else None


def summarize(logs: list[dict]) -> RunSummary:
    """Aggregate one or more call logs into sample-weighted per-role latencies."""
    summary = RunSummary()
    for log in logs:
        metrics = log.get("metrics") or {}
        summary.calls += 1
        ttfa_samples = metrics.get("ttfa_samples", 0)
        if ttfa_samples:
            summary.ttfa_weighted_sum_ms += metrics.get("ttfa_avg_ms", 0) * ttfa_samples
            summary.ttfa_samples += ttfa_samples
        for name, comp in (metrics.get("component_ttfb") or {}).items():
            role = classify(name)
            if role is None or comp.get("samples", 0) <= 0:
                continue
            summary.roles.setdefault(role, RoleStats()).add(
                avg_ms=comp.get("avg_ms", 0),
                samples=comp.get("samples", 0),
                min_ms=comp.get("min_ms", 0),
                max_ms=comp.get("max_ms", 0),
                name=vendor(name),
            )
    return summary


def load_logs(spec: str) -> list[dict]:
    """Resolve a path spec (file | directory | glob) to a list of parsed log dicts."""
    path = Path(spec)
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    elif path.is_file():
        files = [path]
    else:
        files = [Path(p) for p in sorted(glob.glob(spec))]
    logs = []
    for f in files:
        try:
            logs.append(json.loads(Path(f).read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! skipping {f}: {e}", file=sys.stderr)
    return logs


def _fmt(ms: float | None) -> str:
    return f"{round(ms)} ms" if ms is not None else "—"


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "—"
    diff = b - a
    pct = diff / a * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{round(diff)} ms ({sign}{pct:.0f}%)"


def _role_label(role: str, ra: RoleStats | None, rb: RoleStats | None) -> str:
    va = "/".join(sorted(ra.vendors)) if ra and ra.vendors else ""
    vb = "/".join(sorted(rb.vendors)) if rb and rb.vendors else ""
    if va and vb:
        return f"{role} ({va})" if va == vb else f"{role} ({va}→{vb})"
    return f"{role} ({va or vb})" if (va or vb) else role


def render(a: RunSummary, b: RunSummary, label_a: str, label_b: str) -> str:
    lw, w = 28, 14  # label column, value columns
    delta_hdr = f"Δ ({label_b}−{label_a})"
    lines = [
        f"Latency comparison  ({label_a}: {a.calls} call(s) vs {label_b}: {b.calls} call(s))",
        "",
        f"{'metric':<{lw}}{label_a:<{w}}{label_b:<{w}}{delta_hdr}",
        "-" * (lw + w * 2 + len(delta_hdr)),
        f"{'TTFA (perceived)':<{lw}}{_fmt(a.ttfa_avg_ms):<{w}}{_fmt(b.ttfa_avg_ms):<{w}}"
        f"{_delta(a.ttfa_avg_ms, b.ttfa_avg_ms)}",
    ]
    notes = []
    for role in ("STT", "LLM", "TTS"):
        ra, rb = a.roles.get(role), b.roles.get(role)
        avg_a = ra.avg_ms if ra else None
        avg_b = rb.avg_ms if rb else None
        lines.append(
            f"{_role_label(role, ra, rb):<{lw}}{_fmt(avg_a):<{w}}{_fmt(avg_b):<{w}}"
            f"{_delta(avg_a, avg_b)}"
        )
        if role == "LLM" and ((ra and (ra.avg_ms or 0) < 1) or (rb and (rb.avg_ms or 0) < 1)):
            notes.append(
                "* LLM ~0 ms means turns were fast-pathed by the scripted spine "
                "(LLM skipped); only routing / info turns invoke it."
            )
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)


RESULTS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "latency-results.md"
_RESULTS_HEADER = (
    "# Latency results (per provider configuration)\n\n"
    'Recorded with `make compare-latency ARGS="--record <label> <log>"`. TTFA is '
    "perceived latency (caller stops speaking → first agent audio); the rest are "
    "per-component TTFB. LLM ~0 ms = turns fast-pathed by the deterministic spine.\n\n"
    "| config | calls | TTFA | STT | LLM | TTS | recorded |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _cell(summary: RunSummary, role: str) -> str:
    rs = summary.roles.get(role)
    if not rs or rs.avg_ms is None:
        return "—"
    return f"{'/'.join(sorted(rs.vendors))} {round(rs.avg_ms)} ms"


def record_row(summary: RunSummary, label: str, timestamp: str) -> str:
    """One markdown table row summarizing a single run's latencies."""
    ttfa = _fmt(summary.ttfa_avg_ms)
    return (
        f"| {label} | {summary.calls} | {ttfa} | {_cell(summary, 'STT')} | "
        f"{_cell(summary, 'LLM')} | {_cell(summary, 'TTS')} | {timestamp} |"
    )


def append_result(row: str) -> None:
    """Append a result row to docs/latency-results.md, creating it with a header."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(_RESULTS_HEADER)
    with RESULTS_FILE.open("a") as f:
        f.write(row + "\n")


def _most_recent() -> str:
    files = sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit(f"No logs found in {LOGS_DIR}.")
    return str(files[0])


def _two_most_recent() -> list[str]:
    files = sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(files) < 2:
        sys.exit(f"Need at least two logs in {LOGS_DIR} to compare (found {len(files)}).")
    return [str(files[1]), str(files[0])]  # older = before, newer = after


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", nargs="?", help="log file | dir | glob for run A")
    parser.add_argument("after", nargs="?", help="log file | dir | glob for run B")
    parser.add_argument("--labels", nargs=2, metavar=("A", "B"), default=["before", "after"])
    parser.add_argument(
        "--record",
        metavar="LABEL",
        help="record one run's latencies under LABEL to docs/latency-results.md "
        "(uses the 'before' log, or the most recent if omitted)",
    )
    args = parser.parse_args(argv)

    # Record mode: summarize a single run and append a labeled row.
    if args.record:
        if args.after:
            parser.error("--record takes a single log (the 'before' arg), not two")
        spec = args.before or _most_recent()
        summary = summarize(load_logs(spec))
        if summary.calls == 0:
            sys.exit(f"No usable log found at {spec!r}.")
        from datetime import datetime

        row = record_row(summary, args.record, datetime.now().strftime("%Y-%m-%d %H:%M"))
        append_result(row)
        print(f"Recorded '{args.record}' to {RESULTS_FILE}:\n{row}")
        return

    if args.before and args.after:
        specs = [args.before, args.after]
    elif args.before or args.after:
        parser.error("provide both 'before' and 'after', or neither (uses two most recent)")
    else:
        specs = _two_most_recent()
        print(f"Comparing two most recent logs in {LOGS_DIR}:\n  A: {specs[0]}\n  B: {specs[1]}\n")

    run_a, run_b = (summarize(load_logs(s)) for s in specs)
    if run_a.calls == 0 or run_b.calls == 0:
        sys.exit("No usable logs found for one or both runs.")
    print(render(run_a, run_b, *args.labels))


if __name__ == "__main__":
    main()
