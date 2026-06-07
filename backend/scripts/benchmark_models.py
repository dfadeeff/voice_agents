"""
Benchmark local LLMs for voice agent suitability.

Tests each model on three dimensions that matter for phone calls:
  1. Latency — time to first token + total response time
  2. Tool calling — does the model use the API correctly or dump JSON as text?
  3. Conversation — does the model greet naturally?

Outputs:
  - Console table (human-readable)
  - benchmarks/benchmark_results.json (machine-readable, for CI/CD)
  - benchmarks/benchmark_results.md (markdown table, for README/PR)

Usage:
    python3 scripts/benchmark_models.py                    # test all installed models
    python3 scripts/benchmark_models.py qwen3:4b llama3.1  # test specific models
"""

import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
BENCHMARKS_DIR = Path("benchmarks")

SYSTEM_PROMPT = (
    "Du bist die Empfangskraft einer Anwaltskanzlei. "
    "Die Kanzlei bearbeitet Arbeitsrecht, Mietrecht und Verkehrsrecht. "
    "Antworte auf Deutsch in höchstens zwei kurzen Sätzen. "
    "Funktionsaufrufe und JSON dürfen niemals als gesprochener Text erscheinen."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "route_call",
            "description": "Bestimme Anliegen und Rechtsgebiet des Anrufers in einem Schritt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["general_info", "book_consultation"],
                    },
                    "legal_area": {
                        "type": "string",
                        "enum": ["employment", "tenancy", "traffic", "unknown"],
                    },
                    "matter_summary": {"type": "string"},
                },
                "required": ["intent", "legal_area"],
            },
        },
    },
]

SCENARIOS = [
    {
        "name": "greeting_no_tools",
        "display": "Greeting (no tools)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[Ein neuer Anrufer hat sich verbunden]"},
        ],
        "tools": None,
        "expect": "speech",
    },
    {
        "name": "greeting_with_tools",
        "display": "Greeting (with tools)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[Ein neuer Anrufer hat sich verbunden]"},
        ],
        "tools": TOOLS,
        "expect": "speech",
    },
    {
        "name": "tool_call_employment",
        "display": "Tool call (employment)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[Ein neuer Anrufer hat sich verbunden]"},
            {
                "role": "assistant",
                "content": "Guten Tag. Wobei können wir Ihnen behilflich sein?",
            },
            {
                "role": "user",
                "content": (
                    "Mein Arbeitgeber hat mir letzte Woche gekündigt. Ich möchte einen Termin."
                ),
            },
        ],
        "tools": TOOLS,
        "expect": "tool_call",
    },
]


def call_ollama(model, messages, tools=None):
    body = {"model": model, "messages": messages, "stream": False}
    if tools:
        body["tools"] = tools

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.perf_counter() - start
            result = json.loads(resp.read())
            return result, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"error": str(e)}, elapsed


def analyze_response(result, expected):
    if "error" in result:
        return "ERROR", result["error"][:60]

    choice = result["choices"][0]["message"]
    content = choice.get("content", "") or ""
    tool_calls = choice.get("tool_calls", [])
    has_speech = bool(content.strip()) and not content.strip().startswith("{")
    has_json_text = content.strip().startswith("{") or '"name"' in content
    has_tool_call = bool(tool_calls)

    if expected == "speech":
        if has_speech and not has_json_text:
            return "PASS", content.strip()[:80]
        if has_json_text:
            return "FAIL", f"JSON as text: {content.strip()[:60]}"
        if has_tool_call and not has_speech:
            tool_name = tool_calls[0]["function"]["name"]
            return "WARN", f"Tool call instead of speech: {tool_name}"
        return "FAIL", "Empty response"

    if expected == "tool_call":
        if has_tool_call:
            tc = tool_calls[0]["function"]
            args = tc.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            speech_note = " + speech" if has_speech else ""
            return "PASS", f"{tc['name']}({json.dumps(args)}){speech_note}"
        if has_json_text:
            return "FAIL", f"JSON as text (not API): {content.strip()[:60]}"
        if has_speech:
            return "WARN", f"Speech only, no tool call: {content.strip()[:60]}"
        return "FAIL", "No tool call and no speech"

    return "???", content[:60]


def warmup_model(model):
    call_ollama(model, [{"role": "user", "content": "hi"}])


def get_installed_models():
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return [m for m in models if "embed" not in m]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        sys.exit(1)


def get_system_info():
    try:
        ollama_version = subprocess.check_output(
            ["ollama", "--version"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception:
        ollama_version = "unknown"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ollama": ollama_version,
        "cpu": platform.processor() or "unknown",
    }


def get_model_info(model):
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/show")
        data = json.dumps({"model": model}).encode()
        req.data = data
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read())
            details = info.get("details", {})
            return {
                "family": details.get("family", "unknown"),
                "parameter_size": details.get("parameter_size", "unknown"),
                "quantization": details.get("quantization_level", "unknown"),
            }
    except Exception:
        return {"family": "unknown", "parameter_size": "unknown", "quantization": "unknown"}


def generate_markdown(report):
    lines = [
        "# Model Benchmark Results",
        "",
        f"**Date:** {report['timestamp']}  ",
        f"**Platform:** {report['system']['platform']}  ",
        f"**Ollama:** {report['system']['ollama']}",
        "",
        "## Summary",
        "",
        "| Model | Size | Greeting | Greeting + Tools | Tool Call | Avg Latency | Verdict |",
        "|-------|------|----------|-----------------|-----------|-------------|---------|",
    ]

    for model_name, model_data in report["models"].items():
        info = model_data["model_info"]
        size = info.get("parameter_size", "?")
        quant = info.get("quantization", "?")
        size_str = f"{size} ({quant})"

        scenarios = {s["scenario"]: s for s in model_data["scenarios"]}
        s1 = scenarios.get("greeting_no_tools", {}).get("status", "?")
        s2 = scenarios.get("greeting_with_tools", {}).get("status", "?")
        s3 = scenarios.get("tool_call_employment", {}).get("status", "?")

        latencies = [s["latency_s"] for s in model_data["scenarios"]]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        statuses = [s["status"] for s in model_data["scenarios"]]
        errors = statuses.count("ERROR")
        fails = statuses.count("FAIL")
        warns = statuses.count("WARN")
        if errors > 0:
            verdict = "Error (see details)"
        elif fails == 0 and warns == 0:
            verdict = "Recommended"
        elif fails == 0:
            verdict = "Usable (caveats)"
        else:
            verdict = "Not recommended"

        lines.append(
            f"| {model_name} | {size_str} | {s1} | {s2} | {s3} | {avg_lat:.2f}s | {verdict} |"
        )

    lines += [
        "",
        "## Detailed Results",
        "",
    ]

    for model_name, model_data in report["models"].items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Scenario | Status | Latency | Tokens | Detail |")
        lines.append("|----------|--------|---------|--------|--------|")

        for s in model_data["scenarios"]:
            tokens = f"{s['prompt_tokens']}→{s['completion_tokens']}"
            detail = s["detail"][:60]
            lines.append(
                f"| {s['scenario']} | {s['status']} | {s['latency_s']:.2f}s | {tokens} | {detail} |"
            )
        lines.append("")

    lines += [
        "## Criteria",
        "",
        (
            "- **Greeting (no tools):** Can the model have a basic"
            " conversation? Expect natural speech."
        ),
        (
            "- **Greeting (with tools):** Does the model still greet"
            " when tool schemas are present, or does it skip to calling"
            " a tool? For a phone call, the agent must speak first."
        ),
        (
            "- **Tool call:** After conversation context, does the model"
            " use the tool calling API correctly (structured `tool_calls`),"
            " or dump JSON as text?"
        ),
        "",
        "### Verdicts",
        "",
        ("- **Recommended**: PASS on all scenarios — speaks naturally and uses tools correctly."),
        (
            "- **Usable (caveats)**: No failures but some warnings"
            " (e.g., skips greeting when tools present)."
        ),
        ("- **Not recommended**: At least one FAIL — broken tool calling or JSON dumping."),
        "",
        "---",
        "*Generated by `scripts/benchmark_models.py`*",
    ]

    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = get_installed_models()

    if not models:
        print("No models found. Run: ollama pull qwen3:4b")
        sys.exit(1)

    print(f"Benchmarking {len(models)} model(s): {', '.join(models)}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("=" * 100)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "system": get_system_info(),
        "models": {},
    }

    for model in models:
        print(f"\n{'─' * 100}")
        print(f"MODEL: {model}")
        print(f"{'─' * 100}")

        print("  Warming up (loading into memory)...", end=" ", flush=True)
        warmup_model(model)
        print("done")

        model_info = get_model_info(model)
        model_scenarios = []

        for scenario in SCENARIOS:
            print(f"\n  [{scenario['display']}]")
            result, elapsed = call_ollama(model, scenario["messages"], scenario["tools"])
            status, detail = analyze_response(result, scenario["expect"])

            tokens = result.get("usage", {})
            if isinstance(tokens, dict):
                prompt_tok = tokens.get("prompt_tokens", 0)
                comp_tok = tokens.get("completion_tokens", 0)
            else:
                prompt_tok = 0
                comp_tok = 0

            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "ERROR": "⚠"}.get(status, "?")

            print(f"    {icon} {status}  ({elapsed:.2f}s, {prompt_tok}→{comp_tok} tokens)")
            print(f"    → {detail}")

            model_scenarios.append(
                {
                    "scenario": scenario["name"],
                    "status": status,
                    "latency_s": round(elapsed, 3),
                    "detail": detail,
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": comp_tok,
                }
            )

        report["models"][model] = {
            "model_info": model_info,
            "scenarios": model_scenarios,
        }

    # Console summary
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    header = (
        f"\n{'Model':<25} {'Greeting':<10} {'Greet+Tools':<13}"
        f" {'Tool Call':<10} {'Avg Latency':<12} {'Verdict'}"
    )
    print(header)
    print(f"{'─' * 25} {'─' * 10} {'─' * 13} {'─' * 10} {'─' * 12} {'─' * 20}")

    for model_name, model_data in report["models"].items():
        statuses = [s["status"] for s in model_data["scenarios"]]
        latencies = [s["latency_s"] for s in model_data["scenarios"]]
        avg_lat = sum(latencies) / len(latencies)

        s1 = statuses[0] if len(statuses) > 0 else "?"
        s2 = statuses[1] if len(statuses) > 1 else "?"
        s3 = statuses[2] if len(statuses) > 2 else "?"

        errors = statuses.count("ERROR")
        fails = statuses.count("FAIL")
        warns = statuses.count("WARN")
        if errors > 0:
            verdict = "⚠ Error (see details)"
        elif fails == 0 and warns == 0:
            verdict = "✓ Recommended"
        elif fails == 0:
            verdict = "⚠ Usable (with caveats)"
        else:
            verdict = "✗ Not recommended"

        print(f"{model_name:<25} {s1:<10} {s2:<13} {s3:<10} {avg_lat:<12.2f}s {verdict}")

    # Save reports
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = BENCHMARKS_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report: {json_path}")

    md_path = BENCHMARKS_DIR / "benchmark_results.md"
    md_path.write_text(generate_markdown(report))
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
