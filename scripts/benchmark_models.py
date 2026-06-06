"""
Benchmark local LLMs for voice agent suitability.

Tests each model on three dimensions that matter for phone calls:
  1. Latency — time to first token + total response time
  2. Tool calling — does the model use the API correctly or dump JSON as text?
  3. Conversation — does the model greet naturally?

Usage:
    python3 scripts/benchmark_models.py                    # test all installed models
    python3 scripts/benchmark_models.py qwen3:4b llama3.1  # test specific models
"""

import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are the receptionist at a law firm. "
    "You answer phone calls in a warm, professional manner. "
    "The firm handles employment law and tenancy law. "
    "Keep responses to 1-2 short sentences. "
    "ALWAYS respond with speech. Never output JSON."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_caller_intent",
            "description": "Classify what the caller wants: general_info or book_consultation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["general_info", "book_consultation"],
                    }
                },
                "required": ["intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_legal_area",
            "description": "Classify the caller's legal area: employment, tenancy, or unknown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "legal_area": {
                        "type": "string",
                        "enum": ["employment", "tenancy", "unknown"],
                    }
                },
                "required": ["legal_area"],
            },
        },
    },
]

SCENARIOS = [
    {
        "name": "greeting (no tools)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[A new caller has connected]"},
        ],
        "tools": None,
        "expect": "speech",
    },
    {
        "name": "greeting (with tools)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[A new caller has connected]"},
        ],
        "tools": TOOLS,
        "expect": "speech",
    },
    {
        "name": "tool call (employment issue)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[A new caller has connected]"},
            {"role": "assistant", "content": "Hello! Thank you for calling. How can I help you today?"},
            {"role": "user", "content": "I was unfairly dismissed from my job last week and I need legal advice."},
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
        with urllib.request.urlopen(req, timeout=60) as resp:
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
        return "FAIL", f"Empty response"

    if expected == "tool_call":
        if has_tool_call:
            tc = tool_calls[0]["function"]
            args = tc.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = args
            speech_note = f" + speech" if has_speech else ""
            return "PASS", f"{tc['name']}({json.dumps(args)}){speech_note}"
        if has_json_text:
            return "FAIL", f"JSON as text (not API): {content.strip()[:60]}"
        if has_speech:
            return "WARN", f"Speech only, no tool call: {content.strip()[:60]}"
        return "FAIL", "No tool call and no speech"

    return "???", content[:60]


def warmup_model(model):
    """Load model into memory with a tiny request."""
    call_ollama(model, [{"role": "user", "content": "hi"}])


def get_installed_models():
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            # skip embedding models
            return [m for m in models if "embed" not in m]
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        sys.exit(1)


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

    results = {}

    for model in models:
        print(f"\n{'─' * 100}")
        print(f"MODEL: {model}")
        print(f"{'─' * 100}")

        print("  Warming up (loading into memory)...", end=" ", flush=True)
        warmup_model(model)
        print("done")

        model_results = []
        for scenario in SCENARIOS:
            print(f"\n  [{scenario['name']}]")
            result, elapsed = call_ollama(model, scenario["messages"], scenario["tools"])
            status, detail = analyze_response(result, scenario["expect"])

            tokens = result.get("usage", {})
            prompt_tok = tokens.get("prompt_tokens", "?")
            comp_tok = tokens.get("completion_tokens", "?")

            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "ERROR": "⚠"}.get(status, "?")
            color_status = status

            print(f"    {icon} {color_status}  ({elapsed:.2f}s, {prompt_tok}→{comp_tok} tokens)")
            print(f"    → {detail}")

            model_results.append({
                "scenario": scenario["name"],
                "status": status,
                "latency": elapsed,
                "detail": detail,
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
            })

        results[model] = model_results

    # Summary table
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"\n{'Model':<25} {'Greeting':<10} {'Greeting+Tools':<15} {'Tool Call':<10} {'Avg Latency':<12} {'Verdict'}")
    print(f"{'─' * 25} {'─' * 10} {'─' * 15} {'─' * 10} {'─' * 12} {'─' * 20}")

    for model, model_results in results.items():
        statuses = [r["status"] for r in model_results]
        latencies = [r["latency"] for r in model_results]
        avg_lat = sum(latencies) / len(latencies)

        s1 = statuses[0] if len(statuses) > 0 else "?"
        s2 = statuses[1] if len(statuses) > 1 else "?"
        s3 = statuses[2] if len(statuses) > 2 else "?"

        fails = statuses.count("FAIL")
        warns = statuses.count("WARN")
        if fails == 0 and warns == 0:
            verdict = "✓ Recommended"
        elif fails == 0:
            verdict = "⚠ Usable (with caveats)"
        else:
            verdict = "✗ Not recommended"

        print(f"{model:<25} {s1:<10} {s2:<15} {s3:<10} {avg_lat:<12.2f}s {verdict}")

    print()


if __name__ == "__main__":
    main()
