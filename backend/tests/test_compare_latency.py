"""Tests for the latency-comparison harness (pure aggregation/formatting core)."""

from scripts import compare_latency
from scripts.compare_latency import classify, record_row, render, summarize, vendor


def _comp(avg, samples, lo=0, hi=0):
    return {"avg_ms": avg, "samples": samples, "min_ms": lo, "max_ms": hi}


def _log(ttfa_avg, ttfa_n, components):
    return {
        "metrics": {
            "ttfa_avg_ms": ttfa_avg,
            "ttfa_samples": ttfa_n,
            "component_ttfb": components,
        }
    }


class TestClassifyAndVendor:
    def test_classify_by_marker(self):
        assert classify("LocalWhisperSTTService#0") == "STT"
        assert classify("DeepgramSTTService") == "STT"
        assert classify("OLLamaLLMService#11") == "LLM"
        assert classify("CartesiaTTSService#2") == "TTS"
        assert classify("MetricsProcessor#0") is None

    def test_vendor_shortens_and_strips_instance(self):
        assert vendor("LocalWhisperSTTService#11") == "Whisper"
        assert vendor("CartesiaTTSService#0") == "Cartesia"
        assert vendor("OpenAILLMService") == "OpenAI"
        assert vendor("UnknownThing#3") == "UnknownThing"


class TestSummarize:
    def test_sample_weighted_average_across_calls(self):
        # Two calls: STT avg 1000ms×2 samples, then 2000ms×8 samples
        # → weighted mean = (1000*2 + 2000*8) / 10 = 1800ms (not the naive 1500).
        logs = [
            _log(200, 1, {"LocalWhisperSTTService#0": _comp(1000, 2, 0, 1000)}),
            _log(220, 3, {"LocalWhisperSTTService#1": _comp(2000, 8, 1500, 2100)}),
        ]
        s = summarize(logs)
        assert s.calls == 2
        assert round(s.roles["STT"].avg_ms) == 1800
        assert s.roles["STT"].samples == 10
        assert s.roles["STT"].min_ms == 0
        assert s.roles["STT"].max_ms == 2100
        # TTFA weighted: (200*1 + 220*3) / 4 = 215
        assert round(s.ttfa_avg_ms) == 215

    def test_zero_sample_component_ignored(self):
        # The fast-pathed LLM logs 0ms / 1 sample placeholders — but a *0-sample*
        # component must not pollute the average or claim a vendor.
        s = summarize([_log(0, 0, {"OLLamaLLMService#0": _comp(0, 0)})])
        assert "LLM" not in s.roles

    def test_missing_metrics_block(self):
        assert summarize([{}]).calls == 1


class TestRender:
    def test_table_has_rows_and_vendor_arrow(self):
        a = summarize([_log(200, 5, {"LocalWhisperSTTService#0": _comp(1800, 5, 0, 2000)})])
        b = summarize([_log(150, 5, {"DeepgramSTTService#0": _comp(200, 5, 100, 300)})])
        out = render(a, b, "local", "cloud")
        assert "TTFA (perceived)" in out
        assert "STT (Whisper→Deepgram)" in out  # vendors differ → arrow
        assert "200 ms" in out and "1800 ms" in out

    def test_same_vendor_no_arrow(self):
        a = summarize([_log(200, 1, {"LocalPiperTTSService#0": _comp(130, 5, 0, 200)})])
        b = summarize([_log(200, 1, {"LocalPiperTTSService#1": _comp(124, 5, 0, 250)})])
        out = render(a, b, "before", "after")
        assert "TTS (Piper)" in out
        assert "→" not in out.split("TTS")[1].split("\n")[0]

    def test_fastpath_llm_note_shown(self):
        a = summarize([_log(200, 1, {"OLLamaLLMService#0": _comp(0, 1)})])
        out = render(a, a, "a", "b")
        assert "fast-pathed" in out


class TestRecord:
    def test_record_row_columns(self):
        s = summarize(
            [
                _log(
                    157,
                    5,
                    {
                        "DeepgramSTTService#0": _comp(390, 5, 0, 800),
                        "LocalPiperTTSService#0": _comp(126, 6, 0, 300),
                    },
                )
            ]
        )
        row = record_row(s, "cloud-deepgram", "2026-06-09 22:19")
        assert row.startswith("| cloud-deepgram | 1 | 157 ms |")
        assert "Deepgram 390 ms" in row
        assert "Piper 126 ms" in row
        assert "—" in row  # LLM absent → em dash
        assert row.endswith("2026-06-09 22:19 |")

    def test_append_result_creates_header_then_appends(self, tmp_path, monkeypatch):
        target = tmp_path / "latency-results.md"
        monkeypatch.setattr(compare_latency, "RESULTS_FILE", target)
        compare_latency.append_result("| a | 1 | 100 ms | x | y | z | t |")
        compare_latency.append_result("| b | 1 | 200 ms | x | y | z | t |")
        text = target.read_text()
        assert text.startswith("# Latency results")
        assert "| config | calls |" in text  # header table written once
        assert text.count("| config | calls |") == 1
        assert "| a | 1 |" in text and "| b | 1 |" in text
