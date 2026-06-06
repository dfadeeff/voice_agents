"""Tests for the CallLogger transcript writer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.pipeline.processors import CallLogger


class TestCallLogger:
    def test_log_adds_entries(self):
        logger = CallLogger("test-001")
        logger.log("user", "Hello")
        logger.log("agent", "Hi there!")
        assert len(logger.entries) == 2
        assert logger.entries[0]["role"] == "user"
        assert logger.entries[1]["text"] == "Hi there!"

    def test_entries_have_timestamps(self):
        logger = CallLogger("test-001")
        logger.log("user", "Hello")
        assert "timestamp" in logger.entries[0]

    def test_save_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            with patch("app.pipeline.processors.LOGS_DIR", log_dir):
                logger = CallLogger("test-002")
                logger.log("user", "Hi")
                logger.log("agent", "Hello!")
                logger.save()

                path = log_dir / "test-002.json"
                assert path.exists()

                data = json.loads(path.read_text())
                assert data["call_id"] == "test-002"
                assert len(data["transcript"]) == 2
                assert "start_time" in data
                assert "end_time" in data

    def test_save_empty_log_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            with patch("app.pipeline.processors.LOGS_DIR", log_dir):
                logger = CallLogger("test-003")
                logger.save()
                assert not log_dir.exists()
