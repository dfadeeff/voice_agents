import os
import tempfile

import pytest
import pytest_asyncio
from app.conversation.manager import ConversationManager
from app.models.schemas import WordInfo
from app.services.calendar import CalendarService
from app.tools.registry import build_default_registry


@pytest.fixture
def call_id():
    return "test-call-001"


@pytest.fixture
def conversation(call_id):
    return ConversationManager(call_id, lang="en")


@pytest.fixture
def conversation_with_low_confidence(conversation):
    """Conversation where the current turn has low-confidence words."""
    word_infos = [
        WordInfo(word="Siobhan", start_time=0.0, end_time=0.5, confidence=0.4),
        WordInfo(word="Murphy", start_time=0.5, end_time=1.0, confidence=0.9),
    ]
    conversation.add_user_message("My name is Siobhan Murphy", word_infos)
    return conversation


@pytest.fixture
def conversation_with_high_confidence(conversation):
    """Conversation where the current turn has high-confidence words."""
    word_infos = [
        WordInfo(word="John", start_time=0.0, end_time=0.3, confidence=0.95),
        WordInfo(word="Smith", start_time=0.3, end_time=0.6, confidence=0.92),
    ]
    conversation.add_user_message("My name is John Smith", word_infos)
    return conversation


@pytest_asyncio.fixture
async def calendar():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cal = CalendarService(db_path=db_path)
        await cal.init_db()
        yield cal


@pytest_asyncio.fixture
async def seeded_calendar(calendar):
    """Calendar with a few test slots."""
    import aiosqlite

    async with aiosqlite.connect(calendar._db_path) as db:
        await db.executemany(
            "INSERT INTO slots (date, time, duration_minutes, legal_area, lawyer_name)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                ("2026-06-10", "09:00", 30, "employment", "Sarah Chen"),
                ("2026-06-10", "10:00", 30, "employment", "Sarah Chen"),
                ("2026-06-10", "14:00", 30, "tenancy", "James Wilson"),
                ("2026-06-11", "09:00", 30, "employment", "Sarah Chen"),
                ("2026-06-11", "11:00", 30, "tenancy", "James Wilson"),
            ],
        )
        await db.commit()
    return calendar


@pytest_asyncio.fixture
async def registry(seeded_calendar):
    return build_default_registry(seeded_calendar)
