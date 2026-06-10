import os
import tempfile

import pytest
import pytest_asyncio

from app.conversation.manager import ConversationManager
from app.services.calendar import CalendarService
from app.tools.registry import build_default_registry


@pytest.fixture
def call_id():
    return "test-call-001"


@pytest.fixture
def conversation(call_id):
    return ConversationManager(call_id, lang="en")


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
