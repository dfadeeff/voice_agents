"""Tests for the calendar service — slots, bookings, call logs."""

import pytest


class TestCalendarInit:
    @pytest.mark.asyncio
    async def test_init_creates_tables(self, calendar):
        import aiosqlite

        async with aiosqlite.connect(calendar._db_path) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = [row[0] for row in await cursor.fetchall()]

        assert "slots" in tables
        assert "bookings" in tables
        assert "callers" in tables

    @pytest.mark.asyncio
    async def test_init_is_idempotent(self, calendar):
        await calendar.init_db()
        await calendar.init_db()


class TestSlots:
    @pytest.mark.asyncio
    async def test_get_available_slots(self, seeded_calendar):
        slots = await seeded_calendar.get_available_slots(date="2026-06-10")
        assert len(slots) == 3

    @pytest.mark.asyncio
    async def test_filter_by_legal_area(self, seeded_calendar):
        slots = await seeded_calendar.get_available_slots(
            date="2026-06-10", legal_area="employment"
        )
        assert len(slots) == 2
        for s in slots:
            assert s["legal_area"] == "employment"

    @pytest.mark.asyncio
    async def test_filter_morning(self, seeded_calendar):
        slots = await seeded_calendar.get_available_slots(
            date="2026-06-10", time_preference="morning"
        )
        for s in slots:
            assert s["time"] < "12:00"

    @pytest.mark.asyncio
    async def test_filter_afternoon(self, seeded_calendar):
        slots = await seeded_calendar.get_available_slots(
            date="2026-06-10", time_preference="afternoon"
        )
        for s in slots:
            assert s["time"] >= "12:00"

    @pytest.mark.asyncio
    async def test_filter_specific_time(self, seeded_calendar):
        slots = await seeded_calendar.get_available_slots(
            date="2026-06-10", time_preference="09:00"
        )
        assert len(slots) == 1
        assert slots[0]["time"] == "09:00"

    @pytest.mark.asyncio
    async def test_no_slots_returns_empty(self, seeded_calendar):
        slots = await seeded_calendar.get_available_slots(date="2026-12-25")
        assert slots == []

    @pytest.mark.asyncio
    async def test_get_next_available(self, seeded_calendar):
        slots = await seeded_calendar.get_next_available(
            after_date="2026-06-10", legal_area="employment"
        )
        assert len(slots) > 0
        for s in slots:
            assert s["date"] > "2026-06-10"


class TestBookings:
    @pytest.mark.asyncio
    async def test_create_booking(self, seeded_calendar):
        booking = await seeded_calendar.create_booking(
            slot_id=1,
            call_id="test-001",
            caller_name="John Smith",
            caller_email="john@example.com",
            caller_phone="555-0100",
            matter_type="employment",
            matter_description="Unfair dismissal",
        )
        assert booking is not None
        assert booking["caller_name"] == "John Smith"
        assert booking["lawyer_name"] == "Sarah Chen"

    @pytest.mark.asyncio
    async def test_booking_marks_slot_unavailable(self, seeded_calendar):
        await seeded_calendar.create_booking(slot_id=1, call_id="test-001")
        slots = await seeded_calendar.get_available_slots(
            date="2026-06-10", time_preference="09:00", legal_area="employment"
        )
        assert len(slots) == 0

    @pytest.mark.asyncio
    async def test_double_booking_returns_none(self, seeded_calendar):
        await seeded_calendar.create_booking(slot_id=1, call_id="test-001")
        result = await seeded_calendar.create_booking(slot_id=1, call_id="test-002")
        assert result is None

    @pytest.mark.asyncio
    async def test_booking_nonexistent_slot(self, seeded_calendar):
        result = await seeded_calendar.create_booking(slot_id=9999, call_id="test-001")
        assert result is None


class TestFutureSlotFiltering:
    """Regression: slots must be filtered against a full datetime cutoff. A bare
    'now + 4h' time-of-day wraps past midnight late at night and wrongly offered
    today's already-past slots."""

    def _cal_with(self, rows):
        import asyncio
        import sqlite3
        import tempfile

        from app.services.calendar import CalendarService

        db = tempfile.mktemp(suffix=".db")
        cal = CalendarService(db_path=db)
        asyncio.run(cal.init_db())
        conn = sqlite3.connect(db)
        for d, t in rows:
            conn.execute(
                "INSERT INTO slots (date,time,legal_area,lawyer_name,is_booked) VALUES (?,?,?,?,0)",
                (d, t, "traffic", "Michael Weber"),
            )
        conn.commit()
        conn.close()
        return cal

    def _freeze_late(self, monkeypatch):
        from datetime import datetime as real_datetime

        from app.services import calendar as cal_mod

        class _FakeDateTime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime(2026, 6, 9, 23, 0)  # 11pm — today's slots are past

        monkeypatch.setattr(cal_mod, "datetime", _FakeDateTime)

    def test_today_past_slots_excluded_at_night(self, monkeypatch):
        self._freeze_late(monkeypatch)
        cal = self._cal_with(
            [("2026-06-09", "09:00"), ("2026-06-09", "13:00"), ("2026-06-10", "09:00")]
        )
        pairs = {(s["date"], s["time"]) for s in cal.available_slots_sync("traffic")}
        assert ("2026-06-09", "09:00") not in pairs  # past
        assert ("2026-06-09", "13:00") not in pairs  # past
        assert ("2026-06-10", "09:00") in pairs  # future

    def test_find_slot_skips_past_time_today(self, monkeypatch):
        import sqlite3

        self._freeze_late(monkeypatch)
        cal = self._cal_with([("2026-06-09", "13:00")])  # only a past 13:00 today
        assert cal.find_slot_sync("13:00", "traffic") is None
        conn = sqlite3.connect(cal._db_path)
        conn.execute(
            "INSERT INTO slots (date,time,legal_area,lawyer_name,is_booked) "
            "VALUES ('2026-06-10','13:00','traffic','Michael Weber',0)"
        )
        conn.commit()
        conn.close()
        slot = cal.find_slot_sync("13:00", "traffic")
        assert slot and slot["date"] == "2026-06-10"
