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
        assert "call_logs" in tables
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


class TestSaveCaller:
    @pytest.mark.asyncio
    async def test_save_caller_persists(self, calendar):
        row_id = await calendar.save_caller(
            call_id="call-100",
            name="Amelie Sommer",
            phone="+491761234567",
            email="amelie@example.com",
            legal_area="employment",
            matter_type="dismissal",
            matter_summary="unfair dismissal",
            case_reference="44/24",
            insurance_number="",
            outcome="booked",
        )
        assert row_id is not None

        import aiosqlite

        async with aiosqlite.connect(calendar._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM callers WHERE call_id = ?", ["call-100"]
            ) as cursor:
                row = await cursor.fetchone()
        assert row is not None
        assert dict(row)["name"] == "Amelie Sommer"
        assert dict(row)["case_reference"] == "44/24"
        assert dict(row)["legal_area"] == "employment"

    @pytest.mark.asyncio
    async def test_save_caller_with_insurance_number(self, calendar):
        await calendar.save_caller(
            call_id="call-101",
            name="Max Müller",
            phone="+491769876543",
            legal_area="traffic",
            matter_type="accident",
            insurance_number="VS-2026-12345",
            outcome="callback",
        )

        import aiosqlite

        async with aiosqlite.connect(calendar._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM callers WHERE call_id = ?", ["call-101"]
            ) as cursor:
                row = await cursor.fetchone()
        assert dict(row)["insurance_number"] == "VS-2026-12345"
        assert dict(row)["outcome"] == "callback"


class TestCallLogs:
    @pytest.mark.asyncio
    async def test_log_call(self, calendar):
        await calendar.log_call(
            call_id="call-001",
            started_at="2026-06-10T09:00:00Z",
            ended_at="2026-06-10T09:05:00Z",
            legal_area="employment",
            outcome="booked",
            turn_count=8,
            transcript="[call transcript]",
            entities_json='{"name": "John"}',
        )

        import aiosqlite

        async with aiosqlite.connect(calendar._db_path) as db:
            async with db.execute(
                "SELECT * FROM call_logs WHERE call_id = ?", ["call-001"]
            ) as cursor:
                row = await cursor.fetchone()
        assert row is not None
