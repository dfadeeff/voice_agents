from datetime import UTC, datetime

import aiosqlite


class CalendarService:
    def __init__(self, db_path: str = "data/voice_agent.db"):
        self._db_path = db_path

    async def init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    duration_minutes INTEGER DEFAULT 30,
                    legal_area TEXT,
                    lawyer_name TEXT NOT NULL,
                    is_booked INTEGER DEFAULT 0
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS bookings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slot_id INTEGER UNIQUE REFERENCES slots(id),
                    call_id TEXT NOT NULL,
                    caller_name TEXT,
                    caller_email TEXT,
                    caller_phone TEXT,
                    matter_type TEXT,
                    matter_description TEXT,
                    created_at TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS call_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    evaluator_model TEXT,
                    completeness_score REAL,
                    naturalness_score REAL,
                    accuracy_score REAL,
                    escalation_appropriateness REAL,
                    overall_score REAL,
                    findings TEXT,
                    improvement_suggestions TEXT,
                    prompt_version TEXT,
                    evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    prompts_snapshot TEXT NOT NULL,
                    parent_version TEXT,
                    change_description TEXT,
                    performance_delta TEXT,
                    status TEXT DEFAULT 'draft',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT UNIQUE NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    legal_area TEXT,
                    outcome TEXT,
                    turn_count INTEGER,
                    transcript TEXT,
                    entities_json TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS callers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    name TEXT,
                    phone TEXT,
                    email TEXT,
                    legal_area TEXT,
                    matter_type TEXT,
                    matter_summary TEXT,
                    case_reference TEXT,
                    insurance_number TEXT,
                    outcome TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    async def get_slot_by_id(self, slot_id: int) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM slots WHERE id = ?", [slot_id]) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_available_slots(
        self,
        date: str,
        legal_area: str = "",
        time_preference: str = "",
    ) -> list[dict]:
        query = "SELECT * FROM slots WHERE date = ? AND is_booked = 0"
        params: list = [date]

        if legal_area and legal_area != "unknown":
            query += " AND (legal_area = ? OR legal_area IS NULL)"
            params.append(legal_area)

        if time_preference:
            if time_preference.lower() == "morning":
                query += " AND time < '12:00'"
            elif time_preference.lower() == "afternoon":
                query += " AND time >= '12:00'"
            elif ":" in time_preference:
                query += " AND time = ?"
                params.append(time_preference)

        query += " ORDER BY time LIMIT 5"

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_next_available(
        self, after_date: str, legal_area: str = "", limit: int = 3
    ) -> list[dict]:
        query = "SELECT * FROM slots WHERE date > ? AND is_booked = 0"
        params: list = [after_date]

        if legal_area and legal_area != "unknown":
            query += " AND (legal_area = ? OR legal_area IS NULL)"
            params.append(legal_area)

        query += " ORDER BY date, time LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def create_booking(
        self,
        slot_id: int,
        call_id: str,
        caller_name: str = "",
        caller_email: str = "",
        caller_phone: str = "",
        matter_type: str = "",
        matter_description: str = "",
    ) -> dict | None:
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "UPDATE slots SET is_booked = 1 WHERE id = ? AND is_booked = 0", [slot_id]
            )
            if cursor.rowcount == 0:
                return None
            cursor = await db.execute(
                """INSERT INTO bookings
                   (slot_id, call_id, caller_name, caller_email, caller_phone,
                    matter_type, matter_description, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    slot_id,
                    call_id,
                    caller_name,
                    caller_email,
                    caller_phone,
                    matter_type,
                    matter_description,
                    now,
                ],
            )
            await db.commit()
            booking_id = cursor.lastrowid

            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT b.*, s.date, s.time, s.lawyer_name, s.duration_minutes
                   FROM bookings b JOIN slots s ON b.slot_id = s.id
                   WHERE b.id = ?""",
                [booking_id],
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def save_caller(
        self,
        call_id: str,
        name: str = "",
        phone: str = "",
        email: str = "",
        legal_area: str = "",
        matter_type: str = "",
        matter_summary: str = "",
        case_reference: str = "",
        insurance_number: str = "",
        outcome: str = "",
    ) -> int | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """INSERT INTO callers
                   (call_id, name, phone, email, legal_area, matter_type,
                    matter_summary, case_reference, insurance_number, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    call_id,
                    name,
                    phone,
                    email,
                    legal_area,
                    matter_type,
                    matter_summary,
                    case_reference,
                    insurance_number,
                    outcome,
                ],
            )
            await db.commit()
            return cursor.lastrowid

    async def log_call(
        self,
        call_id: str,
        started_at: str,
        ended_at: str,
        legal_area: str,
        outcome: str,
        turn_count: int,
        transcript: str,
        entities_json: str,
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO call_logs
                   (call_id, started_at, ended_at, legal_area, outcome,
                    turn_count, transcript, entities_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    call_id,
                    started_at,
                    ended_at,
                    legal_area,
                    outcome,
                    turn_count,
                    transcript,
                    entities_json,
                ],
            )
            await db.commit()
