import sqlite3
from datetime import UTC, date, datetime, timedelta

import aiosqlite


class CalendarService:
    def __init__(self, db_path: str = "data/voice_agent.db"):
        self._db_path = db_path

    # --- Synchronous helpers for deterministic in-turn booking -------------
    # The conversation manager drives the booking flow in code (no LLM), so it
    # needs sync slot access. SQLite handles a single writer fine for one call.

    def available_slots_sync(
        self,
        legal_area: str = "",
        exclude_ids: list[int] | None = None,
        exclude_times: list[str] | None = None,
        lawyer: str = "",
        limit: int = 3,
    ) -> list[dict]:
        exclude_ids = exclude_ids or []
        exclude_times = exclude_times or []
        now = datetime.now()
        today = now.date().isoformat()
        min_time = (now + timedelta(hours=4)).strftime("%H:%M")
        query = "SELECT * FROM slots WHERE is_booked = 0 AND (date > ? OR (date = ? AND time >= ?))"
        params: list = [today, today, min_time]
        if legal_area and legal_area != "unknown":
            query += " AND (legal_area = ? OR legal_area IS NULL)"
            params.append(legal_area)
        # Prefer a specific lawyer (caller asked for them by name).
        if lawyer:
            query += " AND lawyer_name LIKE ?"
            params.append(f"%{lawyer}%")
        if exclude_ids:
            query += f" AND id NOT IN ({','.join('?' * len(exclude_ids))})"
            params.extend(exclude_ids)
        # Exclude whole (date, time) pairs the caller already declined — so a
        # decline never re-offers the same time via a different lawyer's slot.
        if exclude_times:
            query += f" AND (date || ' ' || time) NOT IN ({','.join('?' * len(exclude_times))})"
            params.extend(exclude_times)
        query += " ORDER BY date, time LIMIT ?"
        params.append(limit * 4)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        finally:
            conn.close()
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for row in rows:
            key = (row["date"], row["time"])
            if key not in seen:
                seen.add(key)
                deduped.append(row)
                if len(deduped) >= limit:
                    break
        return deduped

    def upsert_caller_sync(
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
        preferred_time: str = "",
    ) -> None:
        """Insert or update the caller row for this call (one row per call_id).

        Called after every key decision so captured data is persisted immediately,
        not only on a clean disconnect.
        """
        cols = dict(
            name=name,
            phone=phone,
            email=email,
            legal_area=legal_area,
            matter_type=matter_type,
            matter_summary=matter_summary,
            case_reference=case_reference,
            insurance_number=insurance_number,
            outcome=outcome,
            preferred_time=preferred_time,
        )
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT id FROM callers WHERE call_id = ? ORDER BY id DESC LIMIT 1", [call_id]
            ).fetchone()
            if row:
                assignments = ", ".join(f"{c} = ?" for c in cols)
                conn.execute(
                    f"UPDATE callers SET {assignments} WHERE id = ?",
                    [*cols.values(), row[0]],
                )
            else:
                names = ", ".join(["call_id", *cols])
                placeholders = ", ".join("?" * (len(cols) + 1))
                conn.execute(
                    f"INSERT INTO callers ({names}) VALUES ({placeholders})",
                    [call_id, *cols.values()],
                )
            conn.commit()
        finally:
            conn.close()

    def book_slot_sync(
        self,
        slot_id: int,
        call_id: str,
        caller_name: str = "",
        caller_email: str = "",
        caller_phone: str = "",
        matter_type: str = "",
    ) -> dict | None:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "UPDATE slots SET is_booked = 1 WHERE id = ? AND is_booked = 0", [slot_id]
            )
            if cur.rowcount == 0:
                conn.commit()
                return None  # already taken
            now = datetime.now(UTC).isoformat()
            cur = conn.execute(
                """INSERT INTO bookings
                   (slot_id, call_id, caller_name, caller_email, caller_phone,
                    matter_type, matter_description, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, '', ?)""",
                [slot_id, call_id, caller_name, caller_email, caller_phone, matter_type, now],
            )
            conn.commit()
            row = conn.execute(
                """SELECT b.*, s.date, s.time, s.lawyer_name, s.duration_minutes
                   FROM bookings b JOIN slots s ON b.slot_id = s.id WHERE b.id = ?""",
                [cur.lastrowid],
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

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
                    preferred_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Idempotent column add for databases created before preferred_time.
            try:
                await db.execute("ALTER TABLE callers ADD COLUMN preferred_time TEXT")
            except Exception:
                pass
            await db.commit()

    async def needs_reseed(self) -> bool:
        """True when all slots are in the past or no slots exist."""
        today = date.today().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM slots WHERE date >= ? AND is_booked = 0", [today]
            ) as cur:
                count = (await cur.fetchone())[0]
                return count == 0

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
        preferred_time: str = "",
    ) -> int | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """INSERT INTO callers
                   (call_id, name, phone, email, legal_area, matter_type,
                    matter_summary, case_reference, insurance_number, outcome,
                    preferred_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    preferred_time,
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
