"""Seed the SQLite database with 2 weeks of appointment slots."""

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiosqlite

DB_PATH = "data/voice_agent.db"

LAWYERS = {
    "employment": ["Sarah Mitchell", "James Cooper"],
    "tenancy": ["Rachel Adams", "David Chen"],
    "traffic": ["Michael Weber", "Lisa Hoffmann"],
}

TIMES = [
    "09:00",
    "09:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
    "16:30",
]

PRE_BOOKED = [
    (0, "10:00", "employment"),
    (0, "14:00", "employment"),
    (1, "09:30", "tenancy"),
    (1, "15:00", "tenancy"),
    (2, "10:00", "employment"),
    (3, "14:00", "tenancy"),
]


async def seed():
    Path("data").mkdir(exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
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
        await db.execute("DELETE FROM bookings")
        await db.execute("DELETE FROM slots")

        today = date.today()
        slot_id = 0

        for day_offset in range(14):
            d = today + timedelta(days=day_offset)
            if d.weekday() >= 5:  # skip weekends
                continue

            for area, lawyers in LAWYERS.items():
                for lawyer in lawyers:
                    for t in TIMES:
                        slot_id += 1
                        is_booked = any(
                            day_offset == pb[0] and t == pb[1] and area == pb[2]
                            for pb in PRE_BOOKED
                        )

                        await db.execute(
                            """INSERT INTO slots
                               (id, date, time, duration_minutes,
                                legal_area, lawyer_name, is_booked)
                               VALUES (?, ?, ?, 30, ?, ?, ?)""",
                            [slot_id, d.isoformat(), t, area, lawyer, int(is_booked)],
                        )

        await db.commit()

        async with db.execute("SELECT COUNT(*) FROM slots") as cur:
            total = (await cur.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM slots WHERE is_booked = 1") as cur:
            booked = (await cur.fetchone())[0]

        print(f"Seeded {total} slots ({booked} pre-booked, {total - booked} available)")


if __name__ == "__main__":
    asyncio.run(seed())
