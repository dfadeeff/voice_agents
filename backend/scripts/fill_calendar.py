"""Mark slots as booked so the 'no available slots' path can be demoed.

Usage:
    python3 scripts/fill_calendar.py            # fill (book) every slot
    python3 scripts/fill_calendar.py employment # fill one area only

Re-run `make seed` afterwards to restore an open calendar.
"""

import sqlite3
import sys

DB_PATH = "data/voice_agent.db"


def main() -> None:
    area = sys.argv[1] if len(sys.argv) > 1 else None
    conn = sqlite3.connect(DB_PATH)
    if area:
        cur = conn.execute("UPDATE slots SET is_booked = 1 WHERE legal_area = ?", [area])
        scope = f"area '{area}'"
    else:
        cur = conn.execute("UPDATE slots SET is_booked = 1")
        scope = "all areas"
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM slots WHERE is_booked = 0").fetchone()[0]
    conn.close()
    print(f"Booked {cur.rowcount} slots ({scope}). {remaining} slots still free.")
    print("Run `make seed` to restore an open calendar.")


if __name__ == "__main__":
    main()
