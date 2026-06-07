from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    app = request.app
    calendar = app.state.calendar
    has_calendar = calendar is not None
    has_slots = False
    if has_calendar:
        try:
            from datetime import date, timedelta

            # Check today + next 7 days (seed script skips weekends)
            for offset in range(8):
                day = (date.today() + timedelta(days=offset)).isoformat()
                slots = await calendar.get_available_slots(date=day)
                if len(slots) > 0:
                    has_slots = True
                    break
        except Exception:
            pass
    checks = {
        "database": has_calendar,
        "slots_available": has_slots,
        "tools": app.state.tool_registry is not None,
    }
    all_ok = all(checks.values())
    return {"ready": all_ok, "checks": checks}
