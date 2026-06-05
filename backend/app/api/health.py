from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    app = request.app
    checks = {
        "database": app.state.calendar is not None,
        "tools": app.state.tool_registry is not None,
    }
    all_ok = all(checks.values())
    return {"ready": all_ok, "checks": checks}
