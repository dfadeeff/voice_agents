import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.twilio import router as twilio_router
from app.api.ws import router as ws_router
from app.config import Settings
from app.services.calendar import CalendarService
from app.tools.registry import build_default_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_PATH = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings

    logger.info("Initializing database...")
    calendar = CalendarService(db_path=settings.db_url.replace("sqlite+aiosqlite:///", ""))
    await calendar.init_db()

    if await calendar.needs_reseed():
        logger.info("Calendar slots are stale or empty — reseeding with future dates...")
        from scripts.seed_calendar import seed

        await seed()
        logger.info("Calendar reseeded.")

    app.state.calendar = calendar

    logger.info("Building tool registry...")
    app.state.tool_registry = build_default_registry(calendar)

    if settings.stt_provider == "whisper":
        from app.pipeline.local_whisper import preload_whisper

        preload_whisper(
            settings.whisper_model_size, settings.whisper_device, settings.whisper_compute_type
        )

    logger.info("Voice agent ready")
    yield
    logger.info("Shutting down...")


def _cors_origins(raw: str) -> list[str]:
    """Parse the comma-separated CORS allow-list ("*" → allow all)."""
    return [o.strip() for o in raw.split(",") if o.strip()] or ["*"]


def create_app() -> FastAPI:
    app = FastAPI(title="Voice AI Agent", lifespan=lifespan)

    # Middleware must be registered before startup, so read origins from a fresh
    # Settings here (lifespan builds its own for the rest of app.state).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(Settings().cors_allow_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ws_router)
    app.include_router(twilio_router)

    if FRONTEND_PATH.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(FRONTEND_PATH / "index.html")

    return app


app = create_app()
