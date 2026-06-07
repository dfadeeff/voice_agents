import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin import router as admin_router
from app.api.health import router as health_router
from app.api.twilio import router as twilio_router
from app.api.ws import router as ws_router
from app.config import Settings
from app.services.calendar import CalendarService
from app.services.evaluator import CallEvaluator
from app.services.prompt_optimizer import PromptOptimizer
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
    app.state.calendar = calendar

    logger.info("Building tool registry...")
    app.state.tool_registry = build_default_registry(calendar)

    db_path = settings.db_url.replace("sqlite+aiosqlite:///", "")
    if settings.evaluator_enabled:
        logger.info("Initializing evaluator (model=%s)...", settings.evaluator_model)
        app.state.evaluator = CallEvaluator(
            db_path=db_path,
            provider=settings.evaluator_provider,
            model=settings.evaluator_model,
            base_url=settings.ollama_base_url,
            api_key=settings.openai_api_key,
        )
        app.state.prompt_optimizer = PromptOptimizer(
            db_path=db_path,
            provider=settings.evaluator_provider,
            model=settings.evaluator_model,
            base_url=settings.ollama_base_url,
            api_key=settings.openai_api_key,
        )
    else:
        app.state.evaluator = None
        app.state.prompt_optimizer = None

    logger.info("Voice agent ready (providers created per-call by Pipecat)")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(title="Voice AI Agent", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ws_router)
    app.include_router(twilio_router)
    app.include_router(admin_router)

    if FRONTEND_PATH.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(FRONTEND_PATH / "index.html")

    return app


app = create_app()
