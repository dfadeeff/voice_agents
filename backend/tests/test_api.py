"""Smoke tests: does the FastAPI app start, do endpoints respond?"""

import os
import tempfile

import pytest_asyncio
from app.main import create_app
from app.services.calendar import CalendarService
from app.tools.registry import build_default_registry
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def live_app():
    """Create app with real (temp) database and tool registry — no data/ dir needed."""
    app = create_app()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        calendar = CalendarService(db_path=db_path)
        await calendar.init_db()
        app.state.calendar = calendar
        app.state.tool_registry = build_default_registry(calendar)
        yield app


class TestHealth:
    async def test_health_returns_ok(self):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestReadyWithLifespan:
    async def test_ready_checks_pass(self, live_app):
        async with AsyncClient(
            transport=ASGITransport(app=live_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ready"] is True
            assert data["checks"]["database"] is True
            assert data["checks"]["tools"] is True

    async def test_all_tools_registered(self, live_app):
        tools = live_app.state.tool_registry.list_tools()
        assert len(tools) == 9
        assert "classify_caller_intent" in tools
        assert "book_consultation" in tools
        assert "escalate_to_human" in tools
        assert "complete_intake" in tools
        assert "record_conflict_info" in tools
        assert "record_additional_info" in tools


class TestFrontend:
    async def test_frontend_served(self):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "html" in resp.headers.get("content-type", "")
