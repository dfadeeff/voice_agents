"""Smoke tests: does the FastAPI app start, do endpoints respond?"""

from app.main import create_app
from httpx import ASGITransport, AsyncClient


class TestHealth:
    async def test_health_returns_ok(self):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestReadyWithLifespan:
    async def test_ready_checks_pass(self):
        app = create_app()
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/ready")
                assert resp.status_code == 200
                data = resp.json()
                assert data["ready"] is True
                assert data["checks"]["database"] is True
                assert data["checks"]["tools"] is True

    async def test_all_six_tools_registered(self):
        app = create_app()
        async with app.router.lifespan_context(app):
            tools = app.state.tool_registry.list_tools()
            assert len(tools) == 6
            assert "classify_caller_intent" in tools
            assert "book_consultation" in tools
            assert "escalate_to_human" in tools


class TestFrontend:
    async def test_frontend_served(self):
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "html" in resp.headers.get("content-type", "")
