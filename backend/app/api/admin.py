"""Admin API for the self-improving loop."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _verify_api_key(request: Request, x_api_key: Annotated[str, Header()] = "") -> None:
    expected = request.app.state.settings.admin_api_key
    if not expected:
        raise HTTPException(403, "Admin API key not configured")
    if x_api_key != expected:
        raise HTTPException(401, "Invalid API key")


@router.get("/evaluations")
async def list_evaluations(
    request: Request,
    min_score: float | None = None,
    max_score: float | None = None,
    limit: int = 50,
    _auth: None = Depends(_verify_api_key),
):
    evaluator = request.app.state.evaluator
    if evaluator is None:
        raise HTTPException(503, "Evaluator not enabled")
    return await evaluator.get_evaluations(min_score=min_score, max_score=max_score, limit=limit)


@router.get("/evaluations/{call_id}")
async def get_evaluation(
    request: Request,
    call_id: str,
    _auth: None = Depends(_verify_api_key),
):
    evaluator = request.app.state.evaluator
    if evaluator is None:
        raise HTTPException(503, "Evaluator not enabled")
    results = await evaluator.get_evaluations()
    for r in results:
        if r.get("call_id") == call_id:
            return r
    raise HTTPException(404, "Evaluation not found")


@router.get("/stats")
async def get_stats(
    request: Request,
    _auth: None = Depends(_verify_api_key),
):
    evaluator = request.app.state.evaluator
    if evaluator is None:
        raise HTTPException(503, "Evaluator not enabled")
    return await evaluator.get_stats()


@router.post("/analyze")
async def trigger_analysis(
    request: Request,
    min_calls: int = 5,
    score_threshold: float = 6.0,
    _auth: None = Depends(_verify_api_key),
):
    optimizer = request.app.state.prompt_optimizer
    if optimizer is None:
        raise HTTPException(503, "Evaluator not enabled")
    return await optimizer.analyze(min_calls=min_calls, score_threshold=score_threshold)


@router.get("/prompt-versions")
async def list_prompt_versions(
    request: Request,
    _auth: None = Depends(_verify_api_key),
):
    optimizer = request.app.state.prompt_optimizer
    if optimizer is None:
        raise HTTPException(503, "Evaluator not enabled")
    return await optimizer.list_versions()


@router.post("/prompt-versions/{version}/approve")
async def approve_prompt_version(
    request: Request,
    version: str,
    _auth: None = Depends(_verify_api_key),
):
    optimizer = request.app.state.prompt_optimizer
    if optimizer is None:
        raise HTTPException(503, "Evaluator not enabled")
    approved = await optimizer.approve_version(version)
    if not approved:
        raise HTTPException(404, "Version not found or already approved")
    return {"status": "approved", "version": version}
