"""Post-call quality evaluation using an LLM judge."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import aiosqlite

logger = logging.getLogger(__name__)

EVAL_PROMPT = """\
You are evaluating a law firm intake call handled by an AI receptionist.

Score each dimension from 1-10:
1. **completeness**: Did the agent collect all required information? (name, email, phone, \
legal area, employer, insurance, issue details)
2. **naturalness**: Did the conversation flow naturally? Was the agent warm and professional?
3. **accuracy**: Were entities captured correctly? Did the agent confirm ambiguous details?
4. **escalation_appropriateness**: If escalation occurred, was it warranted? If not, should it have?
5. **overall**: Overall quality of the call experience.

Also provide:
- **findings**: List of specific issues or good practices observed (max 5 items).
- **improvement_suggestions**: Actionable prompt improvements (max 3 items).

TRANSCRIPT:
{transcript}

STATE SNAPSHOT:
{state}

Respond with ONLY valid JSON matching this schema:
{{
  "completeness_score": <1-10>,
  "naturalness_score": <1-10>,
  "accuracy_score": <1-10>,
  "escalation_appropriateness": <1-10>,
  "overall_score": <1-10>,
  "findings": ["..."],
  "improvement_suggestions": ["..."]
}}
"""


class CallEvaluator:
    def __init__(
        self,
        db_path: str,
        provider: str = "ollama",
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
    ):
        self._db_path = db_path
        self._provider = provider
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

    async def evaluate(
        self, call_id: str, transcript: list[dict], state_snapshot: dict
    ) -> dict | None:
        transcript_text = "\n".join(
            f"[{e.get('role', '?')}] {e.get('text', e.get('content', ''))}" for e in transcript
        )
        state_text = json.dumps(state_snapshot, indent=2, default=str)
        prompt = EVAL_PROMPT.format(transcript=transcript_text, state=state_text)

        try:
            result = await self._call_llm(prompt)
        except Exception:
            logger.exception("Evaluation LLM call failed for %s", call_id)
            return None

        if result is None:
            return None

        await self._store(call_id, result)
        return result

    async def _call_llm(self, prompt: str) -> dict | None:
        import httpx

        if self._provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self._api_key}"}
        else:
            url = f"{self._base_url}/v1/chat/completions"
            headers = {}

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        return json.loads(content)

    async def _store(self, call_id: str, result: dict) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO call_evaluations
                   (call_id, evaluator_model, completeness_score, naturalness_score,
                    accuracy_score, escalation_appropriateness, overall_score,
                    findings, improvement_suggestions, evaluated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    call_id,
                    self._model,
                    result.get("completeness_score"),
                    result.get("naturalness_score"),
                    result.get("accuracy_score"),
                    result.get("escalation_appropriateness"),
                    result.get("overall_score"),
                    json.dumps(result.get("findings", [])),
                    json.dumps(result.get("improvement_suggestions", [])),
                    datetime.now(UTC).isoformat(),
                ],
            )
            await db.commit()
        logger.info(
            "Evaluation stored for %s: overall=%.1f",
            call_id,
            result.get("overall_score", 0),
        )

    async def get_evaluations(
        self, min_score: float | None = None, max_score: float | None = None, limit: int = 50
    ) -> list[dict]:
        query = "SELECT * FROM call_evaluations"
        conditions = []
        params: list = []
        if min_score is not None:
            conditions.append("overall_score >= ?")
            params.append(min_score)
        if max_score is not None:
            conditions.append("overall_score <= ?")
            params.append(max_score)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY evaluated_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_stats(self) -> dict:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """SELECT COUNT(*) as total,
                          AVG(overall_score) as avg_overall,
                          AVG(completeness_score) as avg_completeness,
                          AVG(naturalness_score) as avg_naturalness,
                          AVG(accuracy_score) as avg_accuracy
                   FROM call_evaluations"""
            ) as cursor:
                row = await cursor.fetchone()
                if not row or row[0] == 0:
                    return {"total": 0}
                return {
                    "total": row[0],
                    "avg_overall": round(row[1], 2) if row[1] else None,
                    "avg_completeness": round(row[2], 2) if row[2] else None,
                    "avg_naturalness": round(row[3], 2) if row[3] else None,
                    "avg_accuracy": round(row[4], 2) if row[4] else None,
                }
