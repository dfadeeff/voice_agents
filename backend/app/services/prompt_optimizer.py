"""Analyzes low-scoring calls and suggests prompt improvements."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

import aiosqlite

logger = logging.getLogger(__name__)

OPTIMIZER_PROMPT = """\
You are a prompt engineering expert analyzing voice AI call evaluations.

Below are findings and improvement suggestions from {count} calls that scored below {threshold}/10.

AGGREGATED FINDINGS:
{findings}

AGGREGATED SUGGESTIONS:
{suggestions}

CURRENT SYSTEM PROMPT PREAMBLE:
{current_preamble}

Based on these patterns, suggest specific, actionable modifications to the system prompt \
that would improve call quality. Focus on the most impactful changes.

Respond with ONLY valid JSON:
{{
  "changes": [
    {{
      "section": "preamble|phase_prompt|fragment",
      "target": "<specific section or phase name>",
      "current_text_snippet": "<relevant current text>",
      "suggested_text": "<improved text>",
      "rationale": "<why this change helps>"
    }}
  ],
  "summary": "<one-paragraph summary of all changes>"
}}
"""


class PromptOptimizer:
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

    async def analyze(self, min_calls: int = 5, score_threshold: float = 6.0) -> dict:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM call_evaluations
                   WHERE overall_score < ? ORDER BY evaluated_at DESC LIMIT 50""",
                [score_threshold],
            ) as cursor:
                rows = await cursor.fetchall()
                low_scoring = [dict(r) for r in rows]

        if len(low_scoring) < min_calls:
            return {
                "status": "insufficient_data",
                "message": (
                    f"Need at least {min_calls} low-scoring calls, " f"found {len(low_scoring)}."
                ),
            }

        all_findings = []
        all_suggestions = []
        for row in low_scoring:
            try:
                all_findings.extend(json.loads(row.get("findings", "[]")))
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                all_suggestions.extend(json.loads(row.get("improvement_suggestions", "[]")))
            except (json.JSONDecodeError, TypeError):
                pass

        from app.conversation.locales import get_locale

        current_preamble = get_locale("de").PREAMBLE

        prompt = OPTIMIZER_PROMPT.format(
            count=len(low_scoring),
            threshold=score_threshold,
            findings="\n".join(f"- {f}" for f in all_findings[:30]),
            suggestions="\n".join(f"- {s}" for s in all_suggestions[:20]),
            current_preamble=current_preamble,
        )

        try:
            result = await self._call_llm(prompt)
        except Exception:
            logger.exception("Optimizer LLM call failed")
            return {"status": "error", "message": "LLM call failed"}

        if result is None:
            return {"status": "error", "message": "Could not parse LLM response"}

        version_id = f"v-{uuid.uuid4().hex[:8]}"
        await self._save_draft(version_id, result, current_preamble)

        return {
            "status": "draft_created",
            "version": version_id,
            "changes": result.get("changes", []),
            "summary": result.get("summary", ""),
            "based_on_calls": len(low_scoring),
        }

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
            "temperature": 0.3,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]

        return json.loads(content)

    async def _save_draft(self, version: str, result: dict, current_preamble: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO prompt_versions
                   (version, prompts_snapshot, change_description, status, created_at)
                   VALUES (?, ?, ?, 'draft', ?)""",
                [
                    version,
                    json.dumps(
                        {
                            "preamble": current_preamble,
                            "proposed_changes": result.get("changes", []),
                        }
                    ),
                    result.get("summary", ""),
                    datetime.now(UTC).isoformat(),
                ],
            )
            await db.commit()

    async def list_versions(self) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM prompt_versions ORDER BY created_at DESC"
            ) as cursor:
                return [dict(r) for r in await cursor.fetchall()]

    async def approve_version(self, version: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "UPDATE prompt_versions SET status = 'approved' "
                "WHERE version = ? AND status = 'draft'",
                [version],
            )
            await db.commit()
            return cursor.rowcount > 0
