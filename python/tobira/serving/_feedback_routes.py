"""Dashboard feedback routes.

Separated from ``dashboard.py`` to avoid ``from __future__ import annotations``
which prevents FastAPI from resolving request body type hints at runtime.
"""

from typing import Any

from tobira.serving.dashboard import _compute_feedback_stats
from tobira.serving.schemas import (
    DashboardFeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
)


def register_feedback_routes(app: Any, feedback_path: str) -> None:
    """Register feedback-related dashboard endpoints.

    Args:
        app: A FastAPI application instance.
        feedback_path: Path to the feedback JSONL file.
    """
    from tobira.data.feedback_store import store_feedback

    @app.post("/api/feedback", response_model=FeedbackResponse, tags=["dashboard"])
    async def dashboard_feedback(req: DashboardFeedbackRequest) -> FeedbackResponse:
        record = store_feedback(
            req.text,
            req.correct_label,
            req.source,
            path=feedback_path,
        )
        return FeedbackResponse(status="accepted", id=record.id)

    @app.get(
        "/api/feedback/stats",
        response_model=FeedbackStatsResponse,
        tags=["dashboard"],
    )
    async def feedback_stats() -> FeedbackStatsResponse:
        stats = _compute_feedback_stats(feedback_path)
        return FeedbackStatsResponse(**stats)
