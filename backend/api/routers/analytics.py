from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from backend.api.dependencies import AnalyticsServiceDep, AppVersionDep
from backend.api.paths import TEMPLATES_DIR
from backend.services.analytics_service import AnalyticsReport, RoutingDistributionPoint
from backend.services.dashboard_repository import TimeWindow

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _routing_chart(points: list[RoutingDistributionPoint]) -> dict:
    # Chart.js maps a dataset's `data` array to `labels` positionally by
    # index, so every model's series must be aligned to the same full set
    # of dates (zero-filled), not just the dates that model appears on --
    # otherwise bars silently plot against the wrong day whenever model
    # usage isn't uniform across the window.
    labels = sorted({p.date for p in points})
    models = sorted({p.model for p in points})
    counts = {(p.date, p.model): p.request_count for p in points}
    return {
        "labels": [str(d) for d in labels],
        "datasets": [
            {"label": model, "data": [counts.get((day, model), 0) for day in labels]}
            for model in models
        ],
    }


@router.get("/v1/analytics/report", response_model=AnalyticsReport)
async def get_analytics_report(
    analytics_service: AnalyticsServiceDep, days: int = 30,
) -> AnalyticsReport:
    return await analytics_service.get_report(TimeWindow(days=days))


@router.get("/dashboard/analytics")
async def analytics_page(
    request: Request, analytics_service: AnalyticsServiceDep, app_version: AppVersionDep, days: int = 30,
):
    report = await analytics_service.get_report(TimeWindow(days=days))
    return templates.TemplateResponse(request, "analytics.html", {
        "report": report,
        "routing_chart": _routing_chart(report.routing_distribution),
        "app_version": app_version,
        "now": _now_str(),
    })
