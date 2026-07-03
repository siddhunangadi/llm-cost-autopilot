from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from backend.api.dependencies import AnalyticsServiceDep, AppVersionDep
from backend.api.paths import TEMPLATES_DIR
from backend.services.analytics_service import AnalyticsReport
from backend.services.dashboard_repository import TimeWindow

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


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
        "app_version": app_version,
        "now": _now_str(),
    })
