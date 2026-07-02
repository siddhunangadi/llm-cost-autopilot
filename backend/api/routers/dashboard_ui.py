from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates

from backend.api.dependencies import AppVersionDep, DashboardServiceDep
from backend.api.paths import TEMPLATES_DIR
from backend.services.dashboard_repository import TimeWindow

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

FRAGMENT_TEMPLATES = {
    "overview": "fragments/overview.html",
    "providers": "fragments/providers.html",
    "circuits": "fragments/circuits.html",
    "recent-requests": "fragments/recent_requests.html",
}


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@router.get("/dashboard")
async def dashboard_page(
    request: Request, dashboard_service: DashboardServiceDep, app_version: AppVersionDep, days: int = 7,
):
    data = await dashboard_service.get_dashboard_page(TimeWindow(days=days))
    return templates.TemplateResponse(request, "dashboard.html", {
        **data,
        "app_version": app_version,
        "now": _now_str(),
        "standalone": False,
    })


@router.get("/dashboard/fragments/{section}")
async def dashboard_fragment(
    section: str, request: Request, dashboard_service: DashboardServiceDep, days: int = 7,
):
    if section not in FRAGMENT_TEMPLATES:
        raise HTTPException(status_code=404, detail=f"Unknown dashboard section: {section}")

    window = TimeWindow(days=days)
    if section == "overview":
        data = {"overview": await dashboard_service.get_overview_fragment(window)}
    elif section == "providers":
        data = await dashboard_service.get_provider_fragment()
    elif section == "circuits":
        data = await dashboard_service.get_circuit_fragment()
    else:
        data = await dashboard_service.get_recent_requests_fragment()

    return templates.TemplateResponse(request, FRAGMENT_TEMPLATES[section], {
        **data,
        "now": _now_str(),
        "standalone": True,
    })
