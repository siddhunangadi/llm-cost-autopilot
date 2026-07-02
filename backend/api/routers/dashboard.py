from fastapi import APIRouter

from backend.api.dependencies import DashboardServiceDep
from backend.services.dashboard_repository import TimeWindow
from backend.services.dashboard_service import DashboardOverview

router = APIRouter()


@router.get("/dashboard/overview", response_model=DashboardOverview)
async def get_dashboard_overview(
    dashboard_service: DashboardServiceDep, days: int = 7,
) -> DashboardOverview:
    return await dashboard_service.get_overview(TimeWindow(days=days))
