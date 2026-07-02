from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.api.dependencies import ChatServiceDep
from backend.chat.service import ChatResult
from backend.providers.base import ProviderError
from backend.routing.engine import NoEligibleModelError

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str
    strategy: Literal["cost", "latency", "quality", "balanced"] = "balanced"


@router.post("/chat", response_model=ChatResult)
async def chat(
    request: ChatRequest, chat_service: ChatServiceDep, background_tasks: BackgroundTasks
) -> ChatResult:
    try:
        return await chat_service.chat(
            request.prompt, strategy=request.strategy, background_tasks=background_tasks
        )
    except NoEligibleModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
