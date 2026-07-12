from pydantic import BaseModel, Field


class VerificationConfig(BaseModel):
    judge_model_id: str
    pass_threshold: float = Field(ge=0.0, le=1.0)
    judge_prompt_version: str
    escalation_model_id: str
