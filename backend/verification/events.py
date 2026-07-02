from pydantic import BaseModel


class VerificationStarted(BaseModel):
    request_id: str


class VerificationCompleted(BaseModel):
    request_id: str
    score: float


class VerificationFailed(BaseModel):
    request_id: str
    error_type: str
    error: str
