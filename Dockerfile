FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend ./backend

ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    DATABASE_URL=sqlite:///./llm_cost_autopilot.db \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uv run --no-sync uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT}"]
