from backend.api.main import create_app


def test_create_app_registers_health_route():
    app = create_app()
    # app.routes exposes included routers lazily in this FastAPI version
    # (fastapi.routing._IncludedRouter, not flattened APIRoute objects),
    # so check via the generated OpenAPI schema instead -- a stable public
    # API that forces full route resolution.
    schema = app.openapi()
    assert "/v1/health" in schema["paths"]
