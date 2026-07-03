from fastapi.testclient import TestClient

from backend.api.main import create_app


def test_app_boots_and_lists_all_eight_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/wiring.db")
    monkeypatch.setenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "5uL8vG3sVXqQeQ6uKX3nQeYV1o6z5w4C3hK1s0mE6yA=")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/v1/providers/config")

        assert response.status_code == 200
        names = {status["provider"] for status in response.json()}
        assert names == {
            "openai", "anthropic", "ollama",
            "gemini", "nvidia_nim", "openrouter", "groq", "mistral",
        }
