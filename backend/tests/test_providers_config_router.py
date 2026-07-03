from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.mock_provider import MockProvider


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "5uL8vG3sVXqQeQ6uKX3nQeYV1o6z5w4C3hK1s0mE6yA=")
    app = create_app()
    client = TestClient(app)
    with client:
        # Force "openai" to build a MockProvider so health checks are
        # deterministic and don't depend on real credentials/network.
        app.state.provider_factory.register("openai", MockProvider)
        yield client


def test_list_provider_config_unconfigured_by_default(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        response = client.get("/v1/providers/config")

        assert response.status_code == 200
        statuses = {s["provider"]: s for s in response.json()}
        assert set(statuses) == {"openai", "anthropic", "ollama"}
        assert statuses["openai"]["configured"] is False
        assert statuses["openai"]["healthy"] is False


def test_unknown_provider_returns_404(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        for method, path in [
            ("post", "/v1/providers/bogus/config"),
            ("delete", "/v1/providers/bogus/config"),
            ("post", "/v1/providers/bogus/enable"),
            ("post", "/v1/providers/bogus/disable"),
            ("post", "/v1/providers/bogus/test"),
        ]:
            kwargs = {"json": {"api_key": "x"}} if method == "post" and ("test" in path or "config" in path) else {}
            response = client.request(method.upper(), path, **kwargs)
            assert response.status_code == 404


def test_save_provider_config_succeeds_and_persists(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        response = client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        assert response.status_code == 200
        body = response.json()
        assert body == {"saved": True, "activated": True, "reason": None}

        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["configured"] is True
        assert listed["openai"]["healthy"] is True
        assert listed["openai"]["masked_key"] is not None


def test_save_provider_config_fails_health_check_does_not_persist(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        # anthropic keeps its real provider class, which fails health_check
        # against a bogus key with no network access.
        response = client.post("/v1/providers/anthropic/config", json={"api_key": "sk-bad-key"})

        assert response.status_code == 200
        body = response.json()
        assert body["saved"] is False
        assert body["activated"] is False
        assert body["reason"] == "health check failed"

        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["anthropic"]["configured"] is False
        assert listed["anthropic"]["last_failure_reason"] == "health check failed"


def test_test_provider_config_does_not_persist(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        response = client.post("/v1/providers/openai/test", json={"api_key": "sk-test-123456"})

        assert response.status_code == 200
        assert response.json() == {"saved": False, "activated": False, "reason": None}

        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["configured"] is False


def test_delete_provider_config_removes_credential(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        response = client.delete("/v1/providers/openai/config")

        assert response.status_code == 200
        assert response.json()["saved"] is True
        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["configured"] is False


def test_disable_then_enable_provider_toggles_availability(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        disabled = client.post("/v1/providers/openai/disable")
        assert disabled.status_code == 200
        assert disabled.json()["activated"] is False
        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["is_enabled"] is False

        enabled = client.post("/v1/providers/openai/enable")
        assert enabled.status_code == 200
        assert enabled.json()["activated"] is True
        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["is_enabled"] is True


def test_test_connection_without_retyping_key_uses_stored_key(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        response = client.post("/v1/providers/openai/test", json={})

        assert response.status_code == 200
        assert response.json() == {"saved": False, "activated": False, "reason": None}


def test_save_without_retyping_key_keeps_existing_credential(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        response = client.post("/v1/providers/openai/config", json={})

        assert response.status_code == 200
        assert response.json() == {"saved": True, "activated": True, "reason": None}
        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["configured"] is True
        assert listed["openai"]["masked_key"] is not None


def test_disable_is_not_reactivated_by_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-fallback")
    for client in _client(tmp_path, monkeypatch):
        client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        disabled = client.post("/v1/providers/openai/disable")

        assert disabled.status_code == 200
        assert disabled.json()["activated"] is False
        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["is_enabled"] is False


def test_delete_is_not_reactivated_by_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-fallback")
    for client in _client(tmp_path, monkeypatch):
        client.post("/v1/providers/openai/config", json={"api_key": "sk-test-123456"})

        deleted = client.delete("/v1/providers/openai/config")

        assert deleted.status_code == 200
        assert deleted.json()["activated"] is False
        listed = {s["provider"]: s for s in client.get("/v1/providers/config").json()}
        assert listed["openai"]["configured"] is False


def test_new_provider_is_known_once_registered_in_factory(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        # "gemini" is not registered by _build_provider_factory in this
        # branch snapshot's fixture yet -- registering it directly on the
        # already-built factory (as the test/enable/disable/delete routes
        # consult it live via provider_factory.registered_names()) must be
        # enough to make the router treat it as a known provider, with no
        # provider-specific logic needed in the router itself.
        client.app.state.provider_factory.register("gemini", GeminiProvider)

        response = client.post("/v1/providers/gemini/test", json={"api_key": "bogus"})

        assert response.status_code == 200
        assert response.json()["reason"] == "health check failed"


def test_providers_page_renders(tmp_path, monkeypatch):
    for client in _client(tmp_path, monkeypatch):
        response = client.get("/dashboard/providers")

        assert response.status_code == 200
        assert "openai" in response.text
        assert "anthropic" in response.text
        assert "ollama" in response.text
