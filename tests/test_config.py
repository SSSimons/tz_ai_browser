from browser_agent.config import Settings


def test_openai_base_url_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    settings = Settings(_env_file=None)
    assert settings.openai_base_url == "https://proxy.example/v1"
