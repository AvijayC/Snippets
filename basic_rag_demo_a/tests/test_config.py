from rag_demo.config import AgentSettings, ApiSettings, DatabaseSettings


def test_system_instruction_takes_precedence_over_legacy_prompt() -> None:
    settings = AgentSettings(
        name="main",
        system_instruction="new instruction",
        system_prompt="legacy prompt",
    )

    assert settings.effective_system_instruction() == "new instruction"


def test_system_prompt_is_backward_compatible() -> None:
    settings = AgentSettings(name="main", system_prompt="legacy prompt")

    assert settings.effective_system_instruction() == "legacy prompt"


def test_agent_reasoning_defaults_to_high() -> None:
    settings = AgentSettings(name="main")

    assert settings.reasoning_effort == "high"


def test_api_key_env_name_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("MY_RAG_API_KEY", "secret")
    settings = ApiSettings(api_key_env="MY_RAG_API_KEY", base_url="https://example.test/v1")

    assert settings.resolved_api_key() == "secret"
    assert settings.api_key_source() == "env:MY_RAG_API_KEY"


def test_database_path_accepts_legacy_key() -> None:
    settings = DatabaseSettings.model_validate({"sqlite_path": "data/custom.db"})

    assert settings.path == "data/custom.db"
