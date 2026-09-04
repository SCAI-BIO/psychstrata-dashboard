from pathlib import Path

import pytest

from app import settings


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in (
        "BACKEND_CONFIG_FILE",
        "LOG_LEVEL",
        "BACKEND_CORS_ORIGINS",
        "BACKEND_BASIC_AUTH_USERNAME",
        "BACKEND_BASIC_AUTH_PASSWORD",
        "BACKEND_DATABASE_URL",
        "MODEL_ARTIFACT_PATH",
        "FEATURES_CONFIG_PATH",
    ):
        monkeypatch.delenv(env_name, raising=False)
    settings._reset_backend_settings_for_tests()
    yield
    settings._reset_backend_settings_for_tests()


def test_settings_use_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("BACKEND_CORS_ORIGINS", "https://one.example, https://two.example")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_USERNAME", "dashboard-user")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_PASSWORD", "dashboard-password")
    monkeypatch.setenv("BACKEND_DATABASE_URL", "sqlite:///./env.sqlite3")
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "/models/model.pkl")
    monkeypatch.setenv("FEATURES_CONFIG_PATH", "/configs/features.json")

    backend_settings = settings.get_backend_settings()

    assert backend_settings.log_level == "DEBUG"
    assert backend_settings.backend_cors_origins == ["https://one.example", "https://two.example"]
    assert backend_settings.backend_basic_auth_username == "dashboard-user"
    assert backend_settings.backend_basic_auth_password == "dashboard-password"
    assert backend_settings.backend_database_url == "sqlite:///./env.sqlite3"
    assert backend_settings.model_artifact_path == "/models/model.pkl"
    assert backend_settings.features_config_path == "/configs/features.json"


def test_settings_load_values_from_toml_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "backend.toml"
    config_path.write_text(
        "\n".join(
            [
                'log_level = "WARNING"',
                'backend_cors_origins = ["https://file.example"]',
                'backend_basic_auth_username = "file-user"',
                'backend_basic_auth_password = "file-password"',
                'backend_database_url = "sqlite:///./file.sqlite3"',
                'model_artifact_path = "/file/model.pkl"',
                'features_config_path = "/file/features.json"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKEND_CONFIG_FILE", str(config_path))

    backend_settings = settings.get_backend_settings()

    assert backend_settings.log_level == "WARNING"
    assert backend_settings.backend_cors_origins == ["https://file.example"]
    assert backend_settings.backend_basic_auth_username == "file-user"
    assert backend_settings.backend_basic_auth_password == "file-password"
    assert backend_settings.backend_database_url == "sqlite:///./file.sqlite3"
    assert backend_settings.model_artifact_path == "/file/model.pkl"
    assert backend_settings.features_config_path == "/file/features.json"


def test_settings_env_overrides_file_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "backend.toml"
    config_path.write_text(
        "\n".join(
            [
                'log_level = "WARNING"',
                'backend_cors_origins = ["https://file.example"]',
                'backend_basic_auth_username = "file-user"',
                'backend_basic_auth_password = "file-password"',
                'backend_database_url = "sqlite:///./file.sqlite3"',
                'model_artifact_path = "/file/model.pkl"',
                'features_config_path = "/file/features.json"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKEND_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("BACKEND_CORS_ORIGINS", "https://env.example")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_USERNAME", "env-user")
    monkeypatch.setenv("BACKEND_BASIC_AUTH_PASSWORD", "env-password")
    monkeypatch.setenv("BACKEND_DATABASE_URL", "sqlite:///./env.sqlite3")
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "/env/model.pkl")
    monkeypatch.setenv("FEATURES_CONFIG_PATH", "/env/features.json")

    backend_settings = settings.get_backend_settings()

    assert backend_settings.log_level == "DEBUG"
    assert backend_settings.backend_cors_origins == ["https://env.example"]
    assert backend_settings.backend_basic_auth_username == "env-user"
    assert backend_settings.backend_basic_auth_password == "env-password"
    assert backend_settings.backend_database_url == "sqlite:///./env.sqlite3"
    assert backend_settings.model_artifact_path == "/env/model.pkl"
    assert backend_settings.features_config_path == "/env/features.json"
