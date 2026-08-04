from functools import lru_cache
from pathlib import Path
from typing import Any
import os
import tomllib

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://localhost:5173", "http://localhost:5174")


class BackendSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    log_level: str = "INFO"
    backend_cors_origins: list[str] = Field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    backend_basic_auth_username: str | None = None
    backend_basic_auth_password: str | None = None
    model_artifact_path: str | None = None
    features_config_path: str | None = None

    @field_validator(
        "log_level",
        "backend_basic_auth_username",
        "backend_basic_auth_password",
        "model_artifact_path",
        "features_config_path",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("must be a string.")
        stripped = value.strip()
        return stripped or None

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def _normalize_origins(cls, value: Any) -> list[str]:
        if value is None:
            return list(DEFAULT_CORS_ORIGINS)
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        raise ValueError("must be a list of strings or a comma-separated string.")


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("Configuration values must be strings.")
    stripped = value.strip()
    return stripped or None


def _load_config_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_config = tomllib.load(handle)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"Invalid backend config file: {path}.")
    return raw_config.get("backend", raw_config)


def _configured_backend_config_path() -> str | None:
    return _coerce_text(os.getenv("BACKEND_CONFIG_FILE"))


def _load_backend_settings() -> BackendSettings:
    values: dict[str, Any] = {}

    configured_path = _configured_backend_config_path()
    if configured_path is not None:
        config_path = Path(configured_path)
        if not config_path.exists():
            raise RuntimeError(f"BACKEND_CONFIG_FILE is set but file does not exist: {configured_path}.")
        if not config_path.is_file():
            raise RuntimeError(f"BACKEND_CONFIG_FILE must point to a file: {configured_path}.")
        values.update(_load_config_file(config_path))

    env_overrides = {
        "log_level": os.getenv("LOG_LEVEL"),
        "backend_cors_origins": os.getenv("BACKEND_CORS_ORIGINS"),
        "backend_basic_auth_username": os.getenv("BACKEND_BASIC_AUTH_USERNAME"),
        "backend_basic_auth_password": os.getenv("BACKEND_BASIC_AUTH_PASSWORD"),
        "model_artifact_path": os.getenv("MODEL_ARTIFACT_PATH"),
        "features_config_path": os.getenv("FEATURES_CONFIG_PATH"),
    }
    for key, raw_value in env_overrides.items():
        if key == "backend_cors_origins" and raw_value is not None:
            values[key] = raw_value
            continue
        coerced_value = _coerce_text(raw_value)
        if coerced_value is not None:
            values[key] = coerced_value

    return BackendSettings(**values)


@lru_cache(maxsize=1)
def get_backend_settings() -> BackendSettings:
    return _load_backend_settings()


def _reset_backend_settings_for_tests() -> None:
    get_backend_settings.cache_clear()
