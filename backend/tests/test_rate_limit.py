import pytest
from starlette.requests import Request

from app.security.rate_limit import get_client_ip


def _request(headers: dict[str, str] | None = None, client=("203.0.113.5", 1234)) -> Request:
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": raw_headers,
        "client": client,
    }
    return Request(scope)


def test_get_client_ip_prefers_x_real_ip_header() -> None:
    request = _request({"X-Real-IP": "198.51.100.7"})

    assert get_client_ip(request) == "198.51.100.7"


def test_get_client_ip_strips_whitespace_from_header() -> None:
    request = _request({"X-Real-IP": "  198.51.100.7  "})

    assert get_client_ip(request) == "198.51.100.7"


def test_get_client_ip_falls_back_to_remote_address() -> None:
    request = _request()

    assert get_client_ip(request) == "203.0.113.5"


def test_get_client_ip_ignores_empty_header() -> None:
    request = _request({"X-Real-IP": ""})

    assert get_client_ip(request) == "203.0.113.5"
