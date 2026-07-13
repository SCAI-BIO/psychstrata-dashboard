import os
import secrets
from urllib.parse import urlparse

from flask import redirect, render_template_string, request, session, url_for


AUTH_PASSWORD_ENV = "APP_PASSWORD"
SESSION_SECRET_ENV = "APP_SESSION_SECRET"
SESSION_AUTH_KEY = "authenticated"

LOGIN_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Dashboard Login</title>
    <style>
      body {
        margin: 0;
        font-family: Inter, Arial, sans-serif;
        background: #f6f7fb;
        color: #111827;
      }
      .shell {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
      }
      .card {
        width: 100%;
        max-width: 420px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        padding: 24px;
      }
      h1 {
        margin: 0 0 8px;
        font-size: 24px;
      }
      p {
        margin: 0 0 16px;
        color: #6b7280;
        line-height: 1.5;
      }
      label {
        display: block;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
      }
      input[type="password"] {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 14px;
        margin-bottom: 16px;
      }
      button {
        width: 100%;
        border: 0;
        border-radius: 8px;
        padding: 10px 14px;
        background: #006fb9;
        color: white;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
      }
      .error {
        margin-bottom: 16px;
        padding: 10px 12px;
        border-radius: 8px;
        background: #fef2f2;
        color: #991b1b;
        font-size: 14px;
      }
      .hint {
        margin-top: 14px;
        font-size: 12px;
        color: #6b7280;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="card">
        <h1>Dashboard Login</h1>
        <p>Enter the shared password to access the dashboard.</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="post">
          <label for="password">Password</label>
          <input id="password" name="password" type="password" autocomplete="current-password" autofocus required>
          <button type="submit">Sign in</button>
        </form>
        <div class="hint">The password is configured through the container environment.</div>
      </div>
    </div>
  </body>
</html>
"""


def _login_required() -> bool:
    return not bool(session.get(SESSION_AUTH_KEY))


def _is_public_path(path: str) -> bool:
    public_prefixes = (
        "/login",
        "/logout",
        "/api/",
    )
    public_exact = {
        "/login",
        "/logout",
        "/favicon.ico",
    }
    return path in public_exact or any(path.startswith(prefix) for prefix in public_prefixes)


def _safe_next_url(target: str | None) -> str:
    if not target:
        return "/"
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return "/"
    if not target.startswith("/"):
        return "/"
    if target.startswith("//"):
        return "/"
    return target


def configure_auth(server):
    server.secret_key = os.getenv(SESSION_SECRET_ENV) or secrets.token_hex(32)
    server.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    @server.before_request
    def require_login():
        if _is_public_path(request.path):
            return None
        if _login_required():
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("login", next=next_url))
        return None

    @server.route("/login", methods=["GET", "POST"])
    def login():
        configured_password = os.getenv(AUTH_PASSWORD_ENV)
        next_url = _safe_next_url(request.args.get("next"))

        if session.get(SESSION_AUTH_KEY):
            return redirect(next_url)

        if request.method == "POST":
            if not configured_password:
                return (
                    render_template_string(
                        LOGIN_TEMPLATE,
                        error="Login is unavailable because no dashboard password is configured.",
                    ),
                    503,
                )

            submitted_password = request.form.get("password", "")
            if secrets.compare_digest(submitted_password, configured_password):
                session[SESSION_AUTH_KEY] = True
                return redirect(next_url)

            return render_template_string(LOGIN_TEMPLATE, error="Incorrect password."), 401

        if not configured_password:
            return (
                render_template_string(
                    LOGIN_TEMPLATE,
                    error="Login is unavailable because no dashboard password is configured.",
                ),
                503,
            )

        return render_template_string(LOGIN_TEMPLATE, error=None)

    @server.route("/logout", methods=["POST", "GET"])
    def logout():
        session.pop(SESSION_AUTH_KEY, None)
        return redirect(url_for("login"))
