"""Central place for API keys and other environment-provided secrets.

Real values live in a git-ignored ``.env`` file at the repo root (copy
``.env.example`` to ``.env`` and fill it in -- never commit ``.env``).
Every module that needs a key should go through here instead of calling
``os.environ.get(...)`` directly, so there's one place that knows what
secrets the project uses and one place to update if a key's source changes
(e.g. moving from an env var to a secrets manager later).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    # Repo root is two levels up from this file (src/finmodels/config.py).
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    _ENV_LOADED = True


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not configured."""


def get_api_key(name: str, *, required: bool = True) -> str | None:
    """Look up an API key by environment variable name (e.g. ``"FRED_API_KEY"``).

    Loads ``.env`` on first use. Returns ``None`` if unset and
    ``required=False``; raises :class:`MissingAPIKeyError` if unset and
    ``required=True`` (the default).
    """
    _ensure_env_loaded()
    value = os.environ.get(name)
    if not value:
        if required:
            raise MissingAPIKeyError(
                f"{name} is not set. Copy .env.example to .env and fill in "
                f"a value, or export {name} in your shell."
            )
        return None
    return value
