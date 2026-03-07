"""Hugging Face token resolution helpers.

These pure functions return values or None, keeping token logic free
from any CLI/GUI framework dependency.
"""

import os
from pathlib import Path

TOKEN_DIR: Path = Path.home() / ".config" / "surivoice"
TOKEN_FILE: Path = TOKEN_DIR / "token"


def load_saved_token() -> str | None:
    """Read a saved HF token from ~/.config/surivoice/token.

    Returns:
        The token string if found, or None.
    """
    if TOKEN_FILE.is_file():
        content = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if content:
            return content
    return None


def resolve_hf_token(provided_token: str | None) -> str | None:
    """Resolve the HF token from an explicit value, env var, or saved file.

    Resolution order:
        1. Explicitly provided token (CLI flag or GUI field).
        2. ``HF_TOKEN`` environment variable.
        3. Saved token file (``~/.config/surivoice/token``).

    Returns:
        The resolved token string, or None if no token is found.
    """
    if provided_token:
        return provided_token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    return load_saved_token()


def save_token(token: str) -> Path:
    """Save a Hugging Face token to the local config directory.

    Returns:
        The path where the token was saved.
    """
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token.strip(), encoding="utf-8")
    TOKEN_FILE.chmod(0o600)  # read/write for owner only
    return TOKEN_FILE
