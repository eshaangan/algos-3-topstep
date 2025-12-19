"""Compatibility shim for legacy imports expecting `config` at project root."""

from core import config as _config  # noqa: F401
from core.config import *  # type: ignore  # noqa: F403

__all__ = [name for name in dir(_config) if not name.startswith("_")]
