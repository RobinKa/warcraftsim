"""Filesystem locations used by warcraftsim.

Every location can be overridden with an environment variable so that tests and
parallel runs can point at alternative installs, caches or prefixes.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Read-only master copy of the Legacy TFT 1.29 install (copied from the Windows side).
GAME_DIR = _env_path("WC3_GAME_DIR", Path.home() / "wc3" / "legacy-1.29")

# Extracted game data (scripts, SLKs, stock maps). Never committed.
CACHE_DIR = _env_path("WARCRAFTSIM_CACHE", Path.home() / ".cache" / "warcraftsim")

# Wine prefixes, per-instance state and run artifacts.
RUNTIME_DIR = _env_path("WARCRAFTSIM_RUNTIME", Path.home() / "wc3" / "runtime")

STORMLIB_PATH = _env_path("WARCRAFTSIM_STORMLIB", REPO_ROOT / "build" / "stormlib" / "libstorm.so")
PJASS_PATH = _env_path("WARCRAFTSIM_PJASS", REPO_ROOT / "build" / "pjass" / "pjass")

# Base game archives in lookup priority order (first hit wins), matching the game's own order.
GAME_ARCHIVES = ("War3xLocal.mpq", "War3x.mpq", "War3Local.mpq", "War3.mpq", "Deprecated.mpq")
