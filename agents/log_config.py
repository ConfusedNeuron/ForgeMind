# agents/log_config.py
# ═══════════════════════════════════════════════════════════════════════
# Centralised logging for the ForgeMind agent pipeline.
#
# All pipeline modules import `get_logger()` from here.
# Logs go to a FILE so they're visible in a separate terminal
# while the Textual TUI owns the main terminal.
#
# Usage (in any agent module):
#     from .log_config import get_logger
#     log = get_logger("diagnostic")   # → "forgemind.diagnostic"
#
# To watch logs live, open a second terminal and run:
#     type forgemind.log            (CMD  — one-shot)
#     Get-Content forgemind.log -Wait   (PowerShell — live tail)
# ═══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from pathlib import Path

# Log file lives in the project root
LOG_FILE = Path(__file__).resolve().parent.parent / "forgemind.log"

# ── One-time setup of the root "forgemind" logger ─────────────────────────────
_root = logging.getLogger("forgemind")

if not _root.handlers:
    # File handler — the ONLY output channel (keeps the TUI terminal clean)
    _fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    _root.addHandler(_fh)
    _root.setLevel(logging.DEBUG)

    # Suppress propagation to root logger (avoids duplicate lines)
    _root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'forgemind' namespace.

    Example:
        get_logger("diagnostic")  →  logging.getLogger("forgemind.diagnostic")
    """
    return logging.getLogger(f"forgemind.{name}")
