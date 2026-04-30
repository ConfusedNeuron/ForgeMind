from __future__ import annotations

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from .prompts import FLOOR_MANAGER_SYSTEM_PROMPT
from .log_config import get_logger

# Load env — same pattern as diagnostic_agent.py
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("GEMINI_API_KEY_FLOOR_MANAGER")
if not api_key:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY_FLOOR_MANAGER not found in environment. "
        "Floor manager will use template fallback.",
        stacklevel=2,
    )
    client = None
else:
    # print(f"Loaded API key for FLOOR MANAGER agent: {bool(api_key)}")  # commented — noisy before TUI
    client = genai.Client(api_key=api_key)

log = get_logger("floor_manager")

MAX_RETRIES = 1   # One retry — if Gemini fails prose twice, fallback is equally good


# ── Output validation ─────────────────────────────────────────────────────────

def _validate_output(text: str, capacity_report: dict) -> tuple[bool, str]:
    """
    Soft structural checks on Agent 3's plain-text output.
    Does NOT verify number accuracy — trusts the system prompt for that.
    """
    text = text.strip()

    if not text:
        return False, "Empty response from Gemini"

    if not text.startswith("[Floor Manager]"):
        return False, "Response does not begin with '[Floor Manager]'"

    if len(text) > 800:
        return False, f"Response too long: {len(text)} chars (max 800)"

    machine_name = capacity_report.get("machine_name", "")
    if machine_name and machine_name.lower() not in text.lower():
        return False, f"Machine name '{machine_name}' absent from response"

    return True, ""


# ── Template fallback ─────────────────────────────────────────────────────────

def _template_fallback(report: dict) -> str:
    """
    Hardcoded template used when all Gemini attempts fail.
    Uses live numbers from capacity_report — never hardcoded values.
    Three branches: OFFLINE / DEGRADED / ONLINE.
    """
    mid  = report["machine_id"]
    name = report["machine_name"]
    st   = report["status"]
    rul  = report["rul"]
    cap  = report["capacity_pct"]
    req  = report["machine_req"]
    risk = report["breakeven_risk"]

    if st == "OFFLINE":
        overtime = (
            f" ΣPD/T at {req} — authorize overtime, breakeven risk is ACTIVE."
            if risk else
            f" ΣPD/T at {req} — monitor capacity closely."
        )
        return (
            f"[Floor Manager] {name} (Machine {mid}) is OFFLINE at RUL {rul} — "
            f"mandatory shutdown initiated. "
            f"Halt all production on this unit and dispatch maintenance crew immediately. "
            f"Reroute {name} workload to remaining online machines."
            f" Factory at {cap}% capacity.{overtime}"
        )

    elif st == "DEGRADED":
        return (
            f"[Floor Manager] {name} (Machine {mid}) entering DEGRADED status at RUL {rul}. "
            f"Reduce to 50% load — do not schedule additional jobs on this unit. "
            f"Open a maintenance window within the next shift cycle. "
            f"Factory at {cap}% capacity, ΣPD/T at {req}."
        )

    else:  # ONLINE
        return (
            f"[Floor Manager] {name} (Machine {mid}) is ONLINE and operating nominally at RUL {rul}. "
            f"No immediate action required — continue standard monitoring. "
            f"Factory at {cap}% capacity, all systems healthy. "
            f"Schedule next inspection before RUL drops below 30."
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def issue_dispatch_orders(capacity_report: dict) -> tuple[str, bool]:
    """
    Agent 3 public interface. Called by agent_loop.py.

    Converts a capacity report dict into a plain-English dispatch string.
    Uses Gemini 2.5 Flash for live calls; falls back to template on failure.

    Args:
        capacity_report: dict from capacity_agent.update_capacity()

    Returns:
        dispatch_str:   str — the dispatch order to display on terminal
        used_fallback:  bool — True if template was used instead of Gemini
    """
    # If client is None (no API key), skip Gemini entirely
    if client is None:
        log.warning("No API key — skipping Gemini, using template fallback.")
        return _template_fallback(capacity_report), True

    last_error = ""

    # Build the context block — all numbers Agent 3 is allowed to use
    context = (
        f"Machine {capacity_report['machine_id']} "
        f"({capacity_report['machine_name']}) is now {capacity_report['status']}.\n"
        f"RUL = {capacity_report['rul']} cycles remaining.\n"
        f"Factory capacity: {capacity_report['capacity_pct']}%.\n"
        f"Total available time (T): {capacity_report['total_T']} hours.\n"
        f"Sum of Product Demand (ΣPD): {capacity_report['total_PD']} units.\n"
        f"Machine Requirement ratio (ΣPD/T): {capacity_report['machine_req']}.\n"
        f"Breakeven risk: {'YES — CRITICAL' if capacity_report['breakeven_risk'] else 'No — factory stable'}."
    )

    for attempt in range(MAX_RETRIES + 1):

        # On retry: inject specific error into prompt
        if attempt == 0:
            prompt = f"{FLOOR_MANAGER_SYSTEM_PROMPT}\n\nCapacity Report:\n{context}"
        else:
            prompt = (
                f"{FLOOR_MANAGER_SYSTEM_PROMPT}\n\n"
                f"Capacity Report:\n{context}\n\n"
                f"CORRECTION REQUIRED (attempt {attempt + 1}):\n"
                f"Your previous response was rejected: {last_error}\n"
                f"Please fix this and try again."
            )

        try:
            t_call = time.time()
            log.info("Gemini call attempt %d/%d  model=gemini-2.5-flash", attempt + 1, MAX_RETRIES + 1)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            api_ms = round((time.time() - t_call) * 1000, 1)
            text = response.text.strip()
            is_valid, error = _validate_output(text, capacity_report)

            if is_valid:
                log.info(
                    "✓ Attempt %d ACCEPTED  %.0fms  %s...",
                    attempt + 1, api_ms, text[:80],
                )
                return text, False

            else:
                last_error = error
                log.warning("✗ Attempt %d validation fail (%.0fms): %s", attempt + 1, api_ms, error)

        except Exception as e:
            last_error = str(e)
            log.error("✗ Attempt %d API fail: %s", attempt + 1, e)

    # All attempts failed — use template
    log.warning("All %d Gemini attempts failed. Using template fallback for status=%s", MAX_RETRIES + 1, capacity_report['status'])
    return _template_fallback(capacity_report), True
