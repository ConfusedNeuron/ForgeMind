# agents/agent_loop.py

from __future__ import annotations

import time
import numpy as np

from .input_guard      import is_valid_fault_input
from .diagnostic_agent import translate_fault_to_tensor
from .capacity_agent   import update_capacity, get_all_machine_statuses, reset_all
from .floor_manager    import issue_dispatch_orders
from .fallback_cache   import match_scenario

# ── Offline mode toggle ───────────────────────────────────────────────────────
# Set to True on first Gemini failure. Stays True for the rest of the session.
# Reset manually via reset_offline_mode() or a full app restart.
OFFLINE_MODE: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inject_spike(base_window: np.ndarray, spike: dict) -> np.ndarray:
    """Inject spike dict into a copy of base_window. Used in offline mode."""
    sensor_id = spike["sensor_id"]

    # Inline column lookup — mirrors diagnostic_agent.py logic
    if sensor_id.startswith("Xs"):
        col = int(sensor_id.replace("Xs", "")) + 4
    elif sensor_id.startswith("W"):
        col = int(sensor_id.replace("W", ""))
    else:
        raise ValueError(f"Unknown sensor_id: {sensor_id}")

    injected = base_window.copy()
    for pos in spike["affected_window_positions"]:
        injected[pos, col] = spike["spike_value"]
    return injected


def reset_offline_mode() -> None:
    """Re-enable Gemini calls after connectivity is restored."""
    global OFFLINE_MODE
    OFFLINE_MODE = False
    print("  [Pipeline] Offline mode cleared — Gemini calls re-enabled.")


def reset_factory() -> None:
    """Reset all machine states AND offline mode. Full demo restart."""
    reset_all()
    reset_offline_mode()
    print("  [Pipeline] Factory and offline mode fully reset.")


def get_pipeline_status() -> dict:
    """Quick health check — call from UI to show pipeline state in dashboard."""
    return {
        "offline_mode":    OFFLINE_MODE,
        "machine_statuses": get_all_machine_statuses(),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent_loop(
    user_text:      str,
    machine_id:     int,
    base_window:    np.ndarray,
    predict_rul_fn,
) -> dict:
    """
    Full agent pipeline: Input Guard → Diagnostic → DL Oracle → Capacity → Floor Manager.
    This is THE function Team Terminal calls.

    Args:
        user_text:      Professor's fault description string
        machine_id:     Which machine 1–5
        base_window:    (50, 18) numpy float32 array — current sensor baseline
        predict_rul_fn: Team DL's predict_rul(tensor) function

    Returns:
        dict — see schema below (Section 10.1 of TEAM_AGENT.md)
    """
    global OFFLINE_MODE
    t_start = time.time()

    # ── Step 0: Input Guard ── no LLM, no network ─────────────────────────────
    valid, reason = is_valid_fault_input(user_text)
    if not valid:
        print(f"  [Guard] ✗ Rejected: {reason}")
        return {
            "valid":            False,
            "rejection_reason": reason,
            "spike":            None,
            "rul":              None,
            "capacity_report":  None,
            "dispatch_orders":  None,
            "machine_statuses": get_all_machine_statuses(),
            "used_fallback":    False,
            "latency_ms":       round((time.time() - t_start) * 1000, 1),
        }

    print(f"  [Guard] ✓ Input valid")

    # ── Step 1: Diagnostic Agent ── Gemini or fallback cache ──────────────────
    if OFFLINE_MODE:
        scenario   = match_scenario(user_text)
        spike_obj  = scenario["diagnostic_spike"]
        spike_dict = spike_obj.model_dump()
        injected   = _inject_spike(base_window, spike_dict)
        diag_fallback = True
        print(f"  [Agent 1] Offline cache → sensor={spike_dict['sensor_id']}")
    else:
        injected, spike_dict, diag_fallback = translate_fault_to_tensor(base_window, user_text)
        if diag_fallback:
            OFFLINE_MODE = True

    # ── Step 2: DL Oracle ── Team DL's function ───────────────────────────────
    try:
        rul = float(predict_rul_fn(injected))
        print(f"  [Oracle] RUL = {rul}")
    except Exception as e:
        # Oracle failure — use a safe DEGRADED default rather than crashing
        rul = 25.0
        print(f"  [Oracle] ✗ predict_rul failed ({e}) — defaulting to RUL={rul}")

    # ── Step 3: Capacity Agent ── pure Python math, no LLM ───────────────────
    capacity_report = update_capacity(machine_id, rul)
    print(
        f"  [Agent 2] status={capacity_report['status']}  "
        f"cap={capacity_report['capacity_pct']}%  "
        f"risk={'⚠ YES' if capacity_report['breakeven_risk'] else 'No'}"
    )

    # ── Step 4: Floor Manager ── Gemini or fallback cache ─────────────────────
    if OFFLINE_MODE:
        scenario = match_scenario(user_text)
        dispatch = scenario["floor_manager_response"].format(**capacity_report)
        floor_fallback = True
        print(f"  [Agent 3] Offline cache → dispatch ready")
    else:
        dispatch, floor_fallback = issue_dispatch_orders(capacity_report)
        if floor_fallback:
            OFFLINE_MODE = True

    # ── Return ─────────────────────────────────────────────────────────────────
    latency_ms = round((time.time() - t_start) * 1000, 1)
    print(f"  [Pipeline] ✓ Done in {latency_ms}ms  offline={OFFLINE_MODE}")

    return {
        "valid":            True,
        "rejection_reason": "",
        "spike":            spike_dict,
        "rul":              rul,
        "capacity_report":  capacity_report,
        "dispatch_orders":  dispatch,
        "machine_statuses": get_all_machine_statuses(),
        "used_fallback":    OFFLINE_MODE,
        "latency_ms":       latency_ms,
    }
