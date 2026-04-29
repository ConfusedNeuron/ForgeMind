# agents/agent_loop.py

from __future__ import annotations

import time
import numpy as np

from .log_config        import get_logger
from .input_guard      import is_valid_fault_input
from .diagnostic_agent import translate_fault_to_tensor
from .capacity_agent   import update_capacity, get_all_machine_statuses, reset_all
from .floor_manager    import issue_dispatch_orders
from .fallback_cache   import match_scenario

log = get_logger("pipeline")

# ── Offline mode toggle ───────────────────────────────────────────────────────
# Set to True on first Gemini failure. Stays True for the rest of the session.
# Reset manually via reset_offline_mode() or a full app restart.
OFFLINE_MODE: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inject_spike(base_window: np.ndarray, spike: dict) -> np.ndarray:
    """Inject spike dict into a copy of base_window. Used in offline mode.

    Uses the same ADDITIVE multi-sensor ramp pattern as the diagnostic
    agent's injection.  Each hit adds damage on top of current state.
    spike_value is in normalised [0, 1]; we convert to raw physical units.
    """
    from dl_engine.inference import raw_value_for_scaled, get_scaler_ranges
    from .diagnostic_agent import SENSOR_TO_COL, SENSOR_CORRELATIONS, RAMP_ESCALATION

    sensor_id = spike["sensor_id"]

    # Inline column lookup — mirrors diagnostic_agent.py logic
    if sensor_id.startswith("Xs"):
        col = int(sensor_id.replace("Xs", "")) + 4
    elif sensor_id.startswith("W"):
        col = int(sensor_id.replace("W", ""))
    else:
        raise ValueError(f"Unknown sensor_id: {sensor_id}")

    injected = base_window.copy()
    ranges = get_scaler_ranges()

    def _current_scaled(c: int, raw_val: float) -> float:
        lo  = float(ranges["min"][c])
        rng = float(ranges["range"][c])
        return (raw_val - lo) / rng if rng > 0 else 0.0

    # ── Primary sensor: additive ramp ─────────────────────────────────────
    raw_start      = float(injected[0, col])
    current_scaled = _current_scaled(col, raw_start)
    target_scaled  = min(0.98, current_scaled + spike["spike_value"] * RAMP_ESCALATION)
    raw_end        = raw_value_for_scaled(col, target_scaled)
    ramp = np.linspace(raw_start, raw_end, 50).astype(np.float32)
    injected[:, col] = ramp

    # ── Correlated sensors: additive scaled ramp ──────────────────────────
    for corr_sensor_id, intensity in SENSOR_CORRELATIONS.get(sensor_id, []):
        corr_col     = SENSOR_TO_COL[corr_sensor_id]
        corr_start   = float(injected[0, corr_col])
        corr_current = _current_scaled(corr_col, corr_start)
        corr_delta   = spike["spike_value"] * intensity * RAMP_ESCALATION
        corr_target  = min(0.98, corr_current + corr_delta)
        corr_end     = raw_value_for_scaled(corr_col, corr_target)
        corr_ramp = np.linspace(corr_start, corr_end, 50).astype(np.float32)
        injected[:, corr_col] = corr_ramp

    return injected


def reset_offline_mode() -> None:
    """Re-enable Gemini calls after connectivity is restored."""
    global OFFLINE_MODE
    OFFLINE_MODE = False
    log.info("Offline mode cleared — Gemini calls re-enabled")


def reset_factory() -> None:
    """Reset all machine states AND offline mode. Full demo restart."""
    reset_all()
    reset_offline_mode()
    log.info("Factory and offline mode fully reset")


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
    log.info("┌── Pipeline START  machine=%d  input=%r", machine_id, user_text[:80])

    # ── Step 0: Input Guard ── no LLM, no network ─────────────────────────────
    t0 = time.time()
    valid, reason = is_valid_fault_input(user_text)
    guard_ms = round((time.time() - t0) * 1000, 1)
    if not valid:
        log.warning("│ [Guard] ✗ REJECTED (%s)  %.1fms", reason, guard_ms)
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

    log.info("│ [Guard] ✓ Input accepted  %.1fms", guard_ms)

    # ── Step 1: Diagnostic Agent ── Gemini or fallback cache ──────────────────
    t1 = time.time()
    if OFFLINE_MODE:
        scenario   = match_scenario(user_text)
        spike_obj  = scenario["diagnostic_spike"]
        spike_dict = spike_obj.model_dump()
        injected   = _inject_spike(base_window, spike_dict)
        diag_fallback = True
        log.info(
            "│ [Diagnostic] Offline-cache  sensor=%s  val=%.2f  %.1fms",
            spike_dict['sensor_id'], spike_dict['spike_value'],
            (time.time() - t1) * 1000,
        )
    else:
        log.info("│ [Diagnostic] Calling Gemini …")
        injected, spike_dict, diag_fallback = translate_fault_to_tensor(base_window, user_text)
        diag_ms = round((time.time() - t1) * 1000, 1)
        mode = "FALLBACK" if diag_fallback else "GEMINI"
        log.info(
            "│ [Diagnostic] ✓ %s  sensor=%s  val=%.2f  severity=%s  %.1fms",
            mode, spike_dict['sensor_id'], spike_dict['spike_value'],
            spike_dict.get('fault_severity', '?'), diag_ms,
        )
        if diag_fallback:
            OFFLINE_MODE = True
            log.warning("│ [Diagnostic] Gemini failed → OFFLINE_MODE activated")

    # ── Step 2: DL Oracle ── Team DL's function ───────────────────────────────
    t2 = time.time()
    try:
        rul = float(predict_rul_fn(injected))
        oracle_ms = round((time.time() - t2) * 1000, 1)
        log.info("│ [Oracle] ✓ RUL = %.1f  %.1fms", rul, oracle_ms)
    except Exception as e:
        # Oracle failure — use a safe DEGRADED default rather than crashing
        rul = 25.0
        oracle_ms = round((time.time() - t2) * 1000, 1)
        log.error("│ [Oracle] ✗ predict_rul failed (%s) → default RUL=%.1f  %.1fms", e, rul, oracle_ms)

    # ── Step 3: Capacity Agent ── pure Python math, no LLM ───────────────────
    t3 = time.time()
    capacity_report = update_capacity(machine_id, rul)
    cap_ms = round((time.time() - t3) * 1000, 1)
    log.info(
        "│ [Capacity] status=%s  cap=%.0f%%  ΣPD/T=%.2f  risk=%s  %.1fms",
        capacity_report['status'], capacity_report['capacity_pct'],
        capacity_report['machine_req'],
        '⚠ YES' if capacity_report['breakeven_risk'] else 'No',
        cap_ms,
    )

    # ── Step 4: Floor Manager ── Gemini or fallback cache ─────────────────────
    t4 = time.time()
    if OFFLINE_MODE:
        scenario = match_scenario(user_text)
        dispatch = scenario["floor_manager_response"].format(**capacity_report)
        floor_fallback = True
        log.info("│ [Floor Mgr] Offline-cache  %.1fms", (time.time() - t4) * 1000)
    else:
        log.info("│ [Floor Mgr] Calling Gemini …")
        dispatch, floor_fallback = issue_dispatch_orders(capacity_report)
        floor_ms = round((time.time() - t4) * 1000, 1)
        mode = "FALLBACK" if floor_fallback else "GEMINI"
        log.info("│ [Floor Mgr] ✓ %s  %.1fms", mode, floor_ms)
        if floor_fallback:
            OFFLINE_MODE = True
            log.warning("│ [Floor Mgr] Gemini failed → OFFLINE_MODE activated")

    # ── Return ─────────────────────────────────────────────────────────────────
    latency_ms = round((time.time() - t_start) * 1000, 1)
    log.info(
        "└── Pipeline DONE  RUL=%.1f  status=%s  offline=%s  total=%.0fms",
        rul, capacity_report['status'], OFFLINE_MODE, latency_ms,
    )

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
