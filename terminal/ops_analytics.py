"""
Ops Analytics — all operations-focused computations for the factory dashboard.

These functions are intentionally written in "ops language" — what an
operations manager or floor supervisor actually cares about.

No raw sensor jargon, no statistical notation. Just:
  - Which machine needs attention first?
  - Is the factory going to make its shift target?
  - Should I trust this reading?
  - Is this an emergency right now?

All functions are pure (no side effects) and can be called from any thread.
"""

import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 1.4.2  SUDDEN RUL CLIFF DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_rul_cliff(old_rul: float, new_rul: float, threshold: float = 0.40) -> bool:
    """
    Returns True if RUL dropped more than `threshold` fraction in one step.

    Default threshold = 40%.  Normal wear degrades ~5-10 cycles per step;
    a 40%+ drop in one reading means an acute fault has been detected — not
    gradual wear.  This triggers an EMERGENCY log entry in the comms pane.

    Examples:
        old=100, new=55  → drop=45% → True  (cliff)
        old=100, new=80  → drop=20% → False (normal wear)
        old=30,  new=14  → drop=53% → True  (cliff into OFFLINE territory)
        old=50,  new=30  → drop=40% → True  (exactly at threshold → cliff)

    Args:
        old_rul:   RUL before this chaos event.
        new_rul:   RUL after the DL oracle ran.
        threshold: Fraction drop that constitutes a cliff (default 0.40 = 40%).

    Returns:
        bool
    """
    if old_rul <= 0 or new_rul >= old_rul:
        return False
    return (old_rul - new_rul) / old_rul >= threshold  # FIX: >= so exactly 40% is a cliff


# ─────────────────────────────────────────────────────────────────────────────
# 1.1.3  PREDICTION RELIABILITY  (ops framing of "confidence intervals")
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_reliability(rul_history: list) -> Tuple[str, str]:
    """
    Estimate how much the model's RUL predictions have been fluctuating
    for this machine over its last 5 readings.

    An ops manager doesn't want "±σ = 12.3 cycles".  They want:
      HIGH   → "The model is confident. Trust this number."
      MEDIUM → "Some variance. Check manually if in doubt."
      LOW    → "Readings are unstable. Don't make big decisions on this alone."

    Uses Coefficient of Variation (σ/μ) of the last 5 RUL predictions.
    This requires NO changes to the DL engine — it's purely a rolling
    statistic over values we already store.

    Args:
        rul_history: list of recent RUL floats for one machine (latest last).

    Returns:
        (label: str, color: str)  e.g. ("HIGH", "green")
    """
    if len(rul_history) < 3:
        return ("CALIBRATING", "dim")

    raw    = rul_history[-5:]
    values = np.array([v for v in raw if np.isfinite(v)], dtype=float)
    if len(values) < 2:
        return ("CALIBRATING", "dim")
    mean = float(np.mean(values))
    std  = float(np.std(values))
    cv   = std / (mean + 1e-6)

    if cv < 0.08:
        return ("HIGH", "green")
    elif cv < 0.22:
        return ("MEDIUM", "yellow")
    else:
        return ("LOW", "red")


# ─────────────────────────────────────────────────────────────────────────────
# 1.4.1  SENSOR DATA QUALITY / SATURATION WARNINGS
# ─────────────────────────────────────────────────────────────────────────────

def check_sensor_saturation(
    sensor_history: list,
    n_consecutive: int = 5,
) -> List[Tuple[str, str]]:
    """
    Detect sensors that are "stuck" at their maximum or zero value.

    A saturated sensor means the physical instrument has likely:
      - Maxed out (stuck high) → possible overload or sensor damage
      - Dropped to zero (stuck low) → possible disconnection or power loss

    In both cases the RUL prediction for that machine becomes unreliable
    because 1+ of the 18 input features is garbage.  The ops manager sees
    a "DATA QUALITY ALERT" rather than raw sensor indices.

    NOTE: This function only fires if sensor_history has real data.
    In stub mode (random baseline), readings cluster in [0.3, 0.7], so
    this will not produce false positives.

    Args:
        sensor_history: list of 18 lists, each containing recent float values
                        (the ring buffer from FactoryState).
        n_consecutive:  How many consecutive readings must be saturated
                        before alerting (default 5 — avoids single-spike FPs).

    Returns:
        List of (sensor_name, "MAX" or "ZERO") for saturated sensors.
        Empty list = all sensors healthy.
    """
    from .layout import SENSOR_DISPLAY_NAMES
    sensor_names = SENSOR_DISPLAY_NAMES
    saturated = []

    for idx, history in enumerate(sensor_history):
        if len(history) < n_consecutive:
            continue
        recent = history[-n_consecutive:]
        if all(v >= 0.97 for v in recent):
            saturated.append((sensor_names[idx], "MAX"))
        elif all(v <= 0.03 for v in recent):
            saturated.append((sensor_names[idx], "ZERO"))

    return saturated


# ─────────────────────────────────────────────────────────────────────────────
# 1.2.2  PREDICTIVE MAINTENANCE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def compute_maintenance_schedule(machines: dict) -> List[dict]:
    """
    Build a ranked, plain-English maintenance action queue.

    An ops manager asks: "What do I need to do, and in what order?"
    This function answers that directly.  Every machine with RUL ≤ 30
    (WARNING threshold) appears in the queue with a clear action label.

    Urgency rules:
      OFFLINE     → STOP & REPAIR  (machine is down right now)
      RUL ≤ 15   → SCHEDULE TODAY  (will fail within the shift)
      RUL ≤ 30   → PLAN THIS WEEK  (degraded, watch closely)
      RUL > 30   → Not shown       (healthy — no action needed)

    Args:
        machines: dict of {machine_id: MachineState}

    Returns:
        Sorted list of dicts.  Each dict has:
          rank, machine_id, machine_name, status, rul, urgency, action, color
        Sorted: OFFLINE first, then by lowest RUL.
    """
    schedule = []

    for mid, m in machines.items():
        if m.status == "OFFLINE":
            schedule.append({
                "machine_id":   mid,
                "machine_name": m.name,
                "status":       m.status,
                "rul":          m.rul,
                "urgency":      "IMMEDIATE",
                "action":       "STOP & REPAIR",
                "color":        "red",
                "sort_key":     (0, m.rul),
            })
        elif m.rul <= 15:
            schedule.append({
                "machine_id":   mid,
                "machine_name": m.name,
                "status":       m.status,
                "rul":          m.rul,
                "urgency":      "TODAY",
                "action":       "SCHEDULE NOW",
                "color":        "red",
                "sort_key":     (1, m.rul),
            })
        elif m.rul <= 30:
            schedule.append({
                "machine_id":   mid,
                "machine_name": m.name,
                "status":       m.status,
                "rul":          m.rul,
                "urgency":      "THIS WEEK",
                "action":       "PLAN MAINT.",
                "color":        "yellow",
                "sort_key":     (2, m.rul),
            })

    schedule.sort(key=lambda x: x["sort_key"])
    for i, item in enumerate(schedule, 1):
        item["rank"] = i

    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED OPS FEATURE A:  SHIFT HEALTH BANNER
# ─────────────────────────────────────────────────────────────────────────────

def compute_shift_health(machines: dict, capacity_pct: float) -> Tuple[str, str]:
    """
    Single-line shift health status for the ops manager.

    This replaces the cryptic "ΣPD/T: 1.12" metric with something a real
    floor supervisor can read at a glance across a noisy factory floor.

    Examples:
      "CRITICAL  —  2 machines offline. Production severely at risk."  [red]
      "AT RISK   —  1 offline + 1 degraded. Capacity: 72%"            [red]
      "CAUTION   —  1 machine degraded. Capacity: 88%"                [yellow]
      "NOMINAL   —  All 5 machines running. Capacity: 100%"           [green]

    Args:
        machines:     dict of {machine_id: MachineState}
        capacity_pct: current factory capacity percentage (0–100)

    Returns:
        (summary: str, color: str)
    """
    offline  = sum(1 for m in machines.values() if m.status == "OFFLINE")
    degraded = sum(1 for m in machines.values() if m.status == "DEGRADED")
    online   = sum(1 for m in machines.values() if m.status == "ONLINE")

    if offline >= 2:
        return (
            f"CRITICAL  —  {offline} machines offline. Production severely at risk.",
            "bold red",
        )
    elif offline == 1 and degraded >= 1:
        return (
            f"AT RISK  —  1 offline + {degraded} degraded. Capacity: {capacity_pct:.0f}%",
            "red",
        )
    elif offline == 1:
        return (
            f"CAUTION  —  Machine offline. Capacity: {capacity_pct:.0f}%",
            "yellow",
        )
    elif degraded >= 2:
        return (
            f"CAUTION  —  {degraded} machines degraded. Monitor closely.",
            "yellow",
        )
    elif degraded == 1:
        return (
            f"WATCH  —  1 machine degraded. Capacity: {capacity_pct:.0f}%",
            "yellow",
        )
    else:
        return (
            f"NOMINAL  —  All {online} machines running. Capacity: {capacity_pct:.0f}%",
            "green",
        )


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED OPS FEATURE B:  MULTI-MACHINE DEGRADATION LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────

def compute_degradation_leaderboard(
    machines: dict,
    rul_histories: dict,
) -> List[dict]:
    """
    Rank all machines by how fast their RUL is falling.

    Instead of showing a raw 18×5 sensor comparison grid (which means nothing
    to an ops manager), this answers the question they actually ask:
    "Which machine is deteriorating the fastest?"

    Computes a slope (cycles per event) from each machine's recent RUL history.
    A steeper negative slope = faster deterioration = higher rank (more urgent).

    Args:
        machines:      dict of {machine_id: MachineState}
        rul_histories: dict of {machine_id: [rul_1, rul_2, ...]}

    Returns:
        List of dicts sorted by degradation rate (fastest first):
          {machine_id, machine_name, rul, slope, trend_label, trend_color}
    """
    board = []

    for mid, m in machines.items():
        history = rul_histories.get(mid, [])

        if len(history) >= 2:
            n      = min(5, len(history))
            recent = history[-n:]
            try:
                clean = [v for v in recent if np.isfinite(v)]
                if len(clean) >= 2 and (max(clean) - min(clean)) > 1e-9:
                    x     = np.arange(len(clean), dtype=float)
                    slope = float(np.polyfit(x, clean, 1)[0])
                else:
                    slope = 0.0
            except Exception:
                slope = 0.0
        else:
            slope = 0.0

        if slope < -10:
            trend_label, trend_color = "FAST ↘", "red"
        elif slope < -2:
            trend_label, trend_color = "SLOW ↘", "yellow"
        elif slope > 2:
            trend_label, trend_color = "IMPROVING ↗", "green"
        else:
            trend_label, trend_color = "STABLE →", "dim"

        board.append({
            "machine_id":   mid,
            "machine_name": m.name,
            "status":       m.status,
            "rul":          m.rul,
            "slope":        slope,
            "trend_label":  trend_label,
            "trend_color":  trend_color,
        })

    board.sort(key=lambda x: x["slope"])
    return board
