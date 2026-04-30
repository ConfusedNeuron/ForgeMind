# agents/capacity_agent.py
# ════════════════════════════════════════════════════════════════════════
# ZERO LLM calls. ZERO external packages. Pure Python math.
# No google-genai. No pydantic. No numpy. Just arithmetic.
# ════════════════════════════════════════════════════════════════════════

from __future__ import annotations

# ── RUL thresholds ────────────────────────────────────────────────────────────
RUL_OFFLINE_THRESHOLD  = 15    # RUL ≤ 15  → OFFLINE
RUL_DEGRADED_THRESHOLD = 30    # RUL ≤ 30  → DEGRADED (and > 15)
DEGRADED_CAPACITY_FACTOR = 0.5 # DEGRADED machines run at 50% available time

# ── Healthy baseline for breakeven risk calculation ───────────────────────────
# Computed once from the known healthy total (595 units / 40 hours).
# Breakeven risk fires when machine_req exceeds this by more than 7%.
_HEALTHY_TOTAL_PD = 595
_HEALTHY_TOTAL_T  = 40.0
HEALTHY_BASELINE_REQ  = _HEALTHY_TOTAL_PD / _HEALTHY_TOTAL_T  # = 14.875
BREAKEVEN_THRESHOLD   = HEALTHY_BASELINE_REQ * 1.07           # ≈ 15.92

# ── Machine registry ──────────────────────────────────────────────────────────
# Galanz-style microwave oven factory — Inverted-Y topology:
#
#   Metal Press (1) ──↘
#                       → Final Assembly (4) → QC & Pack (5)
#   Paint & Coat (2) ─↗
#   PCB Line (3) ────↗
#
# This is mutable state. Each update_capacity() call modifies it in-place.
# reset_all() restores it to starting values.
# Machine IDs are 1-indexed (1–5) to match what the professor says: "Machine 4"
# Total product_demand = 595 (preserved for MIOM ΣPD/T math consistency)
MACHINES: dict[int, dict] = {
    1: {
        "name": "Metal Press",
        "base_time": 8.0,
        "product_demand": 140,       # high-throughput stamping
        "available_time": 8.0,
        "rul": 999.0,
        "status": "ONLINE",
    },
    2: {
        "name": "Paint & Coat",
        "base_time": 8.0,
        "product_demand": 110,       # slower coating process
        "available_time": 8.0,
        "rul": 999.0,
        "status": "ONLINE",
    },
    3: {
        "name": "PCB Line",
        "base_time": 8.0,
        "product_demand": 125,       # SMT board assembly
        "available_time": 8.0,
        "rul": 999.0,
        "status": "ONLINE",
    },
    4: {
        "name": "Final Assembly",
        "base_time": 8.0,
        "product_demand": 120,       # merge point — chassis + board + magnetron
        "available_time": 8.0,
        "rul": 999.0,
        "status": "ONLINE",
    },
    5: {
        "name": "QC & Pack",
        "base_time": 8.0,
        "product_demand": 100,       # bottleneck — burn-in test + boxing
        "available_time": 8.0,
        "rul": 999.0,
        "status": "ONLINE",
    },
}


# ── Core function ─────────────────────────────────────────────────────────────

def update_capacity(machine_id: int, new_rul: float) -> dict:
    """
    Update one machine's RUL and recompute factory-wide capacity metrics.
    Modifies MACHINES[machine_id] in-place (permanent for this session).

    Called by agent_loop.py after predict_rul() returns a value.

    Args:
        machine_id: 1–5 (matches professor's "Machine 4")
        new_rul:    float from DL oracle — e.g. 12.0, 22.5, 45.0

    Returns:
        dict with all capacity metrics (see data contract above)

    Raises:
        KeyError: if machine_id not in 1–5
    """
    if machine_id not in MACHINES:
        raise KeyError(
            f"machine_id {machine_id} not found. "
            f"Valid IDs are: {sorted(MACHINES.keys())}"
        )

    import math
    if not math.isfinite(new_rul) or new_rul < 0:
        new_rul = 0.0

    machine = MACHINES[machine_id]
    machine["rul"] = new_rul

    # ── Step 1: Determine new status ─────────────────────────────────────────
    if new_rul <= RUL_OFFLINE_THRESHOLD:
        machine["available_time"] = 0.0
        machine["status"] = "OFFLINE"

    elif new_rul <= RUL_DEGRADED_THRESHOLD:
        machine["available_time"] = machine["base_time"] * DEGRADED_CAPACITY_FACTOR
        machine["status"] = "DEGRADED"

    else:
        machine["available_time"] = machine["base_time"]
        machine["status"] = "ONLINE"

    # ── Step 2: Factory-wide metrics ──────────────────────────────────────────
    total_T  = sum(m["available_time"] for m in MACHINES.values())
    total_PD = sum(m["product_demand"]  for m in MACHINES.values())  # always 595

    # Guard against all machines offline (division by zero)
    if total_T > 0:
        machine_req = total_PD / total_T
    else:
        machine_req = float("inf")

    max_T        = len(MACHINES) * 8.0    # = 40.0 hrs
    capacity_pct = (total_T / max_T) * 100

    breakeven_risk = (
        machine_req == float("inf")
        or machine_req > BREAKEVEN_THRESHOLD
    )

    return {
        "machine_id":     machine_id,
        "machine_name":   machine["name"],
        "status":         machine["status"],
        "rul":            round(new_rul, 1),
        "total_T":        round(total_T, 2),
        "total_PD":       total_PD,
        "machine_req":    round(machine_req, 3) if machine_req != float("inf") else float("inf"),
        "capacity_pct":   round(capacity_pct, 1),
        "breakeven_risk": breakeven_risk,
    }


# ── Read-only helpers ─────────────────────────────────────────────────────────

def get_all_machine_statuses() -> list[dict]:
    """
    Current status of all 5 machines.
    Called by agent_loop.py to populate the UI dashboard panel.

    Returns:
        List of dicts, one per machine, sorted by machine_id.
    """
    return [
        {
            "id":             mid,
            "name":           m["name"],
            "status":         m["status"],
            "rul":            round(m["rul"], 1),
            "available_time": m["available_time"],
            "base_time":      m["base_time"],
        }
        for mid, m in sorted(MACHINES.items())
    ]


def get_factory_snapshot() -> dict:
    """
    Current factory-wide metrics without updating any machine.
    Used by the dashboard for a live readout between fault injections.
    """
    total_T  = sum(m["available_time"] for m in MACHINES.values())
    total_PD = sum(m["product_demand"]  for m in MACHINES.values())

    if total_T > 0:
        machine_req = total_PD / total_T
    else:
        machine_req = float("inf")

    max_T        = len(MACHINES) * 8.0
    capacity_pct = (total_T / max_T) * 100
    breakeven_risk = (
        machine_req == float("inf")
        or machine_req > BREAKEVEN_THRESHOLD
    )

    return {
        "total_T":        round(total_T, 2),
        "total_PD":       total_PD,
        "machine_req":    round(machine_req, 3) if machine_req != float("inf") else float("inf"),
        "capacity_pct":   round(capacity_pct, 1),
        "breakeven_risk": breakeven_risk,
    }


# ── State management ──────────────────────────────────────────────────────────

def reset_all() -> None:
    """
    Restore all machines to ONLINE / full capacity / RUL=999.
    Called by the Terminal when the professor presses Ctrl+R to restart the demo.
    Does not reset OFFLINE_MODE in agent_loop.py — call reset_offline_mode()
    separately if needed.
    """
    for m in MACHINES.values():
        m["available_time"] = m["base_time"]
        m["rul"]            = 999.0
        m["status"]         = "ONLINE"
