# agents/fallback_cache.py
# Pre-scripted responses for offline demo mode.
#
# WHEN ACTIVATED:
#   - First Gemini API exception sets OFFLINE_MODE = True in agent_loop.py
#   - All subsequent calls skip Gemini and use these scenarios
#   - No retry thrashing during a live presentation
#
# DESIGN:
#   - floor_manager_response uses .format() placeholders
#   - Placeholders are filled with REAL capacity numbers from Agent 2
#   - So even fully offline, the numbers in dispatch orders are live and correct

from .schemas import SensorSpike

CACHED_SCENARIOS = {

    "bearing_overheat": {
        "trigger_keywords": ["bearing", "overheat", "temperature", "temp", "thermal", "hot"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs4",
            spike_value=0.95,
            affected_window_positions=[45, 46, 47, 48, 49],
            fault_severity="HIGH",
            plain_english_summary="Bearing temperature sensor critical — exceeding thermal limits. [OFFLINE MODE]",
        ),
        "floor_manager_response": (
            "[Floor Manager] {machine_name} (Machine {machine_id}) bearing temp critical — HALT production immediately. "
            "Dispatch maintenance crew and initiate cooldown protocol. "
            "Reroute workload to machines with available capacity headroom. "
            "Factory at {capacity_pct}% — authorize overtime if \u03a3PD/T exceeds breakeven threshold (~15.9)."
        ),
    },

    "pressure_surge": {
        "trigger_keywords": ["pressure", "surge", "psi", "hydraulic", "pneumatic", "compressed"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs2",
            spike_value=0.92,
            affected_window_positions=[40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            fault_severity="HIGH",
            plain_english_summary="Pressure sensor surge detected — possible seal failure. [OFFLINE MODE]",
        ),
        "floor_manager_response": (
            "[Floor Manager] Pressure anomaly on {machine_name} (Machine {machine_id}) — reduce load to 50% pending inspection. "
            "Check upstream valve integrity before restoring full operation. "
            "Do not bypass pressure relief systems during diagnostics. "
            "Factory at {capacity_pct}%, \u03a3PD/T at {machine_req}."
        ),
    },

    "vibration_anomaly": {
        "trigger_keywords": ["vibration", "vibrating", "shaking", "oscillation", "imbalance", "wobble"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs7",
            spike_value=0.88,
            affected_window_positions=[44, 45, 46, 47, 48, 49],
            fault_severity="MEDIUM",
            plain_english_summary="Vibration levels above normal — potential rotor imbalance. [OFFLINE MODE]",
        ),
        "floor_manager_response": (
            "[Floor Manager] Vibration alert on {machine_name} (Machine {machine_id}) — possible rotor imbalance detected. "
            "Schedule dynamic balancing during next shift changeover. "
            "Monitor RUL closely — pull offline immediately if RUL drops below 15. "
            "Factory holding at {capacity_pct}%."
        ),
    },

    "rpm_fluctuation": {
        "trigger_keywords": ["rpm", "speed", "rotation", "spin", "motor", "drive"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs10",
            spike_value=0.90,
            affected_window_positions=[46, 47, 48, 49],
            fault_severity="MEDIUM",
            plain_english_summary="RPM fluctuation detected — possible drive belt issue. [OFFLINE MODE]",
        ),
        "floor_manager_response": (
            "[Floor Manager] RPM instability on {machine_name} (Machine {machine_id}) — reduce to DEGRADED mode at 50% load. "
            "Inspect drive belt and motor assembly before next full-speed cycle. "
            "Do not run at rated speed until drive components are cleared. "
            "\u03a3PD/T now at {machine_req} — factory at {capacity_pct}%."
        ),
    },

    "coolant_leak": {
        "trigger_keywords": ["coolant", "leak", "fluid", "lubrication", "oil", "drip"],
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs12",
            spike_value=0.87,
            affected_window_positions=[43, 44, 45, 46, 47, 48, 49],
            fault_severity="MEDIUM",
            plain_english_summary="Coolant system anomaly — possible fluid leak detected. [OFFLINE MODE]",
        ),
        "floor_manager_response": (
            "[Floor Manager] Coolant anomaly on {machine_name} (Machine {machine_id}) — inspect fluid lines immediately. "
            "Reduce operating temperature and monitor coolant levels every 15 minutes. "
            "If leak confirmed, shut down and dispatch maintenance before next cycle. "
            "Factory at {capacity_pct}%, RUL at {rul} cycles remaining."
        ),
    },

    "general_fault": {
        "trigger_keywords": [],   # catch-all — matches when nothing else does
        "diagnostic_spike": SensorSpike(
            sensor_id="Xs0",
            spike_value=0.93,
            affected_window_positions=[47, 48, 49],
            fault_severity="HIGH",
            plain_english_summary="General sensor anomaly — unclassified fault pattern. [OFFLINE MODE]",
        ),
        "floor_manager_response": (
            "[Floor Manager] Anomaly detected on {machine_name} (Machine {machine_id}) — initiating precautionary slowdown. "
            "Maintenance team: inspect and report within 30 minutes. "
            "Hold all new job assignments on this unit until cleared. "
            "Factory capacity at {capacity_pct}%, \u03a3PD/T at {machine_req}."
        ),
    },
}


def match_scenario(user_text: str) -> dict:
    """
    Find the best matching cached scenario by keyword overlap.
    Falls back to 'general_fault' if nothing matches.

    Args:
        user_text: Professor's fault description string

    Returns:
        One scenario dict with keys: trigger_keywords, diagnostic_spike,
        floor_manager_response
    """
    text_lower = user_text.lower()
    best_match = None
    best_score = 0

    for name, scenario in CACHED_SCENARIOS.items():
        keywords = scenario["trigger_keywords"]
        if not keywords:
            continue   # skip general_fault in scoring — it's the fallback
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_match = scenario

    return best_match if best_match else CACHED_SCENARIOS["general_fault"]
