# agents/diagnostic_agent.py

import os
import time
from pathlib import Path


import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from .schemas import SensorSpike, FaultSeverity
from .prompts import DIAGNOSTIC_SYSTEM_PROMPT
from .log_config import get_logger


# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Get API key for diagnostic agent
api_key = os.getenv("GEMINI_API_KEY_DIAGNOSTIC")

if not api_key:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY_DIAGNOSTIC not found in environment. "
        "Diagnostic agent will use deterministic fallback spikes.",
        stacklevel=2,
    )
    client = None
else:
    # print(f"Loaded API key for DIAGNOSTIC agent: {bool(api_key)}")  # commented — noisy before TUI
    client = genai.Client(api_key=api_key)

# ── Structured logger ──────────────────────────────────────────────────────────
log = get_logger("diagnostic")


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_RETRIES   = 2
VALID_SENSORS = {f"W{i}" for i in range(4)} | {f"Xs{i}" for i in range(14)}

# ── Sensor ID → tensor column index ───────────────────────────────────────────
# Feature order in the (50, 18) tensor: [W0, W1, W2, W3, Xs0, Xs1, ... Xs13]
SENSOR_TO_COL: dict[str, int] = {
    **{f"W{i}":  i     for i in range(4)},   # W0→0, W1→1, W2→2, W3→3
    **{f"Xs{i}": i + 4 for i in range(14)},  # Xs0→4, Xs1→5, ..., Xs13→17
}

# ── Deterministic fallback spikes ─────────────────────────────────────────────
# Keyed by the dominant keyword in the fault description.
# Used when all Gemini retries fail or all return invalid sensor IDs.
# Values here are tuned for the CNN-LSTM to produce a meaningful RUL drop.
FALLBACK_SPIKES: dict[str, SensorSpike] = {
    "temperature": SensorSpike(
        sensor_id="Xs4", spike_value=0.95,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Bearing temperature sensor Xs4 — critical thermal spike. [FALLBACK]"
    ),
    "bearing": SensorSpike(
        sensor_id="Xs4", spike_value=0.93,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Bearing temperature sensor Xs4 — overheat detected. [FALLBACK]"
    ),
    "pressure": SensorSpike(
        sensor_id="Xs2", spike_value=0.92,
        affected_window_positions=[40, 41, 43, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Pressure sensor Xs2 — abnormal surge reading. [FALLBACK]"
    ),
    "vibration": SensorSpike(
        sensor_id="Xs7", spike_value=0.88,
        affected_window_positions=[43, 44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="Vibration sensor Xs7 — oscillation above safe threshold. [FALLBACK]"
    ),
    "rpm": SensorSpike(
        sensor_id="Xs10", spike_value=0.89,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="RPM sensor Xs10 — rotational speed anomaly. [FALLBACK]"
    ),
    "speed": SensorSpike(
        sensor_id="Xs10", spike_value=0.87,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.MEDIUM,
        plain_english_summary="Speed sensor Xs10 — drive fluctuation detected. [FALLBACK]"
    ),
    "coolant": SensorSpike(
        sensor_id="Xs12", spike_value=0.91,
        affected_window_positions=[44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Coolant sensor Xs12 — flow disruption detected. [FALLBACK]"
    ),
    "leak": SensorSpike(
        sensor_id="Xs12", spike_value=0.90,
        affected_window_positions=[43, 44, 45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Coolant sensor Xs12 — possible seal failure. [FALLBACK]"
    ),
    "overload": SensorSpike(
        sensor_id="W0", spike_value=0.94,
        affected_window_positions=[45, 46, 47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Load sensor W0 — machine overload condition. [FALLBACK]"
    ),
    "default": SensorSpike(
        sensor_id="Xs4", spike_value=0.93,
        affected_window_positions=[47, 48, 49],
        fault_severity=FaultSeverity.HIGH,
        plain_english_summary="Sensor anomaly detected — unclassified fault pattern. [FALLBACK]"
    ),
}

# Keyword priority order — first match wins
FALLBACK_KEYWORD_ORDER = [
    "bearing", "temperature", "pressure", "vibration",
    "coolant", "leak", "rpm", "speed", "overload",
]


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_domain(spike: SensorSpike) -> tuple[bool, str]:
    """
    Validates domain-specific constraints that Pydantic cannot enforce.
    Called after every Gemini response, before accepting the spike.

    Pydantic handles: field presence, types, spike_value in [0,1].
    This handles: sensor_id must be a real sensor, positions in valid range.

    Returns:
        (True, "")              → spike is valid, use it
        (False, error_message)  → spike is invalid, log error and retry
    """
    if spike.sensor_id not in VALID_SENSORS:
        return False, (
            f"sensor_id '{spike.sensor_id}' is not a valid sensor. "
            f"Valid sensors are: W0–W3, Xs0–Xs13."
        )

    bad_positions = [p for p in spike.affected_window_positions if not (0 <= p <= 49)]
    if bad_positions:
        return False, (
            f"affected_window_positions contains out-of-range values: {bad_positions}. "
            f"All positions must be integers 0–49."
        )

    if len(spike.affected_window_positions) == 0:
        return False, "affected_window_positions is empty — must contain at least 1 position."

    if len(spike.affected_window_positions) > 10:
        return False, (
            f"affected_window_positions has {len(spike.affected_window_positions)} items — "
            f"maximum is 10."
        )

    # Warn about positions in first half of window (not a failure, just suspicious)
    early_positions = [p for p in spike.affected_window_positions if p < 25]
    if early_positions and len(early_positions) == len(spike.affected_window_positions):
        # All positions are in the first half — likely Gemini misunderstood the window
        # Still accept it (not a hard failure), but log a warning
        log.warning(
            "All spike positions are early in window (%s). "
            "Fault may not affect recent readings strongly.",
            early_positions,
        )

    return True, ""


# ── Fallback selection ─────────────────────────────────────────────────────────

def _get_fallback(user_text: str) -> SensorSpike:
    """
    Keyword-match the user's input to the best deterministic fallback.
    Checks keywords in priority order (FALLBACK_KEYWORD_ORDER).
    Returns a copy of the matching SensorSpike (not the original).
    """
    text_lower = user_text.lower()

    for keyword in FALLBACK_KEYWORD_ORDER:
        if keyword in text_lower:
            spike = FALLBACK_SPIKES[keyword]
            log.info("Fallback matched keyword: '%s' → %s", keyword, spike.sensor_id)
            return spike

    # No keyword matched — use default
    log.info("No keyword matched. Using default fallback.")
    return FALLBACK_SPIKES["default"]


# ── Sensor correlation map ─────────────────────────────────────────────────────
# When the primary sensor spikes, correlated sensors also degrade.
# Model probing showed single-sensor spikes barely move RUL, but 3+ sensor
# ramps produce dramatic drops (3-sensor ramp: RUL 71 → 8).
#
# CRITICAL: the CNN-LSTM is primarily sensitive to Xs2 (col 6) and Xs3 (col 7).
# These are the key degradation indicators in the N-CMAPSS turbofan data.
# Every fault type MUST include Xs2/Xs3 at some intensity — physically all
# machine faults eventually stress these thermal/pressure channels.
#
# Intensity hierarchy (tuned for gradual degradation path):
#   First fault → ONLINE (RUL 50-65), second fault → DEGRADED (RUL 20-35),
#   third fault → OFFLINE (RUL ≤15)
#   The injected[-2:] persistence in app.py compounds damage across hits.
#
#   0.65–0.70 = direct thermal/pressure fault (strongest RUL impact)
#   0.50–0.60 = mechanically coupled fault (moderate impact)
#   0.40–0.50 = indirect/operating-condition fault (mild impact)
#
# RAMP_ESCALATION controls how much each hit adds (additive injection).
# Module-level so agent_loop._inject_spike (offline path) can import it
# and stay in sync with the online injection.
RAMP_ESCALATION: float = 0.35   # fraction of spike added per hit

# ── Critical sensor ceiling caps ──────────────────────────────────────────────
# Model probing (probe_cliff.py, probe_ramp_vs_flat.py) revealed:
#
#   1. The CNN-LSTM's Xs2/Xs3 cliff is razor-sharp in isolation:
#      Xs2=0.48 → RUL 19 (DEGRADED) on a clean baseline
#
#   2. BUT the Xs4 ramp (always present as the primary sensor for most faults)
#      SUPPRESSES the cliff.  With Xs4 ramped to 0.76:
#        Xs2=0.48 → RUL 62 (still ONLINE!)
#        Xs2=0.52 → RUL 31 (DEGRADED — but only with Xs4 also capped)
#
#   3. Xs4 must therefore also be treated as critical: capped and flat-filled.
#      When Xs4 is capped at 0.48 on Hit 2 (not ramping to 0.76+), the
#      Xs2/Xs3 cliff activates reliably.
#
# Probe-validated 3-hit lifecycle:
#   Hit 1: Xs4=0.35, Xs2=0.30, Xs3=0.26 → RUL ~71 (ONLINE)
#   Hit 2: Xs4=0.48, Xs2=0.52, Xs3=0.45 → RUL ~31 (DEGRADED)
#   Hit 3: Xs4=0.81, Xs2=0.74, Xs3=0.65 → RUL ~1  (OFFLINE)
#
# The cap is selected based on the sensor's current scaled position.
CRITICAL_SENSORS: set[str] = {"Xs2", "Xs3", "Xs4"}
CRITICAL_SENSOR_CAPS: list[tuple[float, float]] = [
    # (if current_scaled < threshold, cap_at)
    (0.25, 0.35),   # Hit 1: fresh sensor → cap at 0.35 (well below cliff)
    (0.42, 0.52),   # Hit 2: stressed sensor → cap at 0.52 (DEGRADED with Xs4 present)
    # Beyond 0.42: uncapped (0.98) — Hit 3 pushes past cliff (OFFLINE)
]

# Format: primary_sensor → [(correlated_sensor, intensity_fraction), ...]
SENSOR_CORRELATIONS: dict[str, list[tuple[str, float]]] = {
    # Temperature faults → stress key degradation sensors
    "Xs4":  [("Xs2", 0.68), ("Xs3", 0.60)],
    "Xs5":  [("Xs2", 0.65), ("Xs3", 0.58)],
    # Pressure faults → co-located thermal stress
    "Xs2":  [("Xs3", 0.65), ("Xs6", 0.50)],
    "Xs6":  [("Xs2", 0.62), ("Xs3", 0.55)],
    # Bearing/fan faults → friction heat propagates to thermal sensors
    "Xs0":  [("Xs2", 0.58), ("Xs3", 0.50), ("Xs1", 0.55)],
    "Xs1":  [("Xs2", 0.58), ("Xs3", 0.50), ("Xs0", 0.55)],
    # Vibration/enthalpy → mechanical stress raises temps
    "Xs7":  [("Xs2", 0.62), ("Xs3", 0.52), ("Xs0", 0.35)],
    # Speed/RPM faults → off-design operation strains thermal path
    "Xs8":  [("Xs2", 0.55), ("Xs3", 0.48), ("Xs9", 0.52)],
    "Xs9":  [("Xs2", 0.55), ("Xs3", 0.48), ("Xs8", 0.52)],
    "Xs10": [("Xs2", 0.50), ("Xs3", 0.44), ("Xs4", 0.42)],
    "Xs13": [("Xs2", 0.50), ("Xs3", 0.44), ("Xs4", 0.38)],
    # Coolant/bleed faults → reduced cooling raises degradation temps
    "Xs12": [("Xs2", 0.58), ("Xs3", 0.50), ("Xs11", 0.45)],
    "Xs11": [("Xs2", 0.55), ("Xs3", 0.48), ("Xs12", 0.42)],
    # Operating condition faults → affect thermal equilibrium
    "W0":   [("Xs2", 0.52), ("Xs3", 0.45), ("W2", 0.30)],
    "W1":   [("Xs2", 0.48), ("Xs3", 0.40)],
    "W2":   [("Xs2", 0.50), ("Xs3", 0.42)],
    "W3":   [("Xs2", 0.52), ("Xs3", 0.45)],
}


def _get_critical_cap(sensor_id: str, current_scaled: float) -> float:
    """
    Return the maximum allowed scaled value for a critical sensor given its
    current degradation level.

    Non-critical sensors always get 0.98 (effectively uncapped).
    Critical sensors (Xs2/Xs3) get a ceiling that walks them through the
    CNN-LSTM's sensitivity cliff in controlled steps.

    Args:
        sensor_id:      sensor identifier (e.g. "Xs2")
        current_scaled: sensor's current position in [0, 1]

    Returns:
        float — maximum target scaled value for this injection
    """
    if sensor_id not in CRITICAL_SENSORS:
        return 0.98

    for threshold, cap in CRITICAL_SENSOR_CAPS:
        if current_scaled < threshold:
            return cap

    return 0.98   # past all thresholds — fully uncapped


def _inject_spike(base_window: np.ndarray, spike: SensorSpike) -> np.ndarray:
    """
    Inject a fault into a COPY of base_window using ADDITIVE multi-sensor
    injection with critical-sensor ceiling caps and flat fill.

    Non-critical sensors use a gradual RAMP (linspace) for visual realism.
    Critical sensors (Xs2/Xs3) use FLAT FILL (all 50 rows at target value)
    because the CNN-LSTM reads the entire 50-row window — a ramp averages
    out to a lower effective value and the model ignores it.

    Critical sensors are capped per-hit so degradation walks through the
    CNN-LSTM's sensitivity cliff in steps:
      Hit 1 → ONLINE (RUL ~70)    — Xs2/Xs3 capped at 0.35
      Hit 2 → DEGRADED (RUL ~25)  — Xs2/Xs3 capped at 0.48
      Hit 3 → OFFLINE (RUL ≤15)   — Xs2/Xs3 uncapped (0.98)

    The base_window carries accumulated damage from previous faults via
    factory_state._build_window() padding with h[-1] (latest reading).

    RAMP_ESCALATION controls how much each hit adds:
      delta = spike_value × intensity × RAMP_ESCALATION

    Args:
        base_window: (50, 18) float32 array — sensor readings in raw units
        spike:       validated SensorSpike object

    Returns:
        (50, 18) float32 array — copy with additive correlated injection
    """
    from dl_engine.inference import raw_value_for_scaled, get_scaler_ranges

    injected = base_window.copy()
    primary_col = SENSOR_TO_COL[spike.sensor_id]

    ranges = get_scaler_ranges()

    def _current_scaled(col: int, raw_val: float) -> float:
        """Convert raw sensor value to [0,1] scaled position."""
        lo  = float(ranges["min"][col])
        rng = float(ranges["range"][col])
        return (raw_val - lo) / rng if rng > 0 else 0.0

    # ── Primary sensor: additive ramp ─────────────────────────────────────
    raw_start      = float(injected[0, primary_col])
    current_scaled = _current_scaled(primary_col, raw_start)
    cap            = _get_critical_cap(spike.sensor_id, current_scaled)
    target_scaled  = min(cap, current_scaled + spike.spike_value * RAMP_ESCALATION)
    raw_end        = raw_value_for_scaled(primary_col, target_scaled)

    if spike.sensor_id in CRITICAL_SENSORS:
        # Flat fill: model reads all 50 rows equally
        injected[:, primary_col] = raw_end
    else:
        # Gradual ramp: visual realism for non-critical sensors
        ramp = np.linspace(raw_start, raw_end, 50).astype(np.float32)
        injected[:, primary_col] = ramp

    log.debug(
        "Spike inject: %s (col %d) %s %.1f → %.1f (scaled %.2f → %.2f, cap=%.2f)",
        spike.sensor_id, primary_col,
        "FLAT" if spike.sensor_id in CRITICAL_SENSORS else "RAMP",
        raw_start, raw_end, current_scaled, target_scaled, cap,
    )

    # ── Correlated sensors: additive injection ────────────────────────────
    correlations = SENSOR_CORRELATIONS.get(spike.sensor_id, [])
    for corr_sensor_id, intensity in correlations:
        corr_col       = SENSOR_TO_COL[corr_sensor_id]
        corr_start     = float(injected[0, corr_col])
        corr_current   = _current_scaled(corr_col, corr_start)
        corr_cap       = _get_critical_cap(corr_sensor_id, corr_current)
        corr_delta     = spike.spike_value * intensity * RAMP_ESCALATION
        corr_target    = min(corr_cap, corr_current + corr_delta)
        corr_end       = raw_value_for_scaled(corr_col, corr_target)

        if corr_sensor_id in CRITICAL_SENSORS:
            # Flat fill for critical sensors
            injected[:, corr_col] = corr_end
        else:
            # Gradual ramp for non-critical sensors
            corr_ramp = np.linspace(corr_start, corr_end, 50).astype(np.float32)
            injected[:, corr_col] = corr_ramp

        log.debug(
            "  + correlated %s (col %d) %s → scaled %.2f→%.2f (intensity %.0f%%, cap=%.2f)",
            corr_sensor_id, corr_col,
            "FLAT" if corr_sensor_id in CRITICAL_SENSORS else "RAMP",
            corr_current, corr_target, intensity * 100, corr_cap,
        )

    return injected


# ── Main entry point ───────────────────────────────────────────────────────────

def translate_fault_to_tensor(
    base_window: np.ndarray,
    user_text: str,
) -> tuple[np.ndarray, dict, bool]:
    """
    Agent 1 public interface. Called by agent_loop.py.

    Converts a plain-English fault description into a modified sensor tensor
    by asking Gemini to identify the affected sensor and spike parameters,
    then injecting those values into the base window.

    Args:
        base_window: (50, 18) float32 numpy array — current sensor baseline
        user_text:   professor's fault description

    Returns:
        injected_window: (50, 18) float32 numpy array with spike applied
        spike_dict:      SensorSpike fields as plain dict (for logging/UI)
        used_fallback:   True if Gemini failed and hardcoded fallback was used
    """
    # If client is None (no API key), skip Gemini entirely
    if client is None:
        log.warning("No API key — skipping Gemini, using deterministic fallback.")
        spike = _get_fallback(user_text)
        injected = _inject_spike(base_window, spike)
        return injected, spike.model_dump(), True

    spike: SensorSpike | None = None
    last_error: str = ""

    for attempt in range(MAX_RETRIES + 1):

        # ── Build the prompt ──────────────────────────────────────────────────
        # On retries: inject the specific validation error so Gemini learns
        # what went wrong and corrects it, rather than repeating the same mistake.
        if attempt == 0:
            prompt_contents = (
                f"{DIAGNOSTIC_SYSTEM_PROMPT}\n\n"
                f"Fault description: {user_text}"
            )
        else:
            prompt_contents = (
                f"{DIAGNOSTIC_SYSTEM_PROMPT}\n\n"
                f"Fault description: {user_text}\n\n"
                f"CORRECTION REQUIRED (attempt {attempt + 1} of {MAX_RETRIES + 1}):\n"
                f"Your previous response was rejected for this reason: {last_error}\n"
                f"Please fix this specific issue and return a corrected response."
            )

        # ── Call Gemini ───────────────────────────────────────────────────────
        try:
            t_call = time.time()
            log.info("Gemini call attempt %d/%d  model=gemini-2.5-flash", attempt + 1, MAX_RETRIES + 1)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SensorSpike,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            api_ms = round((time.time() - t_call) * 1000, 1)
            log.info("Gemini responded in %.0fms", api_ms)

            candidate = SensorSpike.model_validate_json(response.text)

            is_valid, error = _validate_domain(candidate)
            if is_valid:
                spike = candidate
                log.info(
                    "✓ Attempt %d ACCEPTED: sensor=%s  value=%.2f  severity=%s  positions=%s",
                    attempt + 1, spike.sensor_id, spike.spike_value,
                    spike.fault_severity, spike.affected_window_positions,
                )
                break
            else:
                last_error = error
                log.warning("✗ Attempt %d domain validation fail: %s", attempt + 1, error)

        except ValidationError as e:
            last_error = f"Pydantic validation error: {e}"
            log.warning("✗ Attempt %d Pydantic fail: %s", attempt + 1, e)

        except Exception as e:
            last_error = f"API error: {e}"
            log.error("✗ Attempt %d API fail: %s", attempt + 1, e)

    # ── Fallback if all attempts failed ───────────────────────────────────────
    used_fallback = False
    if spike is None:
        log.warning("All %d Gemini attempts failed. Using deterministic fallback.", MAX_RETRIES + 1)
        spike = _get_fallback(user_text)
        used_fallback = True

    # ── Inject into tensor ────────────────────────────────────────────────────
    injected = _inject_spike(base_window, spike)

    return injected, spike.model_dump(), used_fallback
