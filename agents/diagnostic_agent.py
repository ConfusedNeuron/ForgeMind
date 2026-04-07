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
    raise ValueError("GEMINI_API_KEY_DIAGNOSTIC not found in environment")

print(f"Loaded API key for DIAGNOSTIC agent: {bool(api_key)}") # comment during production
#logging.basicConfig(level=logging.INFO) #uncomment during production
#logging.info("Using GEMINI_API_KEY_DIAGNOSTIC") #uncomment during production
# Initialize client
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
# Intensity hierarchy:
#   0.80–0.90 = direct thermal/pressure fault (strongest RUL impact)
#   0.55–0.70 = mechanically coupled fault (moderate impact)
#   0.35–0.50 = indirect/operating-condition fault (mild impact)
#
# Format: primary_sensor → [(correlated_sensor, intensity_fraction), ...]
SENSOR_CORRELATIONS: dict[str, list[tuple[str, float]]] = {
    # Temperature faults → directly stress key degradation sensors
    "Xs4":  [("Xs2", 0.85), ("Xs3", 0.78)],
    "Xs5":  [("Xs2", 0.82), ("Xs3", 0.75)],
    # Pressure faults → co-located thermal stress
    "Xs2":  [("Xs3", 0.82), ("Xs6", 0.65)],
    "Xs6":  [("Xs2", 0.78), ("Xs3", 0.72)],
    # Bearing/fan faults → friction heat propagates to thermal sensors
    "Xs0":  [("Xs2", 0.75), ("Xs3", 0.68), ("Xs1", 0.70)],
    "Xs1":  [("Xs2", 0.75), ("Xs3", 0.68), ("Xs0", 0.70)],
    # Vibration/enthalpy → mechanical stress raises temps
    "Xs7":  [("Xs2", 0.78), ("Xs3", 0.70), ("Xs0", 0.45)],
    # Speed/RPM faults → off-design operation strains thermal path
    "Xs8":  [("Xs2", 0.72), ("Xs3", 0.65), ("Xs9", 0.70)],
    "Xs9":  [("Xs2", 0.72), ("Xs3", 0.65), ("Xs8", 0.70)],
    "Xs10": [("Xs2", 0.68), ("Xs3", 0.62), ("Xs4", 0.55)],
    "Xs13": [("Xs2", 0.68), ("Xs3", 0.62), ("Xs4", 0.50)],
    # Coolant/bleed faults → reduced cooling raises degradation temps
    "Xs12": [("Xs2", 0.75), ("Xs3", 0.68), ("Xs11", 0.60)],
    "Xs11": [("Xs2", 0.72), ("Xs3", 0.65), ("Xs12", 0.55)],
    # Operating condition faults → affect thermal equilibrium
    "W0":   [("Xs2", 0.70), ("Xs3", 0.62), ("W2", 0.40)],
    "W1":   [("Xs2", 0.65), ("Xs3", 0.58)],
    "W2":   [("Xs2", 0.68), ("Xs3", 0.60)],
    "W3":   [("Xs2", 0.70), ("Xs3", 0.62)],
}


def _inject_spike(base_window: np.ndarray, spike: SensorSpike) -> np.ndarray:
    """
    Inject a fault into a COPY of base_window using a physics-informed
    multi-sensor RAMP pattern.

    Model probing revealed:
      - Single-sensor step spikes barely move RUL (~0.5 cycle change)
      - 3-sensor ramps across the full window produce dramatic drops
        (71 → 8 RUL for a 3-sensor ramp from 0.1 → 0.95)

    Strategy:
      1. PRIMARY sensor: ramp from its current baseline value to the
         spike_value (in raw units) across the full 50-step window.
      2. CORRELATED sensors: same ramp but scaled by an intensity
         fraction (e.g. 0.70 × spike_value).

    This mimics the gradual multi-sensor degradation patterns the
    CNN-LSTM was trained on (N-CMAPSS turbofan data).

    Args:
        base_window: (50, 18) float32 array — sensor readings in raw units
        spike:       validated SensorSpike object

    Returns:
        (50, 18) float32 array — copy with correlated ramp injected
    """
    from dl_engine.inference import raw_value_for_scaled

    injected = base_window.copy()
    primary_col = SENSOR_TO_COL[spike.sensor_id]

    # ── Primary sensor: full ramp ─────────────────────────────────────────
    raw_start = float(injected[0, primary_col])           # current baseline
    raw_end   = raw_value_for_scaled(primary_col, spike.spike_value)
    ramp = np.linspace(raw_start, raw_end, 50).astype(np.float32)
    injected[:, primary_col] = ramp

    log.debug(
        "Spike inject: %s (col %d) ramp %.1f → %.1f (scaled %.2f → %.2f)",
        spike.sensor_id, primary_col, raw_start, raw_end, 0.10, spike.spike_value,
    )

    # ── Correlated sensors: scaled ramp ───────────────────────────────────
    correlations = SENSOR_CORRELATIONS.get(spike.sensor_id, [])
    for corr_sensor_id, intensity in correlations:
        corr_col = SENSOR_TO_COL[corr_sensor_id]
        corr_start = float(injected[0, corr_col])
        corr_target_scaled = spike.spike_value * intensity
        corr_end = raw_value_for_scaled(corr_col, corr_target_scaled)
        corr_ramp = np.linspace(corr_start, corr_end, 50).astype(np.float32)
        injected[:, corr_col] = corr_ramp

        log.debug(
            "  + correlated %s (col %d) ramp → scaled %.2f (intensity %.0f%%)",
            corr_sensor_id, corr_col, corr_target_scaled, intensity * 100,
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
