# agents/input_guard.py
# Pure Python — no LLM, no network, no external packages.
# Rejects garbage before it wastes a Gemini API call.

VALID_KEYWORDS = {
    # Machine component nouns
    "machine", "cnc", "press", "lathe", "mill", "compressor", "motor", "pump",
    "valve", "filter", "shaft", "rotor", "stator", "gearbox", "bearing", "turbine",
    "actuator", "conveyor", "spindle", "piston",
    # Sensor measurement nouns
    "temperature", "temp", "pressure", "vibration", "rpm", "speed", "sensor",
    "coolant", "lubrication", "friction", "flow", "torque", "voltage", "current",
    "frequency", "amplitude", "noise", "heat",
    # Fault event verbs and nouns
    "surge", "spike", "leak", "overheat", "overload", "failure", "fault", "wear",
    "crack", "misalignment", "imbalance", "corrosion", "degradation", "malfunction",
    "shutdown", "stall", "jam", "rupture", "erosion", "fatigue", "wobble",
    "oscillation", "fluctuation",
}

def is_valid_fault_input(user_text: str) -> tuple[bool, str]:
    """
    Fast keyword check. No LLM call. No network.

    Returns:
        (True, "")            — input looks like a valid fault description
        (False, reason_str)   — input should be rejected, reason shown to user
    """
    text = user_text.strip()

    if len(text) < 5:
        return False, (
            "Input too short. Describe a machine fault "
            "e.g., 'bearing temperature spike on Machine 4'."
        )

    if len(text) > 500:
        return False, "Input too long. Keep fault descriptions under 500 characters."

    words = set(text.lower().split())
    if not words.intersection(VALID_KEYWORDS):
        return False, (
            "Unrecognized fault type. Try something like: "
            "'high pressure in compressor', "
            "'bearing overheat on Machine 3', or "
            "'vibration spike on CNC-Alpha'."
        )

    return True, ""
