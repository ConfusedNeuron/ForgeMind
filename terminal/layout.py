"""
Widget classes for the 4-pane factory dashboard.

Pane 1: SensorFeedWidget   — Live sparklines + RUL + Prediction Reliability
Pane 2: CapacityWidget     — Machines + Shift Health Banner + Maintenance Queue
Pane 3: CommsLogWidget     — Agent comms log (handled in app.py via RichLog)
Pane 4: ChaosInputWidget   — Text input (handled in app.py via Input)

New in this version:
  - SensorFeedWidget: shows Prediction Reliability score (HIGH/MEDIUM/LOW)
  - SensorFeedWidget: marks saturated sensors with ⚠ DATA QUALITY tag
  - CapacityWidget:   shows SHIFT STATUS banner (ops health summary)
  - CapacityWidget:   shows MAINTENANCE QUEUE (ranked action list)
  - CapacityWidget:   shows DEGRADATION LEADERBOARD (fastest declining machines)
"""

from textual.widgets import Static
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS-FRIENDLY SENSOR NAMES
# ─────────────────────────────────────────────────────────────────────────────
# Maps the 18 N-CMAPSS tensor columns to names a factory manager understands.
# Internal pipeline still uses W0/Xs0 etc — this is display-only.
# Order matches tensor column order: [W0, W1, W2, W3, Xs0, Xs1, ... Xs13]

SENSOR_DISPLAY_NAMES = [
    "Motor RPM",       # W0   (col 0)  — drive speed
    "Feed Rate",       # W1   (col 1)  — material feed
    "Power kW",        # W2   (col 2)  — electrical draw
    "Coolant Flow",    # W3   (col 3)  — cooling rate
    "Vibration X",     # Xs0  (col 4)  — X-axis vibration
    "Vibration Y",     # Xs1  (col 5)  — Y-axis vibration
    "Bearing Temp",    # Xs2  (col 6)  — KEY degradation sensor
    "Motor Temp",      # Xs3  (col 7)  — KEY degradation sensor
    "Oil Pressure",    # Xs4  (col 8)  — lubricant pressure
    "Oil Temp",        # Xs5  (col 9)  — lubricant temperature
    "Spindle Load",    # Xs6  (col 10) — tool load %
    "Torque",          # Xs7  (col 11) — applied torque
    "Hydraulic PSI",   # Xs8  (col 12) — hydraulic pressure
    "Coolant Temp",    # Xs9  (col 13) — coolant temperature
    "Ambient Temp",    # Xs10 (col 14) — environment temp
    "Current Amps",    # Xs11 (col 15) — electrical current
    "Acoustic dB",     # Xs12 (col 16) — noise level
    "Cycle Time",      # Xs13 (col 17) — time per cycle
]


# ─────────────────────────────────────────────────────────────────────────────
# PURE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mini_sparkline(values: list, width: int = 20) -> str:
    """
    Render a sparkline using Unicode block characters.
    Always returns exactly `width` characters — left-pads with spaces
    when fewer than `width` values are available.
    """
    if not values:
        return "─" * width
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng    = mx - mn if mx > mn else 1.0
    vals   = values[-width:]
    chars  = []
    for v in vals:
        idx = int(((v - mn) / rng) * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        chars.append(blocks[idx])
    # FIX: left-pad with spaces so output is always exactly `width` chars
    while len(chars) < width:
        chars.insert(0, " ")
    return "".join(chars)


def status_bar(status: str) -> str:
    """Visual bar for machine status."""
    return {
        "ONLINE":   "████████",
        "DEGRADED": "████░░░░",
        "OFFLINE":  "░░░░░░░░",
    }.get(status, "????????")


def rul_bar(rul: float, width: int = 20) -> tuple:
    """
    Proportional fill bar mapped to 0–100 RUL scale.

    Returns (bar_string, color_string) where:
      - bar_string is exactly `width` chars of █ / ░
      - color_string is 'green', 'yellow', or 'red'

    Color thresholds:
      RUL > 30  → green   ████████████████████
      RUL 15-30 → yellow  ██████████░░░░░░░░░░
      RUL ≤ 15  → red     ████░░░░░░░░░░░░░░░░
    """
    filled = max(0, min(width, int(rul / 100.0 * width)))
    empty  = width - filled
    bar    = "█" * filled + "░" * empty
    if rul > 30:
        color = "green"
    elif rul > 15:
        color = "yellow"
    else:
        color = "red"
    return bar, color


def status_color(status: str) -> str:
    """Color for machine status."""
    return {
        "ONLINE":   "green",
        "DEGRADED": "yellow",
        "OFFLINE":  "red",
    }.get(status, "white")


def rul_color(rul: float) -> str:
    """Color based on RUL value."""
    if rul > 30:
        return "green"
    elif rul > 15:
        return "yellow"
    else:
        return "red"


def rul_label(rul: float) -> str:
    """Text label based on RUL value."""
    if rul > 30:
        return "HEALTHY"
    elif rul > 15:
        return "WARNING"
    else:
        return "CRITICAL"


def divider(width: int = 38, char: str = "─") -> str:
    """Return a thin divider line."""
    return char * width


# ─────────────────────────────────────────────────────────────────────────────
# PANE 1: SENSOR FEED WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class SensorFeedWidget(Static):
    """
    Live sensor feed with sparklines, RUL indicator, and prediction reliability.

    New sections:
      - Prediction Reliability: HIGH / MEDIUM / LOW  (below the RUL line)
      - Per-sensor ⚠ DATA QUALITY tag for saturated sensors
    """

    def refresh_content(self, state) -> None:
        """Rebuild sensor pane from factory state."""
        from .ops_analytics import compute_prediction_reliability, check_sensor_saturation

        mid     = state.active_machine_id
        machine = state.machines[mid]
        rc      = rul_color(machine.rul)
        rl      = rul_label(machine.rul)

        rel_label, rel_color = compute_prediction_reliability(
            state.rul_history.get(mid, [])
        )

        machine_sensor_history = state.per_machine_sensor_history.get(
            mid, [[] for _ in range(18)]
        )
        # Saturation thresholds (0.97 / 0.03) live in the scaled [0, 1] domain;
        # the per-machine buffer stores RAW values for the DL oracle, so scale
        # before calling the saturation checker.
        scaled_sensor_history = state.get_scaled_machine_sensor_history(mid)
        saturated_sensors = check_sensor_saturation(scaled_sensor_history)
        saturated_names   = {name for name, _ in saturated_sensors}

        lines = [
            f"[bold]LIVE SENSOR FEED — {machine.name}[/bold]",
            "",
            f"  RUL: [{rc}]{machine.rul:.0f} cycles ▼ {rl}[/{rc}]",
            f"  Reliability: [{rel_color}]{rel_label}[/{rel_color}]",
            "",
        ]

        for i, name in enumerate(SENSOR_DISPLAY_NAMES):
            history = machine_sensor_history[i]
            if history:
                spark = mini_sparkline(history[-20:])
                if name in saturated_names:
                    lines.append(
                        f"  {name:>14s} [{rul_color(0)}]{spark}[/{rul_color(0)}]"
                        f" [bold red]⚠[/bold red]"
                    )
                else:
                    lines.append(f"  {name:>14s} {spark}")
            else:
                lines.append(f"  {name:>14s} {mini_sparkline([])}")

        if saturated_sensors:
            lines.append("")
            lines.append(
                f"  [bold yellow]⚠ {len(saturated_sensors)} sensor(s) unreliable"
                f" — predictions may be inaccurate[/bold yellow]"
            )

        self.update("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# PANE 2: CAPACITY WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class CapacityWidget(Static):
    """
    Factory capacity dashboard — redesigned for operations managers.

    Sections (top to bottom):
      1. SHIFT STATUS banner  — one-line factory health (ops language)
      2. Machine bars         — 5 machines with status + RUL
      3. Capacity metric      — existing ΣPD/T line
      4. MAINTENANCE QUEUE    — ranked action list for the ops manager
      5. DEGRADATION BOARD    — which machine is declining fastest
    """

    def refresh_content(self, state) -> None:
        """Rebuild capacity pane from factory state."""
        lines = [
            "[bold]FACTORY CAPACITY DASHBOARD[/bold]",
            "",
        ]

        # ── SECTION 1: Shift health banner ────────────────────────────────────
        health_text, health_color = state.shift_health
        lines.append(f"  [{health_color}]◉ SHIFT STATUS: {health_text}[/{health_color}]")
        lines.append(f"  {divider(42)}")
        lines.append("")

        # ── SECTION 2: Machine bars ───────────────────────────────────────────
        for mid, machine in state.machines.items():
            bar, bar_color = rul_bar(machine.rul)
            if machine.status == "OFFLINE":
                status_markup = "[blink bold red]OFFLINE[/blink bold red]"
            elif machine.status == "DEGRADED":
                status_markup = "[yellow]DEGRADED[/yellow]"
            else:
                status_markup = "[green]ONLINE[/green]"
            lines.append(
                f"  {machine.name:<16s} [{bar_color}]{bar}[/{bar_color}]"
                f" {status_markup}"
                f"  RUL: [{bar_color}]{machine.rul:.0f}[/{bar_color}]"
            )

        # ── SECTION 3: Capacity metrics ───────────────────────────────────────
        lines.append("")
        spdt_color = "bold red" if state.breakeven_risk else "bold green"
        risk_flag  = " [bold red]⚠ CRITICAL[/bold red]" if state.breakeven_risk else ""
        lines.append(
            f"  Capacity: {state.capacity_pct:.0f}% | "
            f"ΣPD/T: [{spdt_color}]{state.machine_req:.2f}[/{spdt_color}]{risk_flag}"
        )

        # ── SECTION 4: Maintenance queue ──────────────────────────────────────
        if state.maintenance_schedule:
            lines.append("")
            lines.append(f"  [bold]── MAINTENANCE QUEUE ──────────────────[/bold]")
            for item in state.maintenance_schedule:
                color = item["color"]
                lines.append(
                    f"  [dim]#{item['rank']}[/dim]  "
                    f"[{color}]{item['machine_name']:<14s}[/{color}]"
                    f"  [{color}]{item['action']:<14s}[/{color}]"
                    f"  [dim]{item['urgency']}[/dim]"
                )
        else:
            lines.append("")
            lines.append("  [dim]── MAINTENANCE QUEUE  ──  All machines nominal ──[/dim]")

        # ── SECTION 5: Degradation leaderboard ───────────────────────────────
        if state.degradation_leaderboard:
            non_stable = [
                m for m in state.degradation_leaderboard
                if m["trend_label"] != "STABLE →"
            ]
            if non_stable:
                lines.append("")
                lines.append(f"  [bold]── DEGRADATION RATE ────────────────────[/bold]")
                for m in state.degradation_leaderboard[:3]:
                    tc = m["trend_color"]
                    lines.append(
                        f"  [{tc}]{m['machine_name']:<14s}[/{tc}]"
                        f"  [{tc}]{m['trend_label']:<14s}[/{tc}]"
                        f"  RUL: {m['rul']:.0f}"
                    )

        self.update("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# COMMS LOG FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "System":           "dim white",
    "Chaos Engine":     "bold magenta",
    "Diagnostic Agent": "bold cyan",
    "DL Oracle":        "bold yellow",
    "Capacity Agent":   "white",
    "Floor Manager":    "bold green",
    "Ops Alert":        "bold red",
}


def format_log_entry(agent: str, message: str) -> str:
    """Format a comms log entry with timestamp and color."""
    ts    = datetime.now().strftime("%H:%M:%S")
    color = AGENT_COLORS.get(agent, "white")
    return f"[dim]{ts}[/dim] [{color}][{agent}][/{color}] {message}"