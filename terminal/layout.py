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
# PURE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def mini_sparkline(values: list, width: int = 20) -> str:
    """Render a sparkline using Unicode block characters."""
    if not values:
        return "─" * width
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx  = min(values), max(values)
    rng     = mx - mn if mx > mn else 1.0
    vals    = values[-width:]
    chars   = []
    for v in vals:
        idx = int(((v - mn) / rng) * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


def status_bar(status: str) -> str:
    """Visual bar for machine status."""
    return {
        "ONLINE":   "████████",
        "DEGRADED": "████░░░░",
        "OFFLINE":  "░░░░░░░░",
    }.get(status, "????????")


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

        # ── Prediction reliability ────────────────────────────────────────────
        rel_label, rel_color = compute_prediction_reliability(
            state.rul_history.get(mid, [])
        )

        # ── Sensor saturation check ───────────────────────────────────────────
        machine_sensor_history = state.per_machine_sensor_history.get(
            mid, [[] for _ in range(18)]
        )
        saturated_sensors = check_sensor_saturation(machine_sensor_history)
        saturated_names   = {name for name, _ in saturated_sensors}

        # ── Build lines ───────────────────────────────────────────────────────
        lines = [
            f"[bold]LIVE SENSOR FEED — {machine.name}[/bold]",
            "",
            f"  RUL: [{rc}]{machine.rul:.0f} cycles ▼ {rl}[/{rc}]",
            f"  Reliability: [{rel_color}]{rel_label}[/{rel_color}]",
            "",
        ]

        sensor_names = (
            [f"W{i}" for i in range(4)] +
            [f"Xs{i}" for i in range(14)]
        )
        for i, name in enumerate(sensor_names):
            history = machine_sensor_history[i] if machine_sensor_history[i] else state.sensor_history[i]
            if history:
                spark = mini_sparkline(history[-20:])
                # Mark saturated sensors with a ⚠ tag
                if name in saturated_names:
                    lines.append(
                        f"  {name:>4s} [{rul_color(0)}]{spark}[/{rul_color(0)}]"
                        f" [bold red]⚠ DATA QUALITY[/bold red]"
                    )
                else:
                    lines.append(f"  {name:>4s} {spark}")
            else:
                lines.append(f"  {name:>4s} ────────────────────")

        # Show summary of saturation alerts if any
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
            bar   = status_bar(machine.status)
            color = status_color(machine.status)
            lines.append(
                f"  Machine {mid}: [{color}]{bar} {machine.status:>8s}[/{color}]"
                f"  RUL: {machine.rul:.0f}"
            )

        # ── SECTION 3: Capacity metrics ───────────────────────────────────────
        lines.append("")
        risk_flag = " [bold red]⚠ CRITICAL[/bold red]" if state.breakeven_risk else ""
        lines.append(
            f"  Capacity: {state.capacity_pct:.0f}% | "
            f"ΣPD/T: {state.machine_req:.2f}{risk_flag}"
        )

        # ── SECTION 4: Maintenance queue ─────────────────────────────────────
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
            # Only show if at least one machine has a non-stable trend
            non_stable = [
                m for m in state.degradation_leaderboard
                if m["trend_label"] != "STABLE →"
            ]
            if non_stable:
                lines.append("")
                lines.append(f"  [bold]── DEGRADATION RATE ────────────────────[/bold]")
                for m in state.degradation_leaderboard[:3]:  # Top 3
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
    "System":           "dim",
    "Chaos Engine":     "bold magenta",
    "Diagnostic Agent": "cyan",
    "DL Oracle":        "bold green",
    "Capacity Agent":   "yellow",
    "Floor Manager":    "bold red",
    "Ops Alert":        "bold red",      # ← new: cliff detection, saturation
}


def format_log_entry(agent: str, message: str) -> str:
    """Format a comms log entry with timestamp and color."""
    ts    = datetime.now().strftime("%H:%M:%S")
    color = AGENT_COLORS.get(agent, "white")
    return f"[dim]{ts}[/dim] [{color}][{agent}][/{color}] {message}"
