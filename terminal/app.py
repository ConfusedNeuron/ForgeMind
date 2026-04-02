"""
ForgeMind Factory Floor Dashboard — Main Application

4-pane Textual app:
    Pane 1: Live sensor feed with RUL + Prediction Reliability
    Pane 2: Capacity dashboard — Shift Status + Maintenance Queue
    Pane 3: Agent comms log
    Pane 4: Chaos engine input

New ops features wired in this version:
  - Sudden RUL Cliff Detection (1.4.2)   → logs 🚨 EMERGENCY to comms pane
  - Prediction Reliability (1.1.3)       → rendered in SensorFeedWidget
  - Predictive Maintenance Schedule (1.2.2) → rendered in CapacityWidget
  - Sensor Saturation Warnings (1.4.1)   → rendered in SensorFeedWidget + comms
  - Shift Health Banner (ops rec A)      → rendered in CapacityWidget
  - Degradation Leaderboard (ops rec B)  → rendered in CapacityWidget

Run: python -m terminal.app
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.binding import Binding
from textual import work

import re
import numpy as np

from .factory_state import FactoryState
from .layout import SensorFeedWidget, CapacityWidget, format_log_entry
from .dummy_oracle import predict_rul, reset_call_count

# ── Import ops analytics ──────────────────────────────────────────────────────
from .ops_analytics import (
    detect_rul_cliff,
    check_sensor_saturation,
    compute_maintenance_schedule,
    compute_shift_health,
    compute_degradation_leaderboard,
)


class FactoryApp(App):
    """Predictive maintenance terminal dashboard."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 1fr 3;
    }

    #sensor-pane {
        column-span: 1;
        row-span: 1;
        border: solid green;
        padding: 1;
    }

    #capacity-pane {
        column-span: 1;
        row-span: 1;
        border: solid blue;
        padding: 1;
    }

    #comms-pane {
        column-span: 2;
        row-span: 1;
        border: solid yellow;
        padding: 1;
    }

    #chaos-input {
        column-span: 2;
        row-span: 1;
        dock: bottom;
    }

    .status-online   { color: green; }
    .status-degraded { color: yellow; }
    .status-offline  { color: red; }

    .rul-healthy  { color: green; }
    .rul-warning  { color: yellow; }
    .rul-critical { color: red; }
    """

    BINDINGS = [
        Binding("ctrl+r", "reset_factory", "Reset All Machines"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.state = FactoryState()
        self._machine_cycle = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SensorFeedWidget(id="sensor-pane")
        yield CapacityWidget(id="capacity-pane")
        yield RichLog(id="comms-pane", highlight=True, markup=True)
        yield Input(
            id="chaos-input",
            placeholder="CHAOS ENGINE > Type a fault description and press Enter...",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize display on startup."""
        self._refresh_ops_analytics()   # compute initial (all-nominal) analytics
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory simulation online. All machines nominal.")
        self._log("System", "Type a fault description below to inject chaos.")
        self._log("System", "Keyboard: Ctrl+R = Reset | Ctrl+Q = Quit")

    # ═════════════════════════════════════════════════════════════════════════
    # INPUT HANDLER
    # ═════════════════════════════════════════════════════════════════════════

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chaos engine input — non-blocking."""
        user_text = event.value.strip()
        event.input.value = ""

        if not user_text:
            return

        self._log("Chaos Engine", f"Injecting: {user_text}")
        self._run_chaos(user_text)

    # ═════════════════════════════════════════════════════════════════════════
    # BACKGROUND WORKER — THE FULL PIPELINE
    # ═════════════════════════════════════════════════════════════════════════

    @work(thread=True)
    def _run_chaos(self, user_text: str) -> None:
        """
        Background worker — runs the full processing pipeline.

        Flow:
          1. Extract machine ID from text
          2. Record old RUL (needed for cliff detection)
          3. Simulate sensor injection for this machine
          4. Build sensor window, predict RUL
          5. Build result → _process_result → update state
          6. Run ops analytics (cliff, saturation, maintenance, shift health)
          7. Refresh both panes
        """
        machine_id = self._extract_machine_id(user_text)

        # ── Step 2: Snapshot old RUL before the oracle runs ──────────────────
        old_rul = self.state.machines[machine_id].rul

        # ── Step 3: Simulate sensor injection for this machine ───────────────
        # In production this comes from the diagnostic agent's modified window.
        # Here we generate a realistic fault reading so that:
        #   - Sparklines show something interesting
        #   - Saturation detection has real data to work with
        fault_reading = self._simulate_fault_reading(user_text)
        self.state.push_machine_sensor_reading(machine_id, fault_reading)

        # ── Step 4: Build sensor window and predict RUL ───────────────────────
        base_window = self.state.get_machine_sensor_window(machine_id)
        rul = predict_rul(base_window)

        # Guard: DL model or stub can return NaN, inf, or negative on bad input.
        # Clamp to a safe range rather than letting it corrupt state or crash.
        import math
        if not math.isfinite(rul) or rul < 0:
            rul = 0.0
        rul = min(rul, 9999.0)

        # ── Update active display machine so sensor pane follows the chaos ────
        self.state.active_machine_id = machine_id

        # ── Determine machine status ──────────────────────────────────────────
        if rul <= 15:
            new_status = "OFFLINE"
        elif rul <= 30:
            new_status = "DEGRADED"
        else:
            new_status = "ONLINE"

        result = {
            "valid": True,
            "rejection_reason": "",
            "spike": {
                "sensor_id":                 "Xs4",
                "spike_value":               float(np.max(fault_reading[4:9])),
                "affected_window_positions": [45, 46, 47, 48, 49],
                "fault_severity":            "HIGH" if rul <= 15 else ("MEDIUM" if rul <= 30 else "LOW"),
                "plain_english_summary":     "Sensor spike detected — fault signature matched.",
            },
            "rul": rul,
            "capacity_report": {
                "machine_id":    machine_id,
                "machine_name":  self.state.machines[machine_id].name,
                "status":        new_status,
                "rul":           rul,
                "capacity_pct":  self._estimate_capacity(machine_id, new_status),
                "machine_req":   self.state.machine_req,
                "breakeven_risk": new_status in ("OFFLINE", "DEGRADED"),
            },
            "dispatch_orders": self._build_dispatch_order(machine_id, new_status, rul),
            "machine_statuses": [
                {
                    "id":             mid,
                    "name":           m.name,
                    "status":         new_status if mid == machine_id else m.status,
                    "rul":            rul if mid == machine_id else m.rul,
                    "available_time": (
                        0.0 if (mid == machine_id and new_status == "OFFLINE")
                        else m.base_time * 0.5 if (mid == machine_id and new_status == "DEGRADED")
                        else m.available_time
                    ),
                    "base_time": m.base_time,
                }
                for mid, m in self.state.machines.items()
            ],
            "used_fallback": False,
        }

        # ── Step 5: Standard result processing ───────────────────────────────
        self._process_result(result, machine_id, old_rul)

    def _process_result(self, result: dict, machine_id: int, old_rul: float) -> None:
        """Process pipeline result: log, update state, run ops analytics, refresh."""

        if not result["valid"]:
            self.call_from_thread(
                self._log, "Chaos Engine", f"REJECTED: {result['rejection_reason']}"
            )
            return

        # ── Log standard agent outputs ────────────────────────────────────────
        spike = result["spike"]
        self.call_from_thread(
            self._log, "Diagnostic Agent",
            f"Sensor {spike['sensor_id']} spike to {spike['spike_value']:.2f} "
            f"({spike['fault_severity']}) — {spike['plain_english_summary']}"
        )

        new_rul = result["rul"]
        self.call_from_thread(
            self._log, "DL Oracle",
            f"RUL updated: {old_rul:.0f} → {new_rul:.1f} cycles"
        )

        cap = result["capacity_report"]
        self.call_from_thread(
            self._log, "Capacity Agent",
            f"Machine {cap['machine_id']} {cap['status']}. "
            f"Capacity: {cap['capacity_pct']:.0f}%. ΣPD/T: {cap['machine_req']:.2f}"
        )

        self.call_from_thread(
            self._log, "Floor Manager", result["dispatch_orders"]
        )

        if result["used_fallback"]:
            self.call_from_thread(
                self._log, "System",
                "[bold yellow]⚠ Running in OFFLINE MODE — using cached responses[/bold yellow]"
            )

        # ── Update factory state ──────────────────────────────────────────────
        self.state.update_from_agent_result(result)

        # ── Run ops analytics ─────────────────────────────────────────────────
        self._run_ops_analytics(machine_id, old_rul, new_rul)

        # ── Refresh UI ────────────────────────────────────────────────────────
        self.call_from_thread(self._refresh_sensor_pane)
        self.call_from_thread(self._refresh_capacity_pane)

    # ═════════════════════════════════════════════════════════════════════════
    # OPS ANALYTICS  (all 5 features + 2 recommended)
    # ═════════════════════════════════════════════════════════════════════════

    def _run_ops_analytics(self, machine_id: int, old_rul: float, new_rul: float) -> None:
        """
        Run all ops analytics after each chaos cycle and log any alerts.
        Updates state.maintenance_schedule, shift_health, degradation_leaderboard.

        Called from the background thread; all logging uses call_from_thread.
        """

        # ── Feature 1.4.2: Sudden RUL Cliff Detection ────────────────────────
        if detect_rul_cliff(old_rul, new_rul, threshold=0.40):
            drop_pct = int((old_rul - new_rul) / old_rul * 100)
            machine_name = self.state.machines[machine_id].name
            self.call_from_thread(
                self._log, "Ops Alert",
                f"[bold red]🚨 EMERGENCY: {machine_name} sudden degradation "
                f"({drop_pct}% drop: {old_rul:.0f} → {new_rul:.1f} cycles). "
                f"Dispatch maintenance NOW.[/bold red]"
            )

        # ── Feature 1.4.1: Sensor Saturation Warnings ────────────────────────
        machine_sensor_hist = self.state.per_machine_sensor_history.get(
            machine_id, [[] for _ in range(18)]
        )
        saturated = check_sensor_saturation(machine_sensor_hist, n_consecutive=5)
        if saturated:
            names_str = ", ".join(
                f"{n}({'↑MAX' if v == 'MAX' else '↓ZERO'})" for n, v in saturated
            )
            self.call_from_thread(
                self._log, "Ops Alert",
                f"[yellow]⚠ DATA QUALITY: {self.state.machines[machine_id].name} — "
                f"sensor(s) reading unreliable: {names_str}. "
                f"RUL prediction may be inaccurate.[/yellow]"
            )

        # ── Features 1.2.2 + Ops Rec A + Ops Rec B: Update analytics state ───
        self._refresh_ops_analytics()

    def _refresh_ops_analytics(self) -> None:
        """
        (Re)compute all derived ops state from current machine states.
        Safe to call from any thread — only writes to self.state fields.
        """
        # 1.2.2  Maintenance schedule
        self.state.maintenance_schedule = compute_maintenance_schedule(
            self.state.machines
        )

        # Ops Rec A: Shift health banner
        self.state.shift_health = compute_shift_health(
            self.state.machines, self.state.capacity_pct
        )

        # Ops Rec B: Degradation leaderboard
        self.state.degradation_leaderboard = compute_degradation_leaderboard(
            self.state.machines, self.state.rul_history
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PANE REFRESH
    # ═════════════════════════════════════════════════════════════════════════

    def _refresh_sensor_pane(self) -> None:
        """Update Pane 1: Live Sensor Feed."""
        pane = self.query_one("#sensor-pane", SensorFeedWidget)
        pane.refresh_content(self.state)

    def _refresh_capacity_pane(self) -> None:
        """Update Pane 2: Factory Capacity Dashboard."""
        pane = self.query_one("#capacity-pane", CapacityWidget)
        pane.refresh_content(self.state)

    # ═════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═════════════════════════════════════════════════════════════════════════

    def _log(self, agent: str, message: str) -> None:
        """Write to comms log pane (Pane 3)."""
        log_widget = self.query_one("#comms-pane", RichLog)
        formatted  = format_log_entry(agent, message)
        log_widget.write(formatted)
        self.state.add_log_entry(agent, message)

    # ═════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _extract_machine_id(self, text: str) -> int:
        """
        Extract machine ID from user input.
        Looks for 'Machine N' or machine names. Falls back to cycling 1–5.
        """
        match = re.search(r'[Mm]achine\s+(\d)', text)
        if match:
            mid = int(match.group(1))
            if 1 <= mid <= 5:
                return mid

        name_map = {
            "alpha":       1, "cnc-alpha":    1,
            "beta":        2, "cnc-beta":     2,
            "gamma":       3, "press-gamma":  3,
            "delta":       4, "lathe-delta":  4,
            "epsilon":     5, "mill-epsilon": 5,
        }
        text_lower = text.lower()
        for name, mid in name_map.items():
            if name in text_lower:
                return mid

        self._machine_cycle = (self._machine_cycle % 5) + 1
        return self._machine_cycle

    def _simulate_fault_reading(self, user_text: str) -> np.ndarray:
        """
        Generate a realistic fault sensor reading based on user's fault description.

        This runs in stub mode (Days 1–9) to:
          1. Make sparklines look realistic (not flat dashes)
          2. Allow saturation detection to work if severity is extreme
          3. Provide varied data so prediction reliability can be computed

        On Day 10+ with the real agent loop, the diagnostic agent returns the
        actual modified sensor window — this method becomes unused for machine
        sensor injection (but kept as fallback).

        Fault keyword → which sensor group spikes:
          temperature/heat/thermal → Xs4, Xs5  (thermal sensors)
          vibration/bearing        → Xs0, Xs1   (vibration sensors)
          pressure/hydraulic       → Xs8, Xs9   (pressure sensors)
          electric/power           → W0, W1     (operating conditions)
          (default)                → Xs3, Xs4   (general physical sensors)
        """
        text_lower = user_text.lower()

        # Baseline: normal operating range
        reading = np.random.uniform(0.3, 0.6, size=18).astype(np.float32)

        # Determine fault severity from keywords
        if any(w in text_lower for w in ("critical", "severe", "major", "catastrophic", "emergency")):
            spike_val = np.random.uniform(0.88, 0.99)
        elif any(w in text_lower for w in ("high", "surge", "spike", "overload", "fault")):
            spike_val = np.random.uniform(0.72, 0.88)
        else:
            spike_val = np.random.uniform(0.58, 0.72)

        # Determine which sensors to spike
        # Sensor index mapping: W0-W3 = 0-3, Xs0-Xs13 = 4-17
        if any(w in text_lower for w in ("temp", "heat", "thermal", "overheat")):
            spike_indices = [8, 9]    # Xs4, Xs5
        elif any(w in text_lower for w in ("vibr", "bearing", "noise", "rattle")):
            spike_indices = [4, 5]    # Xs0, Xs1
        elif any(w in text_lower for w in ("press", "hydraul", "leak", "fluid")):
            spike_indices = [12, 13]  # Xs8, Xs9
        elif any(w in text_lower for w in ("electric", "power", "volt", "current")):
            spike_indices = [0, 1]    # W0, W1
        else:
            spike_indices = [7, 8]    # Xs3, Xs4 (general)

        for idx in spike_indices:
            reading[idx] = float(spike_val)

        return reading

    def _estimate_capacity(self, machine_id: int, new_status: str) -> float:
        """
        Re-estimate factory capacity after this machine's status changes.
        Simple heuristic: each ONLINE=20%, DEGRADED=10%, OFFLINE=0%.
        """
        total = 0.0
        for mid, m in self.state.machines.items():
            status = new_status if mid == machine_id else m.status
            if status == "ONLINE":
                total += 20.0
            elif status == "DEGRADED":
                total += 10.0
        return round(total, 1)

    def _build_dispatch_order(
        self, machine_id: int, status: str, rul: float
    ) -> str:
        """Build a plain-English floor manager dispatch instruction."""
        name = self.state.machines[machine_id].name
        if status == "OFFLINE":
            return (
                f"{name} OFFLINE (RUL: {rul:.0f}). "
                f"Stop all operations on this unit. Schedule emergency repair."
            )
        elif status == "DEGRADED":
            return (
                f"{name} DEGRADED (RUL: {rul:.0f}). "
                f"Reduce to 50% load. Schedule maintenance within this shift."
            )
        else:
            return (
                f"{name} status nominal (RUL: {rul:.0f}). "
                f"Continue current operations. Monitor trending sensors."
            )

    # ═════════════════════════════════════════════════════════════════════════
    # ACTIONS
    # ═════════════════════════════════════════════════════════════════════════

    def action_reset_factory(self) -> None:
        """Reset all machines to ONLINE. Triggered by Ctrl+R."""
        self.state.reset_all()
        reset_call_count()          # restart dummy oracle degradation curve
        self._refresh_ops_analytics()
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory RESET — all machines restored to ONLINE.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FactoryApp()
    app.run()
