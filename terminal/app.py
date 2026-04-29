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
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.binding import Binding
from textual import work
import pyfiglet

import re
import numpy as np

from .factory_state import FactoryState
from .layout import SensorFeedWidget, CapacityWidget, format_log_entry

# ── DL Engine — live model ────────────────────────────────────────────────────
from dl_engine.inference import predict_rul

# ── Agent pipeline ────────────────────────────────────────────────────────────
from agents.agent_loop import run_agent_loop, reset_factory as reset_agent_state

# ── Import ops analytics ──────────────────────────────────────────────────────
from .ops_analytics import (
    detect_rul_cliff,
    check_sensor_saturation,
    compute_maintenance_schedule,
    compute_shift_health,
    compute_degradation_leaderboard,
)


# ─────────────────────────────────────────────────────────────────────────────
# TITLE BANNER — permanent header, Claude Code style
# ─────────────────────────────────────────────────────────────────────────────

class TitleBanner(Static):
    """Compact ASCII art title that sits at the top of the dashboard."""

    DEFAULT_CSS = """
    TitleBanner {
        height: auto;
        width: 1fr;
        padding: 0 1;
        background: #111111;
        color: #ff7b54;
        text-align: center;
        column-span: 2;
    }
    """

    def on_mount(self) -> None:
        try:
            art = pyfiglet.figlet_format("FORGEMIND", font="ansi_shadow")
        except Exception:
            art = "  F O R G E M I N D\n"
        content = (
            f"[bold #ff7b54]{art}[/bold #ff7b54]"
            "[dim #888888]AGENTIC PREDICTIVE MAINTENANCE · INDUSTRY 4.0 · v1.0[/dim #888888]"
        )
        self.update(content)


class FactoryApp(App):
    """Predictive maintenance terminal dashboard."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 4;
        grid-rows: auto 1fr 1fr 3;
    }

    #sensor-pane {
        column-span: 1;
        row-span: 1;
        border: solid #555555;
        border-title-color: green;
        padding: 1;
    }

    #capacity-pane {
        column-span: 1;
        row-span: 1;
        border: solid #555555;
        border-title-color: cyan;
        padding: 1;
    }

    #comms-pane {
        column-span: 2;
        row-span: 1;
        border: solid #555555;
        border-title-color: #ff7b54;
        padding: 1;
    }

    #chaos-input {
        column-span: 2;
        row-span: 1;
        dock: bottom;
    }
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
        yield TitleBanner()
        yield SensorFeedWidget(id="sensor-pane")
        yield CapacityWidget(id="capacity-pane")
        yield RichLog(id="comms-pane", highlight=True, markup=True, wrap=True)
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
        Background worker — full integrated pipeline.

        Flow:
          1. Extract machine ID from text
          2. Snapshot old RUL for cliff detection
          3. Push a simulated fault reading for sparkline display
          4. Build base window (50×18) for the diagnostic agent
          5. Run full agent pipeline: Input Guard → Diagnostic Agent (Gemini) →
             DL Oracle (CNN-LSTM) → Capacity Agent (ΣPD/T) → Floor Manager (Gemini)
          6. _process_result → update factory state + ops analytics + refresh panes
        """
        machine_id = self._extract_machine_id(user_text)

        # ── Step 2: Snapshot old RUL before the oracle runs ──────────────────
        old_rul = self.state.machines[machine_id].rul

        # ── Step 3: Build base window FIRST (before simulate push) ─────────
        # The base_window must reflect accumulated damage from previous faults.
        # _build_window() pads with h[-1], so we build BEFORE pushing the
        # healthy-ish simulate reading — otherwise h[-1] would be healthy
        # and reset the accumulated damage in the base window.
        base_window = self.state.get_machine_sensor_window(machine_id)

        # ── Step 3b: Push simulated reading for sparklines ────────────────────
        # Push AFTER base_window is built so sparklines update immediately
        # without contaminating the DL oracle's input.
        fault_reading = self._simulate_fault_reading(user_text)
        self.state.push_machine_sensor_reading(machine_id, fault_reading)

        # ── Update active display machine ─────────────────────────────────────
        self.state.active_machine_id = machine_id

        # ── Step 5: Full agent pipeline ───────────────────────────────────────
        # run_agent_loop handles: Input Guard → Diagnostic Agent (injects spike
        # into base_window) → predict_rul() → Capacity Agent → Floor Manager.
        # Returns a result dict directly compatible with _process_result().
        result = run_agent_loop(user_text, machine_id, base_window, predict_rul)

        # ── Step 5b: Persist degraded sensor state ────────────────────────────
        # The diagnostic agent's injected tensor is ephemeral — without this,
        # each new fault starts from a near-healthy base window and the model
        # can predict a HIGHER RUL than the previous fault (health goes UP).
        # By pushing the injected window's final rows into per-machine sensor
        # history, subsequent faults compound on the already-degraded state.
        injected = result.get("injected_window")
        if injected is not None and result["valid"]:
            # Push the last 5 rows of the degraded tensor to build up history
            # that reflects the cumulative degradation pattern.
            for row in injected[-5:]:
                self.state.push_machine_sensor_reading(machine_id, row)

        # ── Step 6: Standard result processing ───────────────────────────────
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
            "metal press":     1, "press":         1, "stamping":  1,
            "paint":           2, "coat":          2, "coating":   2,
            "pcb":             3, "smt":           3, "board":     3,
            "assembly":        4, "final":         4, "merge":     4,
            "qc":              5, "pack":          5, "test":      5, "quality": 5,
        }
        # Use word-set matching to avoid substring collisions
        # (e.g. "press" inside "pressure" falsely routing to Machine 1)
        words = set(text.lower().split())
        for name, mid in name_map.items():
            if " " in name:
                # Multi-word keys: use substring check (e.g. "metal press")
                if name in text.lower():
                    return mid
            else:
                # Single-word keys: require exact word match
                if name in words:
                    return mid

        self._machine_cycle = (self._machine_cycle % 5) + 1
        return self._machine_cycle

    def _simulate_fault_reading(self, user_text: str) -> np.ndarray:
        """
        Generate a simulated sensor reading for sparkline display.

        IMPORTANT: Values MUST be in raw physical units (matching the DL
        engine's scaler ranges), NOT normalised [0, 1].  The per-machine
        sensor history feeds into get_machine_sensor_window(), which is
        passed as the base_window to the diagnostic agent.  If these
        values are normalised, the MinMaxScaler inside predict_rul()
        collapses them to near-zero and the model always predicts ~71.

        Fault keyword → which sensor group spikes:
          temperature/heat/thermal → Xs4, Xs5  (indices 8, 9)
          vibration/bearing        → Xs0, Xs1  (indices 4, 5)
          pressure/hydraulic       → Xs8, Xs9  (indices 12, 13)
          electric/power/overload  → W0, W1    (indices 0, 1)
          (default)                → Xs3, Xs4  (indices 7, 8)
        """
        from dl_engine.inference import get_healthy_baseline, raw_value_for_scaled

        text_lower = user_text.lower()

        # Baseline: one row from the healthy baseline (raw physical units)
        baseline_window = get_healthy_baseline(noise_std_frac=0.03)
        reading = baseline_window[0].copy()  # single (18,) row

        # Determine fault severity from keywords → scaled position [0,1]
        if any(w in text_lower for w in ("critical", "severe", "major", "catastrophic", "emergency")):
            spike_scaled = np.random.uniform(0.88, 0.99)
        elif any(w in text_lower for w in ("high", "surge", "spike", "overload", "fault")):
            spike_scaled = np.random.uniform(0.72, 0.88)
        else:
            spike_scaled = np.random.uniform(0.58, 0.72)

        # Determine which sensors to spike
        if any(w in text_lower for w in ("temp", "heat", "thermal", "overheat")):
            spike_indices = [8, 9]    # Xs4, Xs5
        elif any(w in text_lower for w in ("vibr", "bearing", "noise", "rattle")):
            spike_indices = [4, 5]    # Xs0, Xs1
        elif any(w in text_lower for w in ("press", "hydraul", "leak", "fluid")):
            spike_indices = [12, 13]  # Xs8, Xs9
        elif any(w in text_lower for w in ("electric", "power", "volt", "current", "overload")):
            spike_indices = [0, 1]    # W0, W1
        else:
            spike_indices = [7, 8]    # Xs3, Xs4 (general)

        for idx in spike_indices:
            reading[idx] = raw_value_for_scaled(idx, float(spike_scaled))

        return reading

    def _estimate_capacity(self, machine_id: int, new_status: str) -> float:
        """
        Re-estimate factory capacity after this machine's status changes.
        Simple heuristic: ONLINE=20%, DEGRADED=10%, OFFLINE=0% per machine.
        Replaced by real ΣPD/T math when capacity agent is wired in.
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
        self.state.reset_all()          # resets FactoryState (UI layer)
        reset_agent_state()             # resets capacity_agent.MACHINES + OFFLINE_MODE
        self._refresh_ops_analytics()
        self._refresh_sensor_pane()
        self._refresh_capacity_pane()
        self._log("System", "Factory RESET — all machines restored to ONLINE.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FactoryApp()
    app.run()