"""
Factory state management — single source of truth for all UI panes.

All 4 panes read from FactoryState. When the processing pipeline returns
a result, update_from_agent_result() updates everything in one atomic operation.

New in this version:
  - rul_history:                per-machine rolling RUL log (for reliability score)
  - per_machine_sensor_history: separate 18-sensor ring buffer per machine
  - maintenance_schedule:       ranked ops action queue (computed by ops_analytics)
  - shift_health:               one-line factory status (computed by ops_analytics)
  - degradation_leaderboard:    which machine is declining fastest
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MachineState:
    """State of one factory machine."""
    id: int
    name: str
    status: str = "ONLINE"          # "ONLINE" | "DEGRADED" | "OFFLINE"
    rul: float = 999.0              # Remaining useful life (cycles)
    base_time: float = 8.0          # Max available hours per shift
    available_time: float = 8.0     # Current available hours


@dataclass
class FactoryState:
    """
    Global factory state shared across all 4 panes.

    5 Machines:
        1: CNC-Alpha       3: Press-Gamma
        2: CNC-Beta        4: Lathe-Delta
                           5: Mill-Epsilon
    """

    # ── Core machine state ───────────────────────────────────────────────────
    machines: dict = field(default_factory=lambda: {
        1: MachineState(1, "CNC-Alpha"),
        2: MachineState(2, "CNC-Beta"),
        3: MachineState(3, "Press-Gamma"),
        4: MachineState(4, "Lathe-Delta"),
        5: MachineState(5, "Mill-Epsilon"),
    })

    # ── Shared sensor ring buffer (used for the active machine sparklines) ───
    sensor_history: list = field(default_factory=lambda: [
        [] for _ in range(18)
    ])
    HISTORY_LENGTH: int = 60

    # ── Per-machine sensor ring buffers (for saturation checks) ─────────────
    # Key = machine_id (1–5), Value = list of 18 lists (one per sensor)
    per_machine_sensor_history: dict = field(default_factory=lambda: {
        i: [[] for _ in range(18)] for i in range(1, 6)
    })

    # ── Per-machine RUL history (last MAX_RUL_HISTORY predictions) ───────────
    # Used to compute prediction reliability (rolling CoV).
    rul_history: dict = field(default_factory=lambda: {
        i: [] for i in range(1, 6)
    })
    MAX_RUL_HISTORY: int = 10

    # ── Active display machine ────────────────────────────────────────────────
    active_machine_id: int = 1

    # ── Capacity metrics ─────────────────────────────────────────────────────
    capacity_pct: float = 100.0
    machine_req: float = 0.0
    breakeven_risk: bool = False

    # ── Comms log ────────────────────────────────────────────────────────────
    comms_log: list = field(default_factory=list)
    MAX_LOG_ENTRIES: int = 100

    # ── Ops analytics results (populated by app.py after each chaos cycle) ───
    maintenance_schedule: list = field(default_factory=list)
    shift_health: tuple = field(default_factory=lambda: (
        "NOMINAL  —  All 5 machines running. Capacity: 100%", "green"
    ))
    degradation_leaderboard: list = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────────
    # STATE UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def update_from_agent_result(self, result: dict):
        """
        Update machine states and capacity metrics from agent pipeline result.
        Also maintains per-machine RUL history for reliability scoring.

        Args:
            result: dict with 'valid', 'machine_statuses', 'capacity_report' keys
        """
        if not result.get("valid", False):
            return

        for ms in result.get("machine_statuses", []):
            mid = ms["id"]
            if mid in self.machines:
                self.machines[mid].status = ms["status"]
                self.machines[mid].rul    = ms["rul"]
                self.machines[mid].available_time = ms["available_time"]

                # Track RUL history for this machine (for reliability scoring)
                self.rul_history[mid].append(ms["rul"])
                if len(self.rul_history[mid]) > self.MAX_RUL_HISTORY:
                    self.rul_history[mid] = self.rul_history[mid][-self.MAX_RUL_HISTORY:]

        report = result.get("capacity_report", {})
        if report:
            self.capacity_pct    = report.get("capacity_pct",    self.capacity_pct)
            self.machine_req     = report.get("machine_req",     self.machine_req)
            self.breakeven_risk  = report.get("breakeven_risk",  self.breakeven_risk)

    # ─────────────────────────────────────────────────────────────────────────
    # SENSOR HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    def push_sensor_reading(self, sensor_values: np.ndarray):
        """
        Push one timestep of 18 sensor values into the shared ring buffer.
        Used to populate sparklines for the active machine.

        Args:
            sensor_values: numpy array of shape (18,)
        """
        for i, val in enumerate(sensor_values):
            self.sensor_history[i].append(float(val))
            if len(self.sensor_history[i]) > self.HISTORY_LENGTH:
                self.sensor_history[i].pop(0)

    def push_machine_sensor_reading(self, machine_id: int, sensor_values: np.ndarray):
        """
        Push one timestep of 18 sensor values into a specific machine's ring buffer.

        This is the machine-aware version.  Called when chaos is injected into
        machine X so that:
          1. The sensor pane can show realistic sparklines for that machine.
          2. Saturation detection (ops_analytics.check_sensor_saturation) can
             fire if those values are extreme.

        Also updates the shared sensor_history if machine_id == active_machine_id.

        Args:
            machine_id:    int 1–5
            sensor_values: numpy array of shape (18,)
        """
        if machine_id not in self.per_machine_sensor_history:
            self.per_machine_sensor_history[machine_id] = [[] for _ in range(18)]

        # Guard: wrong-length input silently corrupts the ring buffer.
        # Truncate if too long, skip if empty.
        if len(sensor_values) == 0:
            return
        values_18 = sensor_values[:18] if len(sensor_values) >= 18 else np.pad(
            sensor_values, (0, 18 - len(sensor_values)), constant_values=0.5
        )

        history = self.per_machine_sensor_history[machine_id]
        for i, val in enumerate(values_18):
            history[i].append(float(val))
            if len(history[i]) > self.HISTORY_LENGTH:
                history[i].pop(0)

        # Mirror to shared buffer if this is the displayed machine
        if machine_id == self.active_machine_id:
            self.push_sensor_reading(values_18)

    def get_sensor_window(self) -> np.ndarray:
        """
        Build a (50, 18) sensor window from the shared ring buffer.
        Pads with random baseline [0.3, 0.7] if insufficient history.

        Returns:
            numpy array of shape (50, 18) for predict_rul()
        """
        return self._build_window(self.sensor_history)

    def get_machine_sensor_window(self, machine_id: int) -> np.ndarray:
        """
        Build a (50, 18) sensor window for a specific machine.
        Falls back to the shared buffer if no per-machine history exists.

        Args:
            machine_id: int 1–5

        Returns:
            numpy array of shape (50, 18) for predict_rul()
        """
        history = self.per_machine_sensor_history.get(machine_id)
        if history is None or all(len(h) == 0 for h in history):
            return self.get_sensor_window()
        return self._build_window(history)

    def _build_window(self, history: list) -> np.ndarray:
        """
        Internal helper: build (50, 18) window from a 18-list history buffer.

        When insufficient sensor history exists, we fill with REALISTIC
        baseline values from the DL engine's scaler (the scaler's midpoint
        for each feature, in raw physical units).  This ensures that
        predict_rul() receives data in the correct domain.

        Previously this used np.random.uniform(0.3, 0.7) which produced
        values the MinMaxScaler squashed to near-zero, causing every
        prediction to return ~71.2 regardless of fault type.
        """
        # Lazy-import to avoid circular dependency at module load time
        from dl_engine.inference import get_healthy_baseline

        window = np.zeros((50, 18), dtype=np.float32)
        for sensor_idx in range(18):
            h = history[sensor_idx]
            if len(h) >= 50:
                window[:, sensor_idx] = h[-50:]
            elif len(h) > 0:
                padding = [h[0]] * (50 - len(h))
                window[:, sensor_idx] = np.array(padding + h, dtype=np.float32)
            else:
                # No history at all — fill with realistic baseline
                window[:, sensor_idx] = get_healthy_baseline(
                    noise_std_frac=0.02
                )[:, sensor_idx]
        return window

    # ─────────────────────────────────────────────────────────────────────────
    # COMMS LOG
    # ─────────────────────────────────────────────────────────────────────────

    def add_log_entry(self, agent_name: str, message: str):
        """Add timestamped entry to comms log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.comms_log.append({
            "time":    timestamp,
            "agent":   agent_name,
            "message": message,
        })
        if len(self.comms_log) > self.MAX_LOG_ENTRIES:
            self.comms_log = self.comms_log[-self.MAX_LOG_ENTRIES:]

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset_all(self):
        """Reset all machines to ONLINE and clear per-machine analytics state."""
        for m in self.machines.values():
            m.status         = "ONLINE"
            m.rul            = 999.0
            m.available_time = m.base_time

        self.capacity_pct   = 100.0
        self.machine_req    = 0.0
        self.breakeven_risk = False

        # Clear per-machine histories so reliability resets to CALIBRATING
        for mid in self.rul_history:
            self.rul_history[mid] = []
        for mid in self.per_machine_sensor_history:
            self.per_machine_sensor_history[mid] = [[] for _ in range(18)]
        for i in range(18):
            self.sensor_history[i] = []

        self.maintenance_schedule    = []
        self.degradation_leaderboard = []
        self.shift_health = (
            "NOMINAL  —  All 5 machines running. Capacity: 100%", "green"
        )
