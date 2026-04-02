# Team Terminal — Factory Floor Dashboard

A real-time terminal-based factory dashboard for predictive maintenance simulation using Python's Textual framework.

## Quick Start

```bash
pip install textual rich numpy

python -m terminal.app
```

## Architecture

```
/terminal/
├── __init__.py           # Package metadata
├── app.py                # Main Textual app (4-pane layout, event handlers)
├── factory_state.py      # Shared state management
├── layout.py             # Widget classes (sensor feed, capacity dashboard)
└── dummy_oracle.py       # Stub RUL predictor (swap on Day 10)
```

## The 4 Panes

**Pane 1 — Live Sensor Feed**
- 18 Unicode sparklines (W0-W3, Xs0-Xs13)
- RUL indicator with color coding
- Updates live as data flows through

**Pane 2 — Capacity Dashboard**
- All 5 machines with status bars (ONLINE/DEGRADED/OFFLINE)
- RUL for each machine
- Overall capacity percentage
- Break-even risk indicator (ΣPD/T)

**Pane 3 — Agent Comms Log**
- Color-coded messages from all agents
- Timestamped entries
- Auto-scrolling, scrollable history

**Pane 4 — Chaos Engine**
- Text input for fault descriptions
- Machine ID extraction (supports "Machine 1" or "alpha", "beta", etc.)
- Non-blocking, responsive input

## Key Design Patterns

### Non-Blocking Async
Heavy processing (RUL prediction, agent calls) runs in background thread:
```python
@work(thread=True)
def _run_chaos(self, user_text: str):
    result = process_pipeline(...)
    self.call_from_thread(self._refresh_sensor_pane)
```

### Shared State
All panes read from single `FactoryState` object:
```python
self.state.update_from_agent_result(result)
self._refresh_sensor_pane()      # reads from self.state
self._refresh_capacity_pane()    # reads from self.state
```

## Keyboard Shortcuts

| Key    | Action                 |
|--------|------------------------|
| Ctrl+R | Reset all machines     |
| Ctrl+Q | Quit application       |

## Integration Points

### Day 5: Agent Loop
Replace stub with real pipeline:
```python
# Before:
rul = predict_rul(base_window)
result = {...}  # hardcoded

# After:
from agents.agent_loop import run_agent_loop
result = run_agent_loop(user_text, machine_id, base_window, predict_rul)
```

### Day 10: DL Model
Swap one import:
```python
# Before:
from terminal.dummy_oracle import predict_rul

# After:
from dl_engine.inference import predict_rul
```

## Machine States

| RUL Value | Status   | Color  | Bar       |
|-----------|----------|--------|-----------|
| > 30      | ONLINE   | Green  | ████████  |
| 15–30     | DEGRADED | Yellow | ████░░░░  |
| ≤ 15      | OFFLINE  | Red    | ░░░░░░░░  |

## Color Scheme

| Component        | Color           |
|------------------|-----------------|
| System           | Dim gray        |
| Chaos Engine     | Bold magenta    |
| Diagnostic Agent | Cyan            |
| DL Oracle        | Bold green      |
| Capacity Agent   | Yellow          |
| Floor Manager    | Bold red        |

## Testing

The dummy oracle returns a fixed RUL value for testing UI without external dependencies:

```python
# dummy_oracle.py
def predict_rul(sensor_tensor: np.ndarray) -> float:
    return 15.0  # Test OFFLINE state
    # return 25.0  # Test DEGRADED state
    # return 50.0  # Test ONLINE state
```

Change the return value, then run the app to test different UI states.

## Team Ownership

Team Terminal owns `/terminal/*`. This includes:
- UI layout and styling
- State management
- Dummy oracle for testing
- Integration interface with agent loop

Team Terminal does NOT own:
- Agent loop implementation (Team Agent)
- DL model (Team DL)
- Sensor data generation (comes through agent loop)
