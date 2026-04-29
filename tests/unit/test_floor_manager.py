"""
tests/unit/test_floor_manager.py
FloorManager (Agent 3) — output validation and template fallback.
No Gemini calls. See tests/live/test_gemini_live.py for live API tests.

Run from project ROOT:  pytest tests/unit/test_floor_manager.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from agents.floor_manager import issue_dispatch_orders, _validate_output, _template_fallback


# ── _validate_output ──────────────────────────────────────────────────────────

class TestValidateOutput:
    def test_valid_output_passes(self, make_capacity_report):
        r = make_capacity_report("OFFLINE")
        ok, _ = _validate_output(
            "[Floor Manager] Final Assembly is OFFLINE. Halt. Reroute. 80.0%.", r
        )
        assert ok is True

    def test_missing_floor_manager_prefix_rejected(self, make_capacity_report):
        r = make_capacity_report("OFFLINE")
        ok, _ = _validate_output("Final Assembly is OFFLINE.", r)
        assert ok is False

    def test_empty_string_rejected(self, make_capacity_report):
        r = make_capacity_report("OFFLINE")
        ok, _ = _validate_output("", r)
        assert ok is False

    def test_over_800_chars_rejected(self, make_capacity_report):
        r = make_capacity_report("OFFLINE")
        ok, _ = _validate_output("[Floor Manager] " + "x" * 800, r)
        assert ok is False

    def test_missing_machine_name_rejected(self, make_capacity_report):
        r = make_capacity_report("OFFLINE")
        ok, _ = _validate_output("[Floor Manager] Machine 4 is OFFLINE.", r)
        assert ok is False


# ── _template_fallback ────────────────────────────────────────────────────────

class TestTemplateFallback:
    def test_offline_template_structure(self, make_capacity_report):
        r = make_capacity_report("OFFLINE", machine_id=4, rul=12.0, cap=80.0)
        t = _template_fallback(r)
        assert t.startswith("[Floor Manager]")
        assert "Final Assembly" in t
        assert "12.0" in t
        assert "80.0" in t
        assert "maintenance" in t.lower()
        assert _validate_output(t, r)[0]

    def test_degraded_template_structure(self, make_capacity_report):
        r = make_capacity_report("DEGRADED", machine_id=2, rul=22.0, cap=90.0,
                                  req=16.528, risk=True)
        t = _template_fallback(r)
        assert t.startswith("[Floor Manager]")
        assert "Paint & Coat" in t
        assert "22.0" in t
        assert "50%" in t
        assert _validate_output(t, r)[0]

    def test_online_template_structure(self, make_capacity_report):
        r = make_capacity_report("ONLINE", machine_id=1, rul=55.0, cap=100.0,
                                  req=14.875, risk=False)
        t = _template_fallback(r)
        assert t.startswith("[Floor Manager]")
        assert "Metal Press" in t
        assert "55.0" in t
        assert _validate_output(t, r)[0]

    def test_live_numbers_appear_in_output(self, make_capacity_report):
        r = make_capacity_report("OFFLINE", machine_id=3, rul=9.5, cap=60.0,
                                  req=24.792, risk=True)
        t = _template_fallback(r)
        assert "9.5" in t
        assert "60.0" in t

    @pytest.mark.parametrize("status", ["OFFLINE", "DEGRADED", "ONLINE"])
    def test_all_statuses_produce_valid_output(self, make_capacity_report, status):
        r = make_capacity_report(status)
        t = _template_fallback(r)
        ok, _ = _validate_output(t, r)
        assert ok is True
