"""
tests/terminal/test_layout.py
layout.py pure helper functions — mini_sparkline, status_bar,
status_color, rul_color, rul_label, divider, format_log_entry.
Zero Textual rendering needed (we avoid instantiating widgets).

Run:  pytest tests/terminal/test_layout.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from terminal.layout import (
    mini_sparkline,
    status_bar,
    status_color,
    rul_color,
    rul_label,
    divider,
    format_log_entry,
    AGENT_COLORS,
)


# ── mini_sparkline ────────────────────────────────────────────────────────────

class TestMiniSparkline:
    BLOCK_CHARS = set(" ▁▂▃▄▅▆▇█")

    def test_empty_list_returns_dashes(self):
        result = mini_sparkline([])
        assert all(c == "─" for c in result)

    def test_default_width_is_20(self):
        assert len(mini_sparkline([])) == 20

    def test_custom_width_respected(self):
        result = mini_sparkline([0.5] * 30, width=10)
        assert len(result) == 10

    def test_output_uses_only_block_characters(self):
        result = mini_sparkline([0.1, 0.3, 0.5, 0.7, 0.9])
        assert all(c in self.BLOCK_CHARS for c in result)

    def test_single_value_renders_without_crash(self):
        result = mini_sparkline([0.5], width=5)
        assert len(result) == 1

    def test_all_same_values_renders(self):
        result = mini_sparkline([0.5] * 20)
        assert len(result) == 20
        assert all(c in self.BLOCK_CHARS for c in result)

    def test_low_values_use_lower_blocks(self):
        result = mini_sparkline([0.01] * 10)
        assert all(c in " ▁▂" for c in result)

    def test_high_values_use_upper_blocks(self):
        # Mix of high values with a low anchor so normalisation has range
        result = mini_sparkline([0.1] + [0.99] * 9)
        # With width=20 and 10 inputs, first 10 chars are padding spaces.
        # The last 9 chars are the high-value blocks — check those.
        assert all(c in "▆▇█" for c in result[-9:])

    def test_single_value_renders_without_crash(self):
        result = mini_sparkline([0.5], width=5)
        assert len(result) == 5

    def test_truncates_to_width_from_right(self):
        values = list(range(30))
        result = mini_sparkline(values, width=10)
        assert len(result) == 10


# ── status_bar ────────────────────────────────────────────────────────────────

class TestStatusBar:
    def test_online_is_full_bar(self):
        assert status_bar("ONLINE") == "████████"

    def test_degraded_is_half_bar(self):
        assert status_bar("DEGRADED") == "████░░░░"

    def test_offline_is_empty_bar(self):
        assert status_bar("OFFLINE") == "░░░░░░░░"

    def test_unknown_returns_question_marks(self):
        assert status_bar("UNKNOWN") == "????????"

    def test_all_bars_are_8_chars(self):
        for s in ["ONLINE", "DEGRADED", "OFFLINE"]:
            assert len(status_bar(s)) == 8


# ── status_color ──────────────────────────────────────────────────────────────

class TestStatusColor:
    def test_online_is_green(self):
        assert status_color("ONLINE") == "green"

    def test_degraded_is_yellow(self):
        assert status_color("DEGRADED") == "yellow"

    def test_offline_is_red(self):
        assert status_color("OFFLINE") == "red"

    def test_unknown_is_white(self):
        assert status_color("UNKNOWN") == "white"


# ── rul_color ─────────────────────────────────────────────────────────────────

class TestRulColor:
    @pytest.mark.parametrize("rul, expected", [
        (999.0, "green"),
        (31.0,  "green"),
        (30.0,  "yellow"),   # boundary — rul > 30 is green, ≤ 30 is yellow
        (16.0,  "yellow"),
        (15.0,  "red"),      # boundary — rul > 15 is yellow, ≤ 15 is red
        (0.0,   "red"),
    ])
    def test_rul_color_thresholds(self, rul, expected):
        assert rul_color(rul) == expected


# ── rul_label ─────────────────────────────────────────────────────────────────

class TestRulLabel:
    @pytest.mark.parametrize("rul, expected", [
        (999.0, "HEALTHY"),
        (31.0,  "HEALTHY"),
        (30.0,  "WARNING"),
        (16.0,  "WARNING"),
        (15.0,  "CRITICAL"),
        (0.0,   "CRITICAL"),
    ])
    def test_rul_label_thresholds(self, rul, expected):
        assert rul_label(rul) == expected


# ── divider ───────────────────────────────────────────────────────────────────

class TestDivider:
    def test_default_width_is_38(self):
        assert len(divider()) == 38

    def test_default_char_is_dash(self):
        assert all(c == "─" for c in divider())

    def test_custom_width(self):
        assert len(divider(width=20)) == 20

    def test_custom_char(self):
        result = divider(width=10, char="=")
        assert result == "=" * 10


# ── format_log_entry ──────────────────────────────────────────────────────────

class TestFormatLogEntry:
    def test_output_contains_agent_name(self):
        result = format_log_entry("Floor Manager", "Halt machine")
        assert "Floor Manager" in result

    def test_output_contains_message(self):
        result = format_log_entry("System", "Reset complete")
        assert "Reset complete" in result

    def test_output_contains_timestamp(self):
        result = format_log_entry("System", "msg")
        import re
        assert re.search(r"\d{2}:\d{2}:\d{2}", result)

    def test_unknown_agent_falls_back_to_white(self):
        result = format_log_entry("Unknown Agent", "test")
        assert "white" in result


# ── AGENT_COLORS ──────────────────────────────────────────────────────────────

class TestAgentColors:
    def test_all_expected_agents_have_colors(self):
        expected_agents = {
            "System", "Chaos Engine", "Diagnostic Agent",
            "DL Oracle", "Capacity Agent", "Floor Manager", "Ops Alert",
        }
        assert expected_agents <= set(AGENT_COLORS.keys())

    def test_all_colors_are_strings(self):
        assert all(isinstance(v, str) for v in AGENT_COLORS.values())
