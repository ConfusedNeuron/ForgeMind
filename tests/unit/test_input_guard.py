"""
tests/unit/test_input_guard.py
InputGuard — keyword validation, length limits, edge cases.
Pure Python, zero network calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from agents.input_guard import is_valid_fault_input


class TestInputGuardAccepts:
    @pytest.mark.parametrize("text", [
        "bearing temperature surge on Machine 4",
        "pressure spike in hydraulic line",
        "vibration and shaking on CNC-Alpha",
        "coolant leak near the pump",
        "RPM fluctuation on motor drive",
        "overload on Mill-Epsilon",
        "shaft misalignment detected in gearbox",
        "temperature overheat in stator winding",
        "sensor malfunction on Lathe-Delta",
    ])
    def test_valid_inputs_accepted(self, text):
        valid, reason = is_valid_fault_input(text)
        assert valid is True, f"Expected valid but got: {reason!r}"
        assert reason == ""

    def test_exactly_500_chars_with_keyword_passes(self):
        # "fault" = 5 chars, pad to exactly 500
        text = "fault " + "x" * 494
        assert len(text) == 500
        valid, _ = is_valid_fault_input(text)
        assert valid is True

    def test_minimum_length_with_keyword(self):
        valid, _ = is_valid_fault_input("fault")
        assert valid is True


class TestInputGuardRejects:
    @pytest.mark.parametrize("text", [
        "x", "hi", "", "    ",
        "hello world",
        "what's for lunch?",
        "the quick brown fox jumps over the lazy dog",
        "a" * 501,
        "a" * 1000,
    ])
    def test_invalid_inputs_rejected(self, text):
        valid, reason = is_valid_fault_input(text)
        assert valid is False
        assert len(reason) > 0

    def test_500_char_boundary_passes(self):
        text = "bearing " + "x" * 492   # 500 chars, has keyword
        valid, _ = is_valid_fault_input(text)
        assert valid is True

    def test_501_char_boundary_fails(self):
        text = "bearing " + "x" * 493   # 501 chars
        valid, _ = is_valid_fault_input(text)
        assert valid is False


class TestInputGuardContract:
    def test_returns_two_element_tuple(self):
        result = is_valid_fault_input("bearing surge")
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_bool(self):
        valid, _ = is_valid_fault_input("bearing surge")
        assert isinstance(valid, bool)

    def test_second_element_is_str(self):
        _, reason = is_valid_fault_input("bearing surge")
        assert isinstance(reason, str)

    def test_valid_input_returns_empty_reason(self):
        valid, reason = is_valid_fault_input("bearing overheat on machine 3")
        assert valid is True and reason == ""
