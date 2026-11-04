"""Tests for HermesAdapter._check_hermes_result — Hermes result → A2A state mapping."""

import pytest

from a2a_adapter.integrations.hermes import HermesAdapter


class TestCheckHermesResult:
    """Verify that _check_hermes_result correctly maps Hermes run_conversation()
    return dicts to either a response string or a RuntimeError."""

    def test_success_returns_final_response(self):
        result = {"final_response": "Done!", "completed": True}
        assert HermesAdapter._check_hermes_result(result, "s1") == "Done!"

    def test_partial_with_response_returns_it(self):
        """Max iterations hit but a summary was produced — still usable."""
        result = {
            "final_response": "Partial summary",
            "completed": False,
            "partial": True,
        }
        assert HermesAdapter._check_hermes_result(result, "s1") == "Partial summary"

    def test_failed_with_response_returns_it(self):
        """Error flagged but a response was still generated."""
        result = {
            "final_response": "Best effort",
            "failed": True,
            "error": "context length exceeded",
        }
        assert HermesAdapter._check_hermes_result(result, "s1") == "Best effort"

    def test_failed_no_response_raises(self):
        result = {
            "final_response": "",
            "completed": False,
            "failed": True,
            "error": "Invalid API response after 3 retries",
        }
        with pytest.raises(RuntimeError, match="Invalid API response"):
            HermesAdapter._check_hermes_result(result, "s1")

    def test_not_completed_no_response_raises(self):
        result = {"completed": False}
        with pytest.raises(RuntimeError, match="did not complete"):
            HermesAdapter._check_hermes_result(result, "s1")

    def test_failed_missing_error_key_raises_generic(self):
        result = {"failed": True}
        with pytest.raises(RuntimeError, match="did not complete"):
            HermesAdapter._check_hermes_result(result, "s1")

    def test_missing_failed_key_but_not_completed_raises(self):
        """Some Hermes return dicts omit 'failed' entirely."""
        result = {"completed": False, "error": "something went wrong"}
        with pytest.raises(RuntimeError, match="something went wrong"):
            HermesAdapter._check_hermes_result(result, "s1")

    def test_empty_result_returns_empty_string(self):
        """A fully-empty dict (no error flags) returns empty string."""
        assert HermesAdapter._check_hermes_result({}, "s1") == ""

    def test_none_final_response_treated_as_empty(self):
        result = {"final_response": None, "failed": True, "error": "oops"}
        with pytest.raises(RuntimeError, match="oops"):
            HermesAdapter._check_hermes_result(result, "s1")

    def test_interrupted_no_response_raises(self):
        result = {"completed": False, "interrupted": True}
        with pytest.raises(RuntimeError, match="did not complete"):
            HermesAdapter._check_hermes_result(result, "s1")

    def test_interrupted_with_response_returns_it(self):
        result = {
            "final_response": "Operation interrupted during retry",
            "completed": False,
            "interrupted": True,
        }
        assert (
            HermesAdapter._check_hermes_result(result, "s1") == "Operation interrupted during retry"
        )
