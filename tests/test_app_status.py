import unittest

from app import _advisory_status_text, _friendly_status
from src.ai_retrieval import AdapterStatus


class AdvisoryStatusDisplayTests(unittest.TestCase):
    def test_available_with_candidates(self):
        status = AdapterStatus("llm", True, "ready")
        self.assertEqual(_friendly_status(status), "Available")
        self.assertEqual(
            _advisory_status_text(status, ("VAL-001",)),
            ("VAL-001", "ADVISORY · AVAILABLE"),
        )

    def test_available_without_candidates_is_not_reported_as_unavailable(self):
        status = AdapterStatus("llm", True, "ready")
        self.assertEqual(
            _advisory_status_text(status, ()),
            ("No candidate returned", "ADVISORY · NO CANDIDATE"),
        )

    def test_rate_limit_is_explicit(self):
        status = AdapterStatus("llm", False, "Gemini interpretation failed: 429 RESOURCE_EXHAUSTED")
        self.assertEqual(_friendly_status(status), "Currently rate limited")
        self.assertEqual(
            _advisory_status_text(status, ()),
            (
                "Advisory request was rate limited by the provider.",
                "CURRENTLY RATE LIMITED",
            ),
        )

    def test_missing_credentials_are_explicit(self):
        status = AdapterStatus("llm", False, "Gemini API key is not configured.")
        self.assertEqual(_friendly_status(status), "Credentials not configured")
        self.assertEqual(
            _advisory_status_text(status, ()),
            ("Advisory credentials are not configured.", "CONFIGURATION REQUIRED"),
        )

    def test_other_failures_are_unavailable(self):
        status = AdapterStatus("llm", False, "Gemini SDK/client initialization failed: connection error")
        self.assertEqual(_friendly_status(status), "Currently unavailable")
        self.assertEqual(
            _advisory_status_text(status, ()),
            ("Advisory service is currently unavailable.", "CURRENTLY UNAVAILABLE"),
        )


if __name__ == "__main__":
    unittest.main()
