from datetime import datetime
import unittest

from presentation_models import empty_data_state


class PresentationModelTests(unittest.TestCase):
    def test_normal_empty_window_keeps_requested_copy(self):
        state = empty_data_state(
            "Ceilometer",
            "No samples were found.",
            start=datetime(2026, 7, 18),
            end=datetime(2026, 7, 19),
            detail="Try a wider window.",
        )
        self.assertFalse(state.intentionally_powered_off)
        self.assertEqual(state.instrument_title, "Ceilometer")
        self.assertEqual(state.reason, "No samples were found.")
        self.assertEqual(state.detail, "Try a wider window.")

    def test_powered_off_state_explains_expected_absence(self):
        state = empty_data_state(
            "Ceilometer",
            "No samples were found.",
            pdu_status={"state": "Off", "detail": "PDU sample 2m old"},
        )
        self.assertTrue(state.intentionally_powered_off)
        self.assertEqual(state.eyebrow, "INTENTIONAL POWER-OFF")
        self.assertIn("intentionally powered off", state.reason)
        self.assertEqual(
            state.detail,
            "CL61 is off at its assigned PDU outlet. PDU sample 2m old.",
        )


if __name__ == "__main__":
    unittest.main()
