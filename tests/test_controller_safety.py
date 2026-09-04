"""Focus-loss and stuck-key safety tests for the Pygame controller."""

from __future__ import annotations

import unittest

from pygame_controller import FocusSafety


class FocusSafetyTest(unittest.TestCase):
    def test_focus_loss_requires_neutral_and_key_release(self) -> None:
        focus = FocusSafety()

        active, neutral = focus.update(True, False, False)
        self.assertTrue(active)
        self.assertFalse(neutral)

        active, neutral = focus.update(False, False, True)
        self.assertFalse(active)
        self.assertTrue(neutral)

        # Regaining focus while W is still held must not reactivate control.
        active, neutral = focus.update(True, False, True)
        self.assertFalse(active)
        self.assertFalse(neutral)

        # A full release explicitly arms input again.
        active, neutral = focus.update(True, False, False)
        self.assertTrue(active)
        self.assertFalse(neutral)

    def test_minimize_disables_focus(self) -> None:
        focus = FocusSafety()
        focus.update(True, False, False)
        active, neutral = focus.update(True, True, False)
        self.assertFalse(active)
        self.assertTrue(neutral)


if __name__ == "__main__":
    unittest.main()
