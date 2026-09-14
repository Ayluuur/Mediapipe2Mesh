"""Hand identity and lifecycle regression tests without camera input."""

import unittest
from types import SimpleNamespace

import numpy as np

from mediapipe2mesh.tracking.handedness import HandednessResolver


class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.resolver = HandednessResolver(confirm_frames=3)
        self.states = {}
        for side in ('left', 'right'):
            state = SimpleNamespace(side=side, screen_wrist=None, last_seen=-np.inf)
            def deactivate(state=state):
                state.screen_wrist = None
                state.last_seen = -np.inf
            state.deactivate = deactivate
            self.states[side] = state

    def detection(self, side='left', x=.3, score=.9):
        return dict(raw_side=side, wrist=np.array([x, .5]), score=score)

    def frame(self, detections, time):
        resolved = self.resolver.resolve(detections, self.states, time)
        for side, detection in resolved.items():
            self.states[side].screen_wrist = detection['wrist']
            self.states[side].last_seen = time
        return resolved

    def activate(self):
        self.assertFalse(self.frame([self.detection()], 0))
        self.assertFalse(self.frame([self.detection()], .03))
        self.assertEqual(set(self.frame([self.detection()], .06)), {'left'})

    def test_single_hand_label_flicker_never_enables_other_hand(self):
        self.activate()
        for frame in range(3, 40):
            result = self.frame([self.detection('right')], frame * .03)
            self.assertEqual(set(result), {'left'})
        self.assertIsNone(self.states['right'].screen_wrist)

    def test_unmatched_wrong_label_does_not_bypass_confirmation(self):
        self.activate()
        self.assertFalse(self.frame([self.detection('right', .95)], .09))
        self.assertIsNone(self.states['right'].screen_wrist)
        self.assertEqual(set(self.frame([self.detection()], .12)), {'left'})

    def test_short_loss_holds_and_long_loss_closes_then_reconfirms(self):
        self.activate()
        self.frame([], .2)
        self.assertIsNotNone(self.states['left'].screen_wrist)
        self.frame([], .42)
        self.assertIsNone(self.states['left'].screen_wrist)
        self.assertFalse(self.frame([self.detection('right')], .45))
        self.assertFalse(self.frame([self.detection('right')], .48))
        self.assertEqual(set(self.frame([self.detection('right')], .51)), {'right'})

    def test_isolated_false_second_hand_does_not_activate(self):
        self.activate()
        result = self.frame([self.detection(), self.detection('right', .8)], .09)
        self.assertEqual(set(result), {'left'})
        self.frame([self.detection()], .12)
        result = self.frame([self.detection(), self.detection('right', .8)], .15)
        self.assertEqual(set(result), {'left'})

    def test_confirmed_second_hand_can_activate(self):
        self.activate()
        for time in (.09, .12, .15):
            result = self.frame([self.detection(), self.detection('right', .8)], time)
        self.assertEqual(set(result), {'left', 'right'})


if __name__ == '__main__':
    unittest.main()
