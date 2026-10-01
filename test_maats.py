"""Regression checks for the published MA-ATS post-processing rules."""

import unittest

import numpy as np

from maats_fsm import EnhancedFSM, MAATS_FSM_PARAMS, run_enhanced_fsm_on_sequence
from maats_temporal_smoothing import TemporalSmoother


class TemporalSmootherTests(unittest.TestCase):
    def test_adaptive_weight_and_float_output(self):
        smoother = TemporalSmoother()
        classes = np.array([0])
        first = np.array([[0, 0, 10, 10]])
        second = np.array([[1, 0, 11, 10]])
        smoother.smooth_boxes_adaptive(first, np.array([0.8]), classes)
        boxes, confidence = smoother.smooth_boxes_adaptive(
            second, np.array([0.6]), classes
        )
        expected_alpha = 0.3 + (0.1 - 0.05) * 0.6 / 0.45
        self.assertEqual(boxes.dtype.kind, "f")
        self.assertAlmostEqual(boxes[0, 0], expected_alpha)
        self.assertAlmostEqual(confidence[0], expected_alpha * 0.6 + (1 - expected_alpha) * 0.8)

    def test_invalid_confidence_does_not_update_history(self):
        smoother = TemporalSmoother()
        classes = np.array([0])
        smoother.smooth_boxes_adaptive(np.array([[0., 0., 10., 10.]]), np.array([0.8]), classes)
        smoother.smooth_boxes_adaptive(np.array([[4., 0., 14., 10.]]), np.array([0.1]), classes)
        self.assertEqual(len(smoother.box_history[0]), 1)


class EnhancedFSMTests(unittest.TestCase):
    def test_threshold_applies_to_direct_frame_input(self):
        fsm = EnhancedFSM(**MAATS_FSM_PARAMS)
        self.assertEqual(fsm.process_frame(0.2), (0, 0))
        self.assertEqual(fsm.score, 0.0)
        self.assertEqual(fsm.process_frame(0.6), (0, 1))
        self.assertEqual(fsm.process_frame(0.6), (1, 2))

    def test_zero_means_no_detection_even_with_zero_threshold(self):
        fsm = EnhancedFSM(conf_threshold=0.0)
        self.assertEqual(fsm.process_frame(0.0), (0, 0))

    def test_active_signal_and_gap_decay(self):
        fsm = EnhancedFSM(**MAATS_FSM_PARAMS)
        fsm.process_frame(0.6)
        fsm.process_frame(0.6)
        score = fsm.score
        self.assertEqual(fsm.process_frame(0.0), (1, 2))
        self.assertAlmostEqual(fsm.score, score - fsm.gap_decay)
        self.assertEqual(fsm.process_frame(0.0), (1, 2))
        self.assertAlmostEqual(fsm.score, score - fsm.gap_decay - fsm.decay)

    def test_sequence_requires_raw_confidence(self):
        with self.assertRaises(ValueError):
            run_enhanced_fsm_on_sequence([{"has_detection": True}])

    def test_short_active_segments_are_counted_not_removed(self):
        fsm = EnhancedFSM(**MAATS_FSM_PARAMS)
        fsm.process_frame(0.6)
        fsm.process_frame(0.6)
        stats = fsm.compute_statistics()
        self.assertEqual(stats["event_count"], 1)
        self.assertEqual(stats["false_short_trigger_count"], 1)


if __name__ == "__main__":
    unittest.main()
