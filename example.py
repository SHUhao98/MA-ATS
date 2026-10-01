"""Minimal detector-agnostic MA-ATS example."""

import numpy as np

from maats_fsm import EnhancedFSM, MAATS_FSM_PARAMS
from maats_temporal_smoothing import TemporalSmoother


def main() -> None:
    smoother = TemporalSmoother(alpha=0.7, max_history=5, min_confidence=0.25)
    fsm = EnhancedFSM(**MAATS_FSM_PARAMS)

    class_ids = np.array([0], dtype=np.int64)
    track_ids = np.array([0], dtype=np.int64)
    sequence = [
        (np.array([[100.0, 80.0, 180.0, 160.0]]), np.array([0.82])),
        (np.array([[104.0, 82.0, 184.0, 162.0]]), np.array([0.86])),
        (np.array([[108.0, 84.0, 188.0, 164.0]]), np.array([0.88])),
    ]

    for boxes, confidences in sequence:
        smooth_boxes, smooth_conf = smoother.smooth_boxes_adaptive(
            boxes, confidences, class_ids, track_ids
        )
        effective_conf = float(np.max(confidences))
        trigger, state = fsm.process_frame(effective_conf)
        print(
            f"box={smooth_boxes[0].round(2).tolist()} "
            f"confidence={float(smooth_conf[0]):.3f} "
            f"trigger={trigger} state={state}"
        )


if __name__ == "__main__":
    main()
