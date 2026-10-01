"""Motion-aware adaptive temporal smoothing for detection boxes.

This is the box-level temporal module used by MA-ATS.  It accepts detections
from any detector and does not require a detector-specific package.
"""

from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["TemporalSmoother", "MAATSTemporalSmoother"]


class TemporalSmoother:
    """Fixed or motion-aware EMA smoothing for ``xyxy`` detection boxes.

    ``smooth_boxes_adaptive`` computes normalized centre displacement from the
    previous smoothed box.  The EMA weight is 0.3 below 0.05, 0.9 above 0.50,
    and linearly interpolated between these limits.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        max_history: int = 5,
        min_confidence: float = 0.25,
        enable_class_aware: bool = True,
    ):
        self.alpha = float(alpha)
        self.max_history = int(max_history)
        self.min_confidence = float(min_confidence)
        self.enable_class_aware = bool(enable_class_aware)
        self.box_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.conf_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.cls_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.frame_count = 0

    def smooth_boxes(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        track_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a fixed-weight EMA to ``boxes`` and ``confidences``."""
        if len(boxes) == 0:
            return boxes, confidences
        if track_ids is None:
            track_ids = np.arange(len(boxes))

        smoothed_boxes = np.asarray(boxes, dtype=float).copy()
        smoothed_conf = np.asarray(confidences, dtype=float).copy()
        for i, (box, conf, cls_id, track_id) in enumerate(
            zip(smoothed_boxes, smoothed_conf, class_ids, track_ids)
        ):
            if conf < self.min_confidence:
                continue
            box_hist = self.box_history[int(track_id)]
            conf_hist = self.conf_history[int(track_id)]
            cls_hist = self.cls_history[int(track_id)]
            if self.enable_class_aware and cls_hist and cls_hist[-1] != cls_id:
                box_hist.clear()
                conf_hist.clear()
                cls_hist.clear()
            if box_hist:
                smoothed_boxes[i] = self.alpha * box + (1.0 - self.alpha) * box_hist[-1]
                smoothed_conf[i] = self.alpha * conf + (1.0 - self.alpha) * conf_hist[-1]
            box_hist.append(smoothed_boxes[i].copy())
            conf_hist.append(float(smoothed_conf[i]))
            cls_hist.append(cls_id)
        self.frame_count += 1
        return smoothed_boxes, smoothed_conf

    def smooth_boxes_adaptive(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        track_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the MA-ATS motion-aware adaptive EMA."""
        if len(boxes) == 0:
            return boxes, confidences
        if track_ids is None:
            track_ids = np.arange(len(boxes))

        smoothed_boxes = np.asarray(boxes, dtype=float).copy()
        smoothed_conf = np.asarray(confidences, dtype=float).copy()
        for i, (box, conf, cls_id, track_id) in enumerate(
            zip(smoothed_boxes, smoothed_conf, class_ids, track_ids)
        ):
            if conf < self.min_confidence:
                continue
            box_hist = self.box_history[int(track_id)]
            conf_hist = self.conf_history[int(track_id)]
            cls_hist = self.cls_history[int(track_id)]
            if self.enable_class_aware and cls_hist and cls_hist[-1] != cls_id:
                box_hist.clear()
                conf_hist.clear()
                cls_hist.clear()
            if box_hist:
                motion = self._calculate_motion(box, box_hist[-1])
                alpha = self._get_adaptive_alpha(motion)
                smoothed_boxes[i] = alpha * box + (1.0 - alpha) * box_hist[-1]
                smoothed_conf[i] = alpha * conf + (1.0 - alpha) * conf_hist[-1]
            box_hist.append(smoothed_boxes[i].copy())
            conf_hist.append(float(smoothed_conf[i]))
            cls_hist.append(cls_id)
        self.frame_count += 1
        return smoothed_boxes, smoothed_conf

    @staticmethod
    def _calculate_motion(box1: np.ndarray, box2: np.ndarray) -> float:
        """Return normalized centre displacement between two ``xyxy`` boxes."""
        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        distance = np.linalg.norm(center1 - center2)
        area = max(float((box1[2] - box1[0]) * (box1[3] - box1[1])), 0.0)
        return float(distance / (np.sqrt(area) + 1e-6))

    @staticmethod
    def _get_adaptive_alpha(motion: float) -> float:
        if motion < 0.05:
            return 0.3
        if motion > 0.5:
            return 0.9
        return 0.3 + (motion - 0.05) * 0.6 / 0.45

    def reset(self) -> None:
        self.box_history.clear()
        self.conf_history.clear()
        self.cls_history.clear()
        self.frame_count = 0

    def reset_track(self, track_id: int) -> None:
        self.box_history.pop(int(track_id), None)
        self.conf_history.pop(int(track_id), None)
        self.cls_history.pop(int(track_id), None)


MAATSTemporalSmoother = TemporalSmoother
