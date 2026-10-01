"""Four-state confidence-accumulation FSM used by MA-ATS.

Improves upon the original binary N-frame counting FSM by adding:
  1. Confidence-weighted score accumulation (vs. binary counting)
  2. Hysteresis (T_on > T_off) to prevent boundary flickering
  3. Sustained ACTIVE state with gap-tolerant decay
  4. Short cooldown after ACTIVE exit

Stays compatible with existing detection pipelines: input is per-frame
detection data (confidences or binary flags), output is trigger signal and state.

States: 0=IDLE, 1=CANDIDATE, 2=ACTIVE, 3=COOLDOWN
"""

import numpy as np
from typing import List, Tuple, Dict

MAATS_FSM_PARAMS = {
    "conf_threshold": 0.25,
    "score_max": 5.0,
    "conf_weight": 1.0,
    "decay": 0.2,
    "gap_decay": 0.05,
    "T_on": 1.0,
    "T_off": 0.3,
    "N_conf": 2,
    "gap": 1,
    "cooldown_frames": 3,
    "min_event_length": 3,
}


class EnhancedFSM:
    """Enhanced FSM trigger with confidence-weighted hysteresis and gap tolerance."""

    def __init__(
        self,
        conf_threshold: float = 0.25,
        score_max: float = 5.0,
        conf_weight: float = 1.0,
        decay: float = 0.2,
        gap_decay: float = 0.05,
        T_on: float = 1.0,
        T_off: float = 0.3,
        N_conf: int = 2,
        gap: int = 1,
        cooldown_frames: int = 3,
        min_event_length: int = 3,
    ):
        """
        Args:
            conf_threshold:  Minimum confidence for a detection to contribute.
            score_max:       Maximum accumulated trigger_score.
            conf_weight:     Multiplier for confidence contribution per frame.
            decay:           Score decrease per frame with NO detection (beyond gap).
            gap_decay:       Score decrease per frame WITHIN gap tolerance.
            T_on:            Score threshold to ENTER Active state.
            T_off:           Score threshold to EXIT Active state (must be < T_on).
            N_conf:          Minimum consecutive valid detections to enter CANDIDATE.
            gap:             Number of tolerated consecutive missed frames in ACTIVE.
            cooldown_frames: Frames to wait after ACTIVE exit before re-entry.
            min_event_length: Duration threshold for counting short ACTIVE segments.
        """
        # Score parameters
        self.conf_threshold = conf_threshold
        self.score_max = score_max
        self.conf_weight = conf_weight
        self.decay = decay
        self.gap_decay = gap_decay

        # Hysteresis
        self.T_on = T_on
        self.T_off = T_off
        assert T_on > T_off, f"T_on ({T_on}) must be > T_off ({T_off})"

        # State transition thresholds
        self.N_conf = N_conf
        self.gap = gap
        self.cooldown_frames = cooldown_frames

        # Event statistics
        self.min_event_length = min_event_length
        self.reset()

    def reset(self):
        """Reset all internal state."""
        self.state = 0                # 0=IDLE, 1=CANDIDATE, 2=ACTIVE, 3=COOLDOWN
        self.score = 0.0             # Accumulated trigger score
        self.consecutive_dets = 0    # Consecutive valid detections
        self.gap_counter = 0         # Consecutive missed frames (within gap tolerance)
        self.cooldown_counter = 0    # Remaining cooldown frames
        self.frame_count = 0         # Total frames processed

        # State history for analysis
        self.score_history: List[float] = []
        self.state_history: List[int] = []
        self.trigger_signal: List[int] = []
        self.event_boundaries: List[Tuple[int, int]] = []  # (start_frame, end_frame) of ACTIVE periods

        # Per-event tracking
        self._current_event_start = -1


    def process_frame(self, effective_confidence: float) -> Tuple[int, int]:
        """
        Process one frame and return (trigger_signal, state).

        Args:
            effective_confidence:
                Real confidence (0-1) if detection present and >= conf_threshold.
                0.0 if no valid detection.

        Returns:
            (trigger_signal: 0 or 1, current_state: 0-3)
        """
        # --- Step 1: Determine effective detection contribution ---
        has_valid_det = effective_confidence > 0.0 and effective_confidence >= self.conf_threshold

        if has_valid_det:
            conf_increment = self.conf_weight * effective_confidence
            self.consecutive_dets += 1
        else:
            conf_increment = 0.0
            self.consecutive_dets = 0

        # --- Step 2: State-dependent processing ---
        prev_state = self.state

        if self.state == 0:  # IDLE
            self._process_idle_candidate(conf_increment, has_valid_det)
        elif self.state == 1:  # CANDIDATE
            self._process_idle_candidate(conf_increment, has_valid_det)
        elif self.state == 2:  # ACTIVE
            self._process_active(conf_increment, has_valid_det)
        elif self.state == 3:  # COOLDOWN
            self._process_cooldown(conf_increment, has_valid_det)

        # --- Step 3: Record history ---
        self.score_history.append(self.score)
        self.state_history.append(self.state)

        trig = 1 if self.state == 2 else 0  # ACTIVE = trigger ON
        self.trigger_signal.append(trig)

        # Track event boundaries
        if prev_state != 2 and self.state == 2:
            self._current_event_start = self.frame_count
        elif prev_state == 2 and self.state != 2:
            if self._current_event_start >= 0:
                self.event_boundaries.append((self._current_event_start, self.frame_count - 1))
                self._current_event_start = -1

        self.frame_count += 1
        return trig, self.state

    def _process_idle_candidate(self, conf_increment: float, has_det: bool):
        """Process IDLE or CANDIDATE state."""
        if has_det:
            # Accumulate score with ceiling
            self.score = min(self.score_max, self.score + conf_increment)
        else:
            # Decay score; can drop to 0
            self.score = max(0.0, self.score - self.decay)

        # Transition to ACTIVE if score crosses T_on AND we have N_conf detections
        if self.score >= self.T_on and self.consecutive_dets >= self.N_conf:
            self.state = 2  # ACTIVE
        elif self.score <= 0.0:
            self.state = 0  # IDLE
        elif self.consecutive_dets > 0:
            self.state = 1  # CANDIDATE
        else:
            self.state = 0  # IDLE

    def _process_active(self, conf_increment: float, has_det: bool):
        """Process ACTIVE state with gap tolerance."""
        if has_det:
            self.score = min(self.score_max, self.score + conf_increment)
            self.gap_counter = 0
        else:
            # Gap tolerance: slow decay
            self.gap_counter += 1
            if self.gap_counter <= self.gap:
                # Within gap tolerance — slow decay
                self.score = max(0.0, self.score - self.gap_decay)
            else:
                # Beyond gap tolerance — fast decay
                self.score = max(0.0, self.score - self.decay)

        # Exit ACTIVE if score drops below T_off AND gap exceeded
        if self.score <= self.T_off and self.gap_counter >= self.gap:
            self.state = 3          # COOLDOWN
            self.cooldown_counter = self.cooldown_frames

    def _process_cooldown(self, conf_increment: float, has_det: bool):
        """Process COOLDOWN state."""
        self.cooldown_counter -= 1

        if has_det:
            # Half-weight accumulation during cooldown
            self.score = min(self.score_max, self.score + conf_increment * 0.5)
        else:
            self.score = max(0.0, self.score - self.decay)

        # Exit COOLDOWN
        if self.cooldown_counter <= 0:
            if self.score >= self.T_on:
                self.state = 2  # Re-enter ACTIVE immediately
            elif self.score > 0:
                self.state = 1  # CANDIDATE
            else:
                self.state = 0  # IDLE

    # --- Statistics ---

    def get_trigger_signal(self) -> np.ndarray:
        """Return trigger signal array (0/1 per frame)."""
        return np.array(self.trigger_signal)

    def get_active_state(self) -> np.ndarray:
        """Return state indices (0/1/2/3 per frame)."""
        return np.array(self.state_history)

    def get_score_history(self) -> np.ndarray:
        """Return accumulated score per frame."""
        return np.array(self.score_history)

    def compute_statistics(self) -> Dict:
        """Compute event-level statistics."""
        events = self.event_boundaries
        if self._current_event_start >= 0:
            events = events + [(self._current_event_start, self.frame_count - 1)]

        event_durations = [end - start + 1 for start, end in events]
        short_events = [d for d in event_durations if d < self.min_event_length]
        state_switches = sum(
            1 for i in range(1, len(self.state_history))
            if self.state_history[i] != self.state_history[i-1]
        )
        total_active = sum(1 for s in self.state_history if s == 2)

        return {
            "event_count": len(events),
            "event_durations": event_durations,
            "mean_event_duration": np.mean(event_durations) if event_durations else 0,
            "state_switch_count": state_switches,
            "active_duration_frames": total_active,
            "active_ratio": total_active / max(1, len(self.state_history)),
            "false_short_trigger_count": len(short_events),
            "short_event_durations": short_events if short_events else [],
            "mean_score_active": np.mean([
                self.score_history[i] for i in range(len(self.score_history))
                if self.state_history[i] == 2
            ]) if total_active > 0 else 0.0,
            "max_score": max(self.score_history) if self.score_history else 0.0,
        }

    def get_params(self) -> Dict:
        """Return current FSM parameter configuration."""
        return {
            "conf_threshold": self.conf_threshold,
            "score_max": self.score_max,
            "conf_weight": self.conf_weight,
            "decay": self.decay,
            "gap_decay": self.gap_decay,
            "T_on": self.T_on,
            "T_off": self.T_off,
            "N_conf": self.N_conf,
            "gap": self.gap,
            "cooldown_frames": self.cooldown_frames,
            "min_event_length": self.min_event_length,
        }


def run_enhanced_fsm_on_sequence(
    detections: List[Dict],
    conf_key: str = "confidence",
    **fsm_kwargs
) -> Tuple[EnhancedFSM, Dict]:
    """
    Run the enhanced FSM on a per-frame detection sequence.

    Args:
        detections: List of per-frame dicts with 'has_detection' (bool)
                    and the maximum raw bird 'confidence' (float 0-1) when
                    a detection is present.
        conf_key: Key for confidence value in detection dicts.
        **fsm_kwargs: Passed to EnhancedFSM constructor.

    Returns:
        (fsm_instance, statistics_dict)
    """
    fsm = EnhancedFSM(**fsm_kwargs)
    for det in detections:
        if det.get("has_detection", False):
            if conf_key not in det:
                raise ValueError(f"Missing raw detection confidence: {conf_key}")
            conf = det[conf_key]
            effective_conf = conf if conf >= fsm.conf_threshold else 0.0
        else:
            effective_conf = 0.0
        fsm.process_frame(effective_conf)
    return fsm, fsm.compute_statistics()


# ============================================================
# Original FSM (for comparison)
# ============================================================

def original_fsm_trigger(d_t: np.ndarray, N: int = 2, gap: int = 1, cooldown: int = 10
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original FSM implementation for comparison purposes.

    Returns:
        (triggers: activation pulses, state: 0/1/2 per frame)
    """
    T = len(d_t)
    triggers = np.zeros(T, dtype=int)
    state = np.zeros(T, dtype=int)
    cons = 0
    g = 0
    cd_until = -1

    for t in range(T):
        if d_t[t] == 1:
            cons += 1
            g = 0
        else:
            if g < gap:
                g += 1
            else:
                cons = 0
                g = 0

        if cons >= N and t > cd_until:
            triggers[t] = 1
            state[t] = 2
            cd_until = t + cooldown
            cons = 0
            g = 0
        elif cons > 0:
            state[t] = 1

    return triggers, state
