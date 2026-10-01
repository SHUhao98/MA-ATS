# MA-ATS

Reference implementation of the Motion-Aware Adaptive Temporal Smoothing (MA-ATS) post-processing method for orchard bird detection and event-triggered deterrence.

This repository contains the two detector-output post-processing modules, not the complete YOLOv11n training and evaluation pipeline. The modules operate in parallel: smoothed boxes provide localisation output, while the FSM uses the maximum valid **raw** bird-detection confidence from each frame. Smoothed confidence is not fed back into the FSM.

## Files

- `maats_temporal_smoothing.py`: motion-aware adaptive EMA for `xyxy` boxes, confidence scores, class IDs, and optional track IDs.
- `maats_fsm.py`: confidence accumulation, hysteresis, gap tolerance, and the four states `IDLE`, `CANDIDATE`, `ACTIVE`, and `COOLDOWN`.
- `example.py`: a minimal detector-agnostic usage example.
- `test_maats.py`: focused checks for the smoothing and triggering rules.
- `requirements.txt`: runtime dependency.

## Install

    python -m pip install -r requirements.txt

Run the example and tests with `python example.py` and `python -m unittest test_maats.py`.

## Input and state

Pass `xyxy` boxes, confidences, class IDs, and optional cross-frame track IDs to `smooth_boxes_adaptive()`. When track IDs are unavailable, the smoother uses each frame's detection index as a short-term correspondence key; detection reordering can therefore mix targets. A detection below 0.25 does not update smoothing history. Call `reset()` between videos, or `reset_track(id)` when an external track ends.

For each frame, pass the maximum **raw** bird confidence to `EnhancedFSM.process_frame()`, or `0.0` when there is no bird detection. The FSM itself applies the 0.25 validity threshold. Its return value is `(active_signal, state)`, where `active_signal` is 1 throughout the Active state, not a single activation pulse. `MAATS_FSM_PARAMS` contains the paper's reported configuration.

## Data availability

The 317 raw orchard MP4 videos used in the MA-ATS experiments are available in the [v1.0.0 dataset release](https://github.com/SHUhao98/MA-ATS/releases/tag/v1.0.0). Download all five `maats-dataset-part*.tar` files and extract each archive into the same directory. Use `SHA256SUMS.txt` to verify the downloaded archives before extraction. The video files total approximately 7.93 GiB.

This release contains raw videos only; annotations, detector weights, manuscript files, figures, and cached detections are not included.
