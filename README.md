# MA-ATS

Reference implementation of the Motion-Aware Adaptive Temporal Smoothing (MA-ATS) post-processing method for orchard bird detection and event-triggered deterrence.

## Files

- `maats_temporal_smoothing.py`: motion-aware adaptive EMA for `xyxy` boxes, confidence scores, class IDs, and optional track IDs.
- `maats_fsm.py`: confidence accumulation, hysteresis, gap tolerance, and the four states `IDLE`, `CANDIDATE`, `ACTIVE`, and `COOLDOWN`.
- `example.py`: a minimal detector-agnostic usage example.
- `requirements.txt`: runtime dependency.

## Install

    python -m pip install -r requirements.txt

## Data availability

The 317 raw orchard MP4 videos used in the MA-ATS experiments are available in the [v1.0.0 dataset release](https://github.com/SHUhao98/MA-ATS/releases/tag/v1.0.0). Download all five `maats-dataset-part*.tar` files and extract each archive into the same directory. Use `SHA256SUMS.txt` to verify the downloaded archives before extraction. The video files total approximately 7.93 GiB.

This release contains raw videos only; annotations, detector weights, manuscript files, figures, and cached detections are not included.
