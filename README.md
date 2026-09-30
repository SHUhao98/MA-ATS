# MA-ATS

Reference implementation of the Motion-Aware Adaptive Temporal Smoothing (MA-ATS) post-processing method for orchard bird detection and event-triggered deterrence.

## Files

- maats_temporal_smoothing.py: motion-aware adaptive EMA for xyxy boxes, confidence scores, class IDs, and optional track IDs.
- maats_fsm.py: confidence accumulation, hysteresis, gap tolerance, and the four states IDLE, CANDIDATE, ACTIVE, and COOLDOWN.
- example.py: a minimal detector-agnostic usage example.
- requirements.txt: the runtime dependency.

## Install

    python -m pip install -r requirements.txt

## Data availability

The 317 raw orchard videos used in the MA-ATS experiments are provided in GitHub Release v1.0.0 as five TAR parts. Download all parts and extract them into the same directory; verify integrity with SHA256SUMS.txt. The repository does not include manuscript files, figures, cached detections, or model weights.
