# Daily preregistration addendum: E1b

**Status:** Reconstructed from `scripts/run_e1b_smolvlm2.py`; the original E1b task text is unavailable in the current workspace.

**Timing deviation:** This addendum is written after the E1b run started; the user had not seen any E1b result.

## Reconstructed scoring procedure

- Use `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` on train/validation manifest rows only. The locked test split is not read.
- For each query, build the deterministic K=5 candidate set with `run_scene_cv.build_kway`; candidate 0 is the query's own WHY text, and the remaining candidates are distinct WHY texts from the same video.
- Evaluate prefixes of 2, 4, 6, and 8 frames. The model receives the first `k` frames and the fixed goal question in `scripts/run_e1b_smolvlm2.py`.
- Score each candidate with the image-conditioned average token log probability minus the no-image average token log probability (PMI). PMI is the primary score; the prediction is the candidate with the highest PMI.
- Save one JSONL record per `(sample_id, k)` under `/media/dhqg/d3/vlia_outputs/e1b/`; keep run logs under `/media/dhqg/d3/logs/`.
- Report K=5 accuracy and chance (0.20). Lift is accuracy divided by chance; accuracy-minus-chance is a separate quantity.

## Limits of reconstruction and deviations

The checked-in script does not implement or record compute ladder levels L0/L1/L2, a mismatched-video control, or explicit E1b-A/E1b-B decision thresholds. Those criteria cannot be reconstructed reliably from this code, so no thresholds or verdicts are inferred here. The existing `results/e1b_smolvlm2_v1/summary.csv` is a 20-query smoke result, not a full-run result. The current full run began before this addendum was written.
