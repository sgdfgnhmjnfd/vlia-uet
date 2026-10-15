# Prefix-length goal disambiguation summary

Source: /media/dhqg/d3/vlia_outputs/temporal_progression_v1/summary.json

| Condition | Prefix frames | MRR mean | MRR std | Top-1 mean | Top-3 mean | Eval examples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gru_final | 2 | 0.039537 | 0.006215 | NA | NA | 3 |
| gru_final | 4 | 0.043881 | 0.006837 | NA | NA | 3 |
| gru_final | 6 | 0.046622 | 0.005515 | NA | NA | 3 |
| gru_final | 8 | 0.052122 | 0.006968 | NA | NA | 3 |
| gru_progression | 2 | 0.040027 | 0.002643 | NA | NA | 3 |
| gru_progression | 4 | 0.044794 | 0.004447 | NA | NA | 3 |
| gru_progression | 6 | 0.047372 | 0.002767 | NA | NA | 3 |
| gru_progression | 8 | 0.052492 | 0.005355 | NA | NA | 3 |
| mean_pool | full | 0.058870 | 0.001585 | 0.029457 | 0.041860 | 3 |

## Interpretation

- The early prefix curves increase from 2 to 6 frames and are near their best at 8 frames.
- Gains are monotonic but modest on the current split; the dataset is small and there is no untouched test set.
- Same-task versus cross-task splits are not available in the current summary artifact, so selective-risk analysis was not computed here.

## Notes

- Reported metrics are validation-only under the existing source-video split.
- The original temporal progression output remains under /media/dhqg/d3/vlia_outputs/temporal_progression_v1.
