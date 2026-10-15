# Selective prediction summary

Validation-only decision baselines using the saved temporal progression checkpoints.

| Condition | Prefix | Policy | Threshold | Coverage | Error rate | Accepted |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| gru_final | 2 | confidence | 0.0000 | 1.0000 | 0.9814 | 215 |
| gru_final | 2 | entropy | 0.9492 | 1.0000 | 0.9814 | 215 |
| gru_final | 2 | margin | -1.0000 | 1.0000 | 0.9814 | 215 |
| gru_final | 4 | confidence | 0.0000 | 1.0000 | 0.9767 | 215 |
| gru_final | 4 | entropy | 0.9322 | 1.0000 | 0.9767 | 215 |
| gru_final | 4 | margin | -1.0000 | 1.0000 | 0.9767 | 215 |
| gru_final | 6 | confidence | 0.0000 | 1.0000 | 0.9721 | 215 |
| gru_final | 6 | entropy | 0.9661 | 1.0000 | 0.9721 | 215 |
| gru_final | 6 | margin | -1.0000 | 1.0000 | 0.9721 | 215 |
| gru_final | 8 | confidence | 0.0000 | 1.0000 | 0.9674 | 215 |
| gru_final | 8 | entropy | 0.9492 | 1.0000 | 0.9674 | 215 |
| gru_final | 8 | margin | -1.0000 | 1.0000 | 0.9674 | 215 |
| gru_progression | 2 | confidence | 0.0000 | 1.0000 | 0.9814 | 215 |
| gru_progression | 2 | entropy | 0.9831 | 1.0000 | 0.9814 | 215 |
| gru_progression | 2 | margin | -1.0000 | 1.0000 | 0.9814 | 215 |
| gru_progression | 4 | confidence | 0.0000 | 1.0000 | 0.9721 | 215 |
| gru_progression | 4 | entropy | 0.9831 | 1.0000 | 0.9721 | 215 |
| gru_progression | 4 | margin | -1.0000 | 1.0000 | 0.9721 | 215 |
| gru_progression | 6 | confidence | 0.0000 | 1.0000 | 0.9767 | 215 |
| gru_progression | 6 | entropy | 0.9831 | 1.0000 | 0.9767 | 215 |
| gru_progression | 6 | margin | -1.0000 | 1.0000 | 0.9767 | 215 |
| gru_progression | 8 | confidence | 0.0000 | 1.0000 | 0.9674 | 215 |
| gru_progression | 8 | entropy | 0.9831 | 1.0000 | 0.9674 | 215 |
| gru_progression | 8 | margin | -1.0000 | 1.0000 | 0.9674 | 215 |
| mean_pool | 2 | confidence | 0.0000 | 1.0000 | 0.9721 | 215 |
| mean_pool | 2 | entropy | 0.9661 | 1.0000 | 0.9721 | 215 |
| mean_pool | 2 | margin | -1.0000 | 1.0000 | 0.9721 | 215 |
| mean_pool | 4 | confidence | 0.0000 | 1.0000 | 0.9721 | 215 |
| mean_pool | 4 | entropy | 0.9661 | 1.0000 | 0.9721 | 215 |
| mean_pool | 4 | margin | -1.0000 | 1.0000 | 0.9721 | 215 |
| mean_pool | 6 | confidence | 0.0000 | 1.0000 | 0.9674 | 215 |
| mean_pool | 6 | entropy | 0.9492 | 1.0000 | 0.9674 | 215 |
| mean_pool | 6 | margin | -1.0000 | 1.0000 | 0.9674 | 215 |
| mean_pool | 8 | confidence | 0.0000 | 1.0000 | 0.9628 | 215 |
| mean_pool | 8 | entropy | 0.9661 | 1.0000 | 0.9628 | 215 |
| mean_pool | 8 | margin | -1.0000 | 1.0000 | 0.9628 | 215 |

Interpretation: confidence and margin-based filtering are simple predict-vs-defer baselines, not validated semantic-evidence claims.
The policy should be interpreted as a decision support mechanism, not as proof that the model understands when the observation is semantically sufficient.
