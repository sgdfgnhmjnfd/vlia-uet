# Stage 18b (side check, descriptive): EgoIntent verbs, within-video split

Balanced accuracy, %, 95% CI by video bootstrap (10,000, seed 0). Chance 10.0, majority-class acc 25.5.

| test | feature | p | acc | bal_acc [95% CI] | macro-F1 |
|---|---|---:|---:|---|---:|
| a_within_video | S | 0 | 31.1 | 37.9 [26.7, 41.0] | 30.7 |
| a_within_video | M | 50 | 15.2 | 14.2 [11.6, 16.0] | 13.2 |
| a_within_video | S+M | 50 | 33.5 | 37.0 [26.6, 39.8] | 32.6 |
| a_within_video | M | 100 | 19.1 | 17.0 [12.8, 19.9] | 16.5 |
| a_within_video | S+M | 100 | 36.1 | 40.7 [30.9, 43.4] | 35.7 |
| c_permuted | S | 0 | 14.2 | 9.4 [7.6, 11.2] | 9.4 |
| c_permuted | M | 50 | 13.2 | 10.0 [8.0, 11.9] | 10.1 |
| c_permuted | S+M | 50 | 14.1 | 9.4 [7.3, 11.4] | 9.4 |
| c_permuted | M | 100 | 14.1 | 9.7 [8.3, 11.3] | 9.8 |
| c_permuted | S+M | 100 | 13.1 | 8.5 [6.9, 10.0] | 8.4 |
| b_train_fit_C100 | S | 0 | 100.0 | 100.0 [100.0, 100.0] | 100.0 |
| b_train_fit_C100 | M | 50 | 99.8 | 99.8 [99.5, 100.0] | 99.7 |
| b_train_fit_C100 | S+M | 50 | 100.0 | 100.0 [100.0, 100.0] | 100.0 |
| b_train_fit_C100 | M | 100 | 99.8 | 99.8 [99.5, 100.0] | 99.7 |
| b_train_fit_C100 | S+M | 100 | 100.0 | 100.0 [100.0, 100.0] | 100.0 |

(a) (S+M p50) - S paired: -0.8 [-2.9, +1.4]

**Reading (prereg_18b rules):** Features carry verb information; Stage 18 chance level is due to new scenes. S ~ S+M: within-video learning is mainly scene/objects, not motion.
