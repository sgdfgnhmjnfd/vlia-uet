# Deviations from prereg.md

1. (29/11 01:56, before Part B results were looked at) pour_oil has 0 segments in the test people of split 4 (P42-P54).
   That (action, split) cell is skipped; pour_oil is pooled over splits 1-3 only. Not a design change.
2. Clarification: T model input uses the same StandardScaler -> PCA-256 pipeline as the frame classifier (prereg wording
   "PCA-256 of seq1s"); the PCA is fit on train people only, as pre-registered.
3. EXPLORATORY (asked by Trang 29/11 ~01:30: "if not ok, try methods to improve"; written before seeing final tables):
   stage23_explore.py, a causal dilated TCN (10 layers, 64 ch, dropout 0.3, 25 epochs, no early stopping) on the 1-s PCA
   sequences. E1: per-second fine labels -> H-pred2 / HIST-pred2 (train features cross-fitted, 3 person groups).
   E2: T2 = same TCN, multi-task fine+dish, dish at prefix = mean softmax over observed seconds; 3 seeds and their ensemble.
   Crossfit uses the split's full-train PCA (unsupervised, train people only). All numbers from E1/E2 are exploratory.
