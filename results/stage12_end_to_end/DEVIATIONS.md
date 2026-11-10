# Stage 12: deviations from the task text

1. **299 units, not 300.** Task 8 has 29 demos, so (t = 8, e = 29) is dropped, M1 became "at least 29 demos per task", and τ uses the 80 calibration demos of the other 9 tasks. Decided by Trang before any scoring. Recorded in `results/prereg_stage12_end_to_end_amendment_1.md`. The original pre-registration also carries it (section "Amendments", A1) because it was committed before the amendment file was requested.
2. **Text side of the context fallback.** F's real first-step fallback also subtracts a mean from the text embeddings. As designed (and as gate E1 requires), Stage 12 subtracts a mean from the frames only. Disclosed in the pre-registration, A2.
3. **Scoring time.** `score` took 52 min (estimate: minutes). It encoded 8 frames per call. This has no effect on the numbers.
