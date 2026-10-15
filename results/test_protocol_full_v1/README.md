# Locked candidate test protocol: EgoIntent full v1

**Protocol locked before reading candidate test feature tensors.** This is a video-UID-held-out candidate test, not a pristine or scene-disjoint benchmark.

- Train: 2091 samples / 23 UIDs.
- Validation: 412 samples / 4 UIDs.
- Candidate test: 407 samples / 4 UIDs.
- Excluded pilot-validation test UIDs: cc883eb4-7772-4ef8-bd85-2c9cb5c6565d.
- Candidate UIDs: 062f1e55-67d8-4cdc-89c6-7fb361a9b0f9, 19db1542-5537-46cd-82d1-7f72e356a2a9, cf279625-1654-43e7-8b96-9139132b1b47, ea183f27-9dbd-4fe6-827d-9d02882863f9.
- Model: mean-pool baseline trained and evaluated at each of 2/4/6/8 frames; seeds 0/1/2; 100 epochs.
- Checkpoint selection: validation MRR at matching training/evaluation prefix; candidate test is not used for selection.
- Gallery: candidate-test samples only; exact WHY text match defines relevance.
- Test use: one run, no tuning or repeated selection on test results.

## Limitations

- Candidate set is video-UID-disjoint from pilot training and validation, but not scene-disjoint: garden appeared in pilot validation.
- Candidate UIDs had no exact reference in inspected local result files, but unrecorded external use cannot be excluded.
- This is a candidate video-level test, not a pristine or scene-independent test benchmark.
- The text target PCA was fit on pilot train, not refit on all full-train UIDs.

Input and code hashes are recorded in `locked_protocol.json`. Do not change this manifest after candidate-test evaluation.
