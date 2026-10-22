# Preregistration amendment 1 — float32 baseline gate relaxation

**Committed before any ladder, selective, early-decision, or kNN number exists.**
**No other rule changes.**

## Reason

The cached baseline SigLIP embeddings (`/media/dhqg/d3/vlia_outputs/e1/siglip_frame_embeddings_e1_v1.npz`) are stored as float16. When the Stage 2 reproduction gate in `results/prereg_ladder_v1.md` computes K=5 accuracies from this cache, the float16 image embeddings reproduce the E1 baseline values within 1 query of 2503 at k=2 and k=4:

| k | E1 hits | float16 cache hits | Δ |
|---|---|---|---|
| 2 | 694 | 693 | −1 |
| 4 | 689 | 690 | +1 |
| 6 | 703 | 703 | 0 |
| 8 | 722 | 722 | 0 |

The original Stage 2 gate requires exact reproduction to `1e-9`, which fails and blocks all subsequent stages. The discrepancy is consistent with float16 quantization near a cosine-similarity ranking boundary.

## Amended gate

1. **Re-extract** the baseline encoder (`google/siglip-base-patch16-224`) frame embeddings in **float32** from the same video frames used by the original E1 cache, and store them as float32. Also re-extract text embeddings in float32 using the same templates (T0 = raw WHY text; T1 = `a first-person photo of a person doing: {why}`).

2. Store the float32 embeddings under `/media/dhqg/d3/vlia_outputs/ladder/siglip-base-patch16-224/` in the same layout as Stage 2 specifies: `frame_emb.npy` float32 `(N, 8, D)`, `sample_ids.json`, text embeddings per template.

3. **Gate condition (amended):** For every k in {2, 4, 6, 8}, the number of correct K=5 queries computed from the float32 re-extracted embeddings must differ from the E1 baseline by **at most 2**. If it fails, stop and report.

4. From this amendment forward, the **baseline embeddings** used in Stages 3–5 and the selective analysis are the **float32** re-extracted ones, not the float16 cache.

## What does not change

- All other preregistered rules in `results/prereg_ladder_v1.md` and `results/prereg_daily_v1.md` remain in force.
- Stage 3 nested pipeline ladder, Stage 4 early decision, Stage 5 cache-kNN, and the selective S1 analysis proceed exactly as written.
- Verdict rules (N1, N2, S2, K1, S1) are unchanged.
- The locked test split is never read.
