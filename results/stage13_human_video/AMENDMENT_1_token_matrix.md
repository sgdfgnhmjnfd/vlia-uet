# Stage 13, Amendment 1: token matrix

**Amendment 1 (2026-11-10, before any human video or Stage 13 score).**

**Look-up instead of new rollouts.** `E2E-human` and `E2E-human-ask` are computed by look-up in a token matrix instead of by new rollouts. The matrix holds one rollout of the seed-0 VLIA checkpoint for every task t, every token task k (0–9) and every episode e = 0..29. Each rollout uses an empty instruction, `init_state_id = e` and seed 1000 + e. Entries with k = t are the existing `I-only` episodes; entries with k = (t+5) mod 10 are the existing `I-wrong` episodes. All other entries are new `I-tok` episodes.

**How each condition is read.**
- `E2E-human` for unit (t, e) is the matrix entry (t, k = F's 10-way prediction for Trang's clip of task t with rep (e mod 3) + 1 at p = 50% under C-scene, e).
- `E2E-human-ask` uses the `I-only` entry when the clip's margin is below τ_h, as before.

**What stays the same.** Rules H1, H2 and H3, τ_h, the clips, F and all descriptive outputs are unchanged. Step 2.4–2.5 of the plan (smoke and `E2E-human` rollouts) is replaced by the look-up. If a seed-1 matrix exists, the same look-up gives a descriptive seed-1 `E2E-human`.

**Reason.** A rollout depends only on (checkpoint, t, k, e). Running all combinations before the videos exist means no rollout can be influenced by the human-video results.
