# Pre-registration: Stage 14, the model writes the intention itself (2026-11-12)

Committed and pushed before any generation, rollout or score. It has four parts:
- **Part 1:** the design, verbatim from the Stage 14 task.
- **Part 1b:** A3, copied from the amendment of 2026-11-12.
- **Part 2:** this run's scope.
- **Part 3:** implementation details the texts leave open, fixed now, before any number.

## Part 1: design (verbatim from the task)

### The generator G (fixed before any run)

- **Input.** k frames: the given frames in time order, at 448 px, in one user message.
- **Prompt for EgoIntent:**

  > These frames are from a first-person video of a person doing one step of a task. In one short sentence, say what the person wants to achieve in this step. Start with "I want to".

- **Prompt for LIBERO demos and Trang's clips:**

  > These frames show the start of someone doing a household task with objects on a table. In one short sentence, say what they want the objects to end up like. Start with "I want".

- **Decoding.** Greedy, `max_new_tokens=40`. Keep only the first sentence, and strip quotes and whitespace.
- **The intention vector of a generated sentence s** is the SigLIP T1 text embedding of s: "a first-person photo of a person doing: {s}", L2-normalised, exactly as for any other VLIA token.
- **Blank control.** The same prompt with k grey 448 × 448 images. It shows what G writes without seeing the video.

### A1. EgoIntent: open-ended intention (2,503 development steps, k ∈ {2, 4, 8})

G sees only the frames, never the candidate WHYs. Afterwards, for scoring only, the generated sentence is compared with the 5 same-video candidate WHYs that F used (`CV.build_kway`, the same sets as Stage 3). The comparison uses the SigLIP text–text cosine between the plain sentences, without the template.

- **Open-ended 5-way:** the true WHY must be the closest of the 5; a tie counts as wrong. Chance is 20%.
- **Open-ended rank over the whole video:** the rank of the true WHY among all distinct WHYs of the same video, reported as MRR.
- **Rules.** All CIs come from a video bootstrap (`CV.video_bootstrap`, B = 10,000, seed 0).
  - **GA1.** Open-ended 5-way at k = 8 has CI lower bound > 20%.
  - **GA2.** Video − blank control at k = 8, paired over steps, has CI lower bound > 0. The blank control runs on all 2,503 steps at k = 8 only.
- **Descriptive:**
  - all k, with F (with the list) on the same steps as a reference;
  - the open-ended accuracy of the Stage 3 Qwen multiple-choice answers;
  - 50 random (step, sentence) pairs, chosen with seed 0 and k = 8, written to `human_check.csv` with an empty `rating` column for Trang to fill in (correct / partly / wrong). Write it before any score is shown.

### A2. LIBERO-10: the robot acts on a generated intention

- **Query.** Demos d = 0..29 of each task (299 units; task 8 has 29), agent view, 8 frames over the first p of the demo, the same as Stage 12.
- **Primary.** p = 25%: G writes s, the token is the T1 embedding of s, and VLIA runs with an empty instruction (condition **`E2E-gen`**, 299 new rollouts).
  - Unit (t, e) uses demo e of task t, `init_state_id = e` and seed 1000 + e.
  - `z_int_task` holds s.
- **Control `E2E-gen-blank`.** The blank-control sentence for each task runs on the same units. It is one sentence per prompt, so it is the same on every unit and costs 299 rollouts.
- **Rule.**
  - **GA3.** success(`E2E-gen`) − success(`I-wrong`) has CI lower bound > 0 (`paired_bootstrap`, 299 units).
- **Descriptive:**
  - `E2E-gen` against `I-only`, Stage 12 `E2E` (list-based) and `E2E-gen-blank`.
  - Text match: the description among the 10 that is nearest to s by text cosine, and how often it is the true task. This is computed after the fact; the robot never uses it.
  - **Snap variant**, without new rollouts: the token of the nearest description, looked up in the token matrix. This separates "G understood the task" from "the robot does not understand a free sentence".
  - Success of `E2E-gen` against s's cosine to the true task's description, in tertiles.

### B. Early prediction over time

- **Prefixes.**
  - **LIBERO:** p ∈ {0, 10, 25, 50, 100}% for both F (saved Stage 12 scores) and G (new generation, 299 demos × 4 prefixes, p > 0).
  - **EgoIntent:** k ∈ {2, 4, 8} frames of the step (A1).
- **Belief update for F.** At prefix j, the belief is
  b_j(g) ∝ b_{j−1}(g) · softmax(s_j / T)(g),
  starting from uniform.
  - The temperature T is chosen on the calibration demos (d ≥ 30, all tasks) from {0.005, 0.01, 0.02, 0.05}, by log-likelihood.
  - The commit threshold θ is chosen on the same calibration demos as the smallest θ in {0.5, 0.6, 0.7, 0.8, 0.9} whose accuracy-at-commit is ≥ 0.9 × the p = 100 accuracy. If none qualifies, use 0.9.
  - The robot commits at the first prefix where max_g b_j(g) ≥ θ; if that never happens, it commits at p = 100.
- **Early commit for G.** The robot commits at the first prefix where the generated sentence's nearest description, by text cosine, is the same at two consecutive prefixes. Otherwise it commits at p = 100.
- **Outputs, descriptive only:**
  - accuracy against p for F (argmax), F (belief) and G (nearest description);
  - for the commit rules: the mean observed fraction at commit, the accuracy at commit, and robot success at commit from a token-matrix lookup (F belief) or from new rollouts (G). Run G at its commit point only for units whose commit sentence differs from the p = 25% one, and record that number;
  - the earliness–accuracy curve over θ.
- **Rule.**
  - **GB1.** With the F belief rule, robot success (token-matrix lookup) − success of Stage 12 `E2E` at fixed p = 25%, paired over the 299 units, has CI lower bound > −5 points (non-inferiority), **and** the mean observed fraction at commit is ≤ 25%.
  - This needs `matrix_seed0.npz`; if the matrix is not finished, report GB1 as pending.

### Stage 13 (human clips, later)

When Trang's clips exist, the Stage 13 analysis also runs G on every clip at p ∈ {25, 50, 75, 100}, plus `E2E-gen-human` rollouts at p = 50% (300 units, the same unit → clip mapping as Stage 13) and both early rules.

## Part 1b: A3, "Open-ended early intention" (from the amendment of 2026-11-12)

The updated pause prompt that holds the full A3 text was not available on the lab machine. The A3 items below are therefore copied verbatim from the amendment's list:

- `gen-libero` at p ∈ {0, 10, 25, 50, 75, 100};
- open-ended 10-way accuracy at each p (nearest of the 10 descriptions, for scoring only);
- the within-scene 2-way for tasks 0 and 1;
- rules GA4 (p = 100 lower bound > 50%), GA5 (p = 50 lower bound > 30%) and GA6 (p100 − p25 > 0);
- `E2E-gen` rollouts at p = 50, reusing the p = 25 episode wherever the sentence is unchanged.
- Gate M1: the nearest-description matcher must put ≥ 90% of the Stage 11 paraphrase and goal phrasings on their own task, checked before any generation;
- `human_check_libero.csv`: 60 sentences for Trang to rate, written before any score is shown.

**GPU order:** probe → `gen-libero` (all 6 p, plus blank) → `E2E-gen` and `E2E-gen-blank` at p = 25 → `E2E-gen` at p = 50 (changed units only) → `gen-ego` → `analyze`.

## Part 2: scope of this run

- **Run now:** A1, A2 and A3, i.e. part A, as the Stage 15 task orders ("Stage 14 part A … A1 and A2 only, no part B"), plus A3 from the amendment.
- **Not run now:** part B and the Stage 13 clip code.
  - They are kept above, verbatim, so that they are fixed in advance.
  - The text-match accuracy of G at p ∈ {0, 10, 25, 50, 75, 100} from A3 is reported. The B commit rules and GB1 are not.
- **Snap variant.** It needs `matrix_seed0.npz`, which is not built yet (the token matrix is paused). It is reported as pending until the matrix exists.
- **Seed 1.** VLIA seed 1 is not finished (paused at `checkpoint_015000`), so only seed 0 is used.

## Part 3: implementation details fixed before any result

### Generator

- **Model.** `Qwen/Qwen2.5-VL-3B-Instruct`, loaded as the `MLLM` class of `scripts/v3_stage3_mllm.py` (bf16, sdpa, CUDA).
- **Message.** Frames are resized with `resize448` and passed in one user message: k image entries, then the prompt text. The chat template uses `add_generation_prompt=True`.
- **Decoding.** `do_sample=False`, `max_new_tokens=40`.
- **Cleaning.** Take the decoded new tokens, strip whitespace, and cut at the first sentence end: the first ".", "!" or "?" that is followed by a space or ends the text. The end mark is kept. Then strip surrounding quote characters (`"`, `'`, `“`, `”`, `‘`, `’`, `` ` ``) and whitespace. An empty result is kept as "" and counts as wrong in every text match.
- **Blank control.** k grey images, RGB (128, 128, 128), 448 × 448.
  - Greedy decoding on an identical input gives an identical output. The blank sentence is therefore generated once per (prompt, k) and applied to every step or unit.
  - EgoIntent: k = 8. LIBERO: k = 8.

### Frames

- **EgoIntent.** As Stage 3: `sample_exact_video_frames` (8 frames) per step; prefix k uses the first k of the 8. Only manifest rows with split train / val (the development steps) are used.
- **LIBERO.**
  - p ∈ {0, 10, 25, 50, 100}: the saved Stage 12 frames `t{t}_d{d}_p{p}.npy`. At p = 0 this is a single frame, so G sees 1 frame there.
  - p = 75: extracted with Stage 12's `prefix_indices` and decoder, into the same folder.
  - No rotation (Stage 12 decision `rotate180 = false`).

### Text matching (scoring only)

- **Text embedding.** SigLIP so400m, computed with `X.text_features` (max_length 64) on the plain sentences (no template), L2-normalised.
- **A1.**
  - Open-ended 5-way is the strict top-1 of the cosine among the 5 `build_kway` candidates.
  - MRR uses the rank of the true WHY among the distinct WHY strings of the same video. Ties count against the true WHY: rank = 1 + the number of other WHYs with cosine ≥ the true one.
  - GA2 compares, step by step, the 5-way hit of the real k = 8 sentence with the 5-way hit of the blank sentence on the same 5 candidates.
- **A2 / A3.**
  - The nearest description is the strict top-1 of the cosine between s and the 10 original LIBERO-10 descriptions (no template). A tie counts as wrong.
  - Within-scene 2-way for tasks 0 and 1: on demos of tasks 0 and 1, the cosine to the true description is strictly greater than the cosine to the other one.
- **Gate M1.**
  - Over the 20 Stage 11 phrasings (`goal` and `para`, `results/stage11_libero_goal/phrasings.json`), the nearest description must be the own task for ≥ 18 / 20 (≥ 90%).
  - It is checked by `run_stage14_generate.py gate-m1` before any generation. If it fails, stop and report.

### Statistics

- **GA1 and GA2.** `CV.video_bootstrap(videos, num, den)` (ratio of sums over resampled videos).
  - GA1: num = the 5-way hits, den = 1 per step, so it is the accuracy CI.
  - GA2: num = hit(real) − hit(blank), den = 1.
  - Steps whose `build_kway` row is −1 (fewer than 5 distinct WHYs in the video) are excluded, as for F. The number used is reported.
- **GA3.** `G.paired_bootstrap` (B = 10,000, seed 0) over the 299 units.
- **GA4, GA5 and GA6.**
  - Over the 299 query demos, bootstrap over demos stratified by task (B = 10,000, `default_rng(0)`, percentile interval), as in Stage 15.
  - GA4: accuracy at p = 100 has CI lower bound > 50%.
  - GA5: accuracy at p = 50 has CI lower bound > 30%.
  - GA6: the paired difference hit(p = 100) − hit(p = 25) has CI lower bound > 0.

### Robot

- **Tokens.** The intention vector of each sentence is computed by `run_stage14_generate.py plan`, with the exact function and template of `v3_stage4_task_embeddings.py`. It is written to `gen_embeddings.npz` (sentence → vector), together with the 10 original descriptions.
  - The rollout looks up the vector of its own sentence there. It is never a row of another task.
  - Gate E1: the 10 original descriptions computed this way equal `task_embeddings_v2.npz` (max abs diff < 1e-6).
- **Conditions** in `libero_goal_ablation.py`, all with an empty instruction:
  - `E2E-gen` (p = 25 sentence);
  - `E2E-gen-blank` (blank sentence);
  - `E2E-gen-p50` (p = 50 sentence).
  - The files are `{condition}_t{t}_e{e}.json`.
- **p = 50 reuse.** For `E2E-gen-p50`, a unit whose cleaned p = 50 sentence equals its p = 25 sentence (exact string) reuses the `E2E-gen` episode. Only the other units are rolled out. Their number is recorded.
- **`human_check_libero.csv`.**
  - 60 (task, demo, p) rows drawn with `default_rng(0)`, without replacement, from all 299 × 6 generated sentences.
  - Columns: task, demo, p, true description, sentence, empty `rating` (correct / partly / wrong). No match column.
  - Written right after `gen-libero`, before any text-match number is computed or printed.
- **`human_check.csv` (EgoIntent).** Written right after `gen-ego` k = 8, before any score.
