# Pre-registration: Stage 11, does the robot understand an intention phrased as a goal? (2026-10-30)

Written and committed before any Stage 11 rollout. No training: the existing checkpoints are only evaluated. The phrasings and rules below are copied verbatim from the task. They will not be edited, tuned or re-worded after any result.

## Phrasings (`results/stage11_libero_goal/phrasings.json`)

Use the keys `goal` and `para`, indexed by LIBERO-10 task id 0–9, in the order of `results/libero_goal_ablation_v1/instructions.json`.

| task | original | goal (person's desired end state) | para (re-worded instruction) |
|---:|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | I want the alphabet soup and the tomato sauce to end up in the basket | place the alphabet soup and the tomato sauce into the basket |
| 1 | put both the cream cheese box and the butter in the basket | I want the cream cheese box and the butter inside the basket | move the cream cheese box and the butter into the basket |
| 2 | turn on the stove and put the moka pot on it | I want the stove switched on and the moka pot sitting on it | switch on the stove and place the moka pot on top of it |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | I want the black bowl stored in the bottom drawer of the cabinet, with the drawer shut | place the black bowl into the cabinet's bottom drawer and shut it |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | I want the white mug on the left plate and the yellow and white mug on the right plate | set the white mug on the left plate and the yellow and white mug on the right plate |
| 5 | pick up the book and place it in the back compartment of the caddy | I want the book stored in the back compartment of the caddy | take the book and put it into the rear compartment of the caddy |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | I want the white mug on the plate and the chocolate pudding just to the right of it | place the white mug on the plate and set the chocolate pudding to the plate's right |
| 7 | put both the alphabet soup and the cream cheese box in the basket | I want the alphabet soup and the cream cheese box to end up in the basket | place the alphabet soup and the cream cheese box into the basket |
| 8 | put both moka pots on the stove | I want both moka pots sitting on the stove | place the two moka pots on the stove |
| 9 | put the yellow and white mug in the microwave and close it | I want the yellow and white mug inside the microwave with the door closed | place the yellow and white mug into the microwave and shut the door |

## Conditions

All conditions use the same 300 paired units: 10 tasks × episodes e = 0..29.

| condition | checkpoint | language instruction | intention vector |
|---|---|---|---|
| `I-only` | VLIA | "" | T1 embedding of the original task description (as Stage 4) |
| `I-wrong` | VLIA | "" | T1 embedding of the original description of task (t+5) mod 10 |
| `I-goal` | VLIA | "" | T1 embedding of the `goal` phrasing of task t (**new**) |
| `I-para` | VLIA | "" | T1 embedding of the `para` phrasing of task t (**new**) |
| `correct` | SmolVLA baseline | original instruction | — |
| `empty` | SmolVLA baseline | "" | — |

Embeddings use exactly the Stage 4 recipe:
- template T1, "a first-person photo of a person doing: {text}";
- so400m, max_length 64, L2-normalised.

## Pre-registered rules (paired over (task, episode); 95% CI by paired bootstrap, B=10000, as in `paired_bootstrap`)

- **G1 (goal phrasing carries the task):** success(I-goal) − success(I-wrong) has CI lower bound > 0.
- **G2 (re-wording carries the task):** success(I-para) − success(I-wrong) has CI lower bound > 0.
- **Descriptive, no pass/fail:**
  - I-goal − I-only;
  - I-para − I-only;
  - I-only − correct (the Stage 4 claim with 300 episodes);
  - I-only − empty.
- **Text-side diagnostic** (CPU, no robot), for each phrasing:
  - the cosine similarity to its own original embedding;
  - the rank of its own task among the 10 original embeddings (1 = nearest).
  - Report per task, together with the success counts. This is the explanation if a task fails.

## Gates

- **E1:** for every original task string, the v2 embedding (`task_embeddings_v2.npz`) equals the v1 embedding (max abs diff < 1e-6). `task_embeddings.npz` is not overwritten.
- **Self-test:** `scripts/test_stage11.py` passes before any rollout.
- **Smoke:** 1 `I-goal` episode on task 0 (`--probe 1`, temporary out dir); its `z_int_task` holds the goal string.
- **R1:** before the quick look, 3 existing `I-only` episode JSONs are re-loaded; for each, the v2 table entry for its `z_int_task` equals the v1 entry for the original string of its task (max abs diff < 1e-6), and `z_int_task` equals that original string.
- **R2:** for e = 0..9, the re-analysed I-only, I-wrong, correct and empty counts equal Stage 4 exactly: 20, 0, 19 and 0 per 100.

## Run order

1. Quick look: `I-goal` and `I-para`, e = 0..9 (200 episodes), VLIA eval dir.
2. Extend VLIA to e = 0..29: `I-only,I-wrong,I-goal,I-para` (existing JSONs are skipped).
3. Then, not at the same time, extend the baseline to e = 0..29: `correct,empty`, in the `libero_goal_ablation_v1` dir.
4. `--analyze-stage11`.

## Correction to the task text (disclosed before any rollout)

The task names the baseline checkpoint as `/media/dhqg/d3/lerobot_outputs/smolvla_libero/checkpoints/025000/pretrained_model` ("the 24% reference").

The existing baseline episodes (`/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/episodes`, e = 0..9, correct 19/100) and the 24% reference eval were produced with a different checkpoint:
- `/media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/baseline/seed_0/checkpoint_025000/pretrained_model`;
- sources: `results/libero_goal_ablation_v1/config.json`, `run_meta.json` and `INTERPRETATION.md`, and the "Model directory" line of `/media/dhqg/d3/logs/libero_goal_ablation_v1.log`.

The two `model.safetensors` differ:
- md5 `9628e448…` (d1);
- md5 `03c783ec…` (`lerobot_outputs`).

Their pre/post-processor JSONs are identical.

Episodes e = 10..29 must be paired with e = 0..9 and with R2. The baseline extension therefore uses the **d1 checkpoint**, i.e. the checkpoint that actually produced the 19/100 and the 24% reference. The `lerobot_outputs` checkpoint is not used.
