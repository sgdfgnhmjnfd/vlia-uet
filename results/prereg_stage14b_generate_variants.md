# Stage 14b pre-registration: a better intention-writing module on LIBERO video (no training, no robot)

Written and committed on 2026-11-16, before any Stage 14b sentence was generated. Stage 14 results are already known; Stage 14b only changes the prompt and the information given to the writer, and the variant is chosen on a calibration set that is disjoint from the 299 test units.

## Question

Does a robot-aware prompt (G1), or a two-step "list the objects, then write the goal" prompt (G2), make Qwen2.5-VL-3B write a sentence that names the right LIBERO-10 task more often than the Stage 14 prompt (G0), without the task list? Separately, how much does it help to give the names of the objects in the scene (G+obj, O1)?

## Data and split

- **Calibration set:** the 80 demonstrations that Stage 12 used to fix τ, i.e. demos d = 30..49 of each task where they exist (task 0: 3, 1: 19, 2: 11, 3: 5, 4: 8, 5: 11, 6: 6, 7: 13, 8: 0, 9: 4). They are not among the 299 units.
- **Test set:** the 299 Stage 12 / Stage 14 units (demos 0..29 of each task; task 8 has 29).
- **Frames:** the Stage 12 frame files (`vlia_outputs/stage12/frames/t{t}_d{d}_p{p}.npy`, agent view), resized to 448 px as in Stage 14.
  - p = 0 is the first frame only.
  - Other p: 8 frames evenly spaced over the first p% of the demo, last included.
- No list of tasks is given to the writer in any variant.

## Writer

- **Model:** Qwen2.5-VL-3B-Instruct, bf16 (the Stage 14 loader).
- **Decoding:** greedy, `max_new_tokens` = 40, first sentence kept (`clean_sentence` of Stage 14, unchanged).
- The prompt asks the answer to start with "I want". Nothing forces that prefix. The share of sentences that start with "I want" is reported.

## Variants (prompts verbatim)

### Family G: no information about the scene

- **G0.** The Stage 14 prompt, unchanged: "These frames show the start of someone doing a household task with objects on a table. In one short sentence, say what they want the objects to end up like. Start with "I want"."
  - On the 299 units the Stage 14 outputs are reused, so only the calibration set is generated.
- **G1.** "These frames, in time order, show a robot arm starting a tabletop task. Look at which object the gripper moves towards or holds, and where things are being moved. In one short sentence, say what the final arrangement of the objects should be. Name the objects and the place. Start with 'I want'."
- **G2.** Two calls.
  - **Call 1**, on the first frame only (the p = 0 frame, the same for every p of a demo, so it runs once per demo): "List the objects on the table and the furniture or containers, as a short comma-separated list."
    - `max_new_tokens` = 60. The list is the first line of the raw answer, stripped of whitespace and of a trailing full stop.
  - **Call 2**, on the frames of the prefix: "Objects in the scene: {list}. These frames, in time order, show a robot arm starting a tabletop task. In one short sentence, say what the final arrangement of the objects should be, using the names above. Start with 'I want'."
- **G3.** G1 or G2 (the better on calibration) with Qwen2.5-VL-7B-Instruct. **Not run.**
  - The safetensors index of the 7B model gives total weights of 16,584,333,312 bytes (15.45 GiB) in 16-bit. The GPU (RTX 2000 Ada) has 16,380 MiB (16.00 GiB) in total.
  - After the CUDA context (about 0.4 GiB) the model cannot run with its activations and KV cache on this GPU in fp16 or bf16. G3 is therefore skipped without downloading the weights. This is recorded in `DEVIATIONS.md`.

### Family G+obj: the robot knows the object names, not the tasks

- **O1.** The G1 prompt with the line "Objects in the scene: {names}." inserted just before its last sentence ("Start with 'I want'.").
- **Where the names come from:** the BDDL file of the LIBERO-10 scene of that task (`libero/bddl_files/libero_10`; the task's file is the one whose name ends with the task description, spaces as underscores).
  - Every declaration in `(:fixtures …)` and then `(:objects …)` is taken, in file order. Each instance name has its trailing `_<digits>` removed and underscores replaced by spaces.
  - Equal names are merged at their first position. A name that occurs n > 1 times is written as "{two/three/…} {plural}", the plural formed by adding "s" ("es" after s, x, ch or sh).
  - The names are joined with ", ".
- The list therefore holds every object of the scene, not only those of the task. Tasks 0 and 1 share LIVING_ROOM_SCENE2 and get the same list.
- `test_stage14b.py` checks both properties: every declared instance is counted, and the lists of tasks in the same scene are identical.
- G+obj is reported as its own row everywhere and is never labelled open-ended.

### Blank-image controls

- Grey 448×448 images (Stage 14 `grey`), 8 for the sentence call and 1 for G2's call 1. Each is generated once per prompt, since greedy decoding is deterministic.
- For O1, one blank sentence is generated per distinct scene list.
- G0's blank is the Stage 14 blank sentence.
- Blank hit for a unit = the nearest description to the unit's blank sentence is the unit's true task.

## Selection (calibration only)

- Every G variant (G0, G1, G2) and O1 is generated on the 80 calibration demos at p = 25 and p = 50.
- Within family G, the variant with the highest mean of its two calibration accuracies is chosen. A tie goes to the lower number (G0 < G1 < G2).
- Family G+obj has one variant, O1.
- `chosen.json` records the means and the choice.
- The chosen G variant (if not G0) and O1 are then run **once** on the 299 units at p = 0, 10, 25, 50, 75, 100, plus their blank controls.

## Scoring (Stage 14 scorer, unchanged)

- SigLIP so400m text encoder (`TextEncoder` of Stage 14).
- The nearest of the 10 task descriptions by cosine, strict top-1 (a tie for the top counts as wrong).
- Sentences and descriptions are encoded as plain text, without the T1 template. This is exactly the Stage 14 `analyze_libero` scorer, so G0 reproduces the Stage 14 numbers. The task text says "with the template" but also "the Stage 14 scorer, unchanged"; the unchanged scorer is kept.

## Rules (fixed now)

- **Bootstrap.** Accuracy CIs use Stage 14's `strat_boot` (stratified by task, B = 10,000, seed 0). Deltas use a paired bootstrap over the 299 units (`libero_goal_ablation.paired_bootstrap`, B = 10,000, seed 0).
- **GB-1.** hit(chosen G) − hit(G0) at p = 25: paired CI lower bound > 0.
- **GB-2.** Accuracy of the chosen G at p = 50: CI lower bound > 30% (Stage 14 rule GA5, unchanged, `strat_boot`).
- **GB-3.** hit(chosen G) − hit(its blank control) at p = 50: paired CI lower bound > 0.
- If G0 is chosen, GB-1 is a fail by construction (delta 0) and is reported as such.
- **Descriptive** (no pass or fail):
  - O1 − chosen G at each p (paired);
  - each chosen variant next to F (Stage 12 C2/T1, p = 0, 10, 25, 50, 100; 69.9% at p = 25) and C (Stage 15 agent primary, nested CV on 379 demos, a different demo set; reference only).

## Descriptive additions

- **Snap.**
  - Per chosen variant and p, `snap_tokens_{variant}.json` is written in the format of `e2e_sup_tokens.json`. Its keys are:
    - `tokens[cond]["t,e"]`, the nearest description text;
    - `pred[cond]["t,e"]`, its index;
    - `reuse[cond]`, the units where pred = true task (the `I-only` episode can be reused, as in Stage 15);
    - `new[cond]`, the other units.
  - `cond` = `E2E-snap-{variant}-p{p}`. On an exact cosine tie the lowest index of the tied ones is used.
  - No rollouts now.
- **Agreement.** Per unit and variant: whether the nearest task at p = 25 equals the nearest task at p = 50. The p = 50 accuracy is reported for agreeing and non-agreeing units (`agreement.csv`).
- **Similarity.**
  - The mean cosine between the sentence and the true description, per variant and p.
  - For G0 only, the Stage 14 robot successes by tertile of that cosine:
    - at p = 25, from the `E2E-gen` episodes;
    - at p = 50, from the `E2E-gen-p50` episodes, with the `E2E-gen` episode for reused units.
- **Examples.** The sentence of each chosen variant (and G0) at every p, for units (task 2, demo 0), (task 5, demo 0) and (task 9, demo 0), fixed now.

## Outputs (`results/stage14b_generate_variants/`)

- `calibration.csv`, `chosen.json`
- `accuracy_by_p.csv` / `.md`, `per_task.csv`, `confusion_{variant}.csv`
- `sentences_{variant}.csv`, `examples.md`, `snap_tokens_{variant}.json`, `agreement.csv`, `similarity.csv`
- `config.json` (prompts verbatim, model ids, commit, GPU minutes), `DEVIATIONS.md`

Sentences are committed as CSV; frames, features and model outputs in JSON form stay in `/media/dhqg/d3/vlia_outputs/stage14b/`.

## Not done here

No training, no robot rollout, no EgoIntent. The 100k training and any robot evaluation wait for a later prompt.
