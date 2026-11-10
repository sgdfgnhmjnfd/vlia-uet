# Pre-registration: Stage 13, a real person in the loop (written 2026-11-02, committed 2026-11-10)

Committed before any Stage 13 video reached the machine, so before any extraction, scoring or rollout. The design below is copied verbatim from the plan. There is no training: F is frozen, and the VLIA checkpoint is the Stage 4 one (seed 0).

> **a real person starts a task → F picks the goal among the 10 LIBERO-10 tasks → the token of the picked task → VLIA acts in simulation.**

Trang films herself doing the 10 LIBERO-10 tasks with household objects, with a phone worn at the chest. The videos are grouped by table (scene):

| scene | tasks |
|---|---|
| A | 0, 1, 7 |
| B | 4, 6 |
| C | 2, 8 |
| D | 3, 9 |
| E | 5 |

- At one table, every object stays in the same place in every clip. So the scene alone cannot say which of that table's tasks she is about to do.
- She stands still for about 2 s before acting.

`labels.csv` has the columns `video_id,task,scene,rep,person,start_s,end_s`.
- Person `p1` is Trang; she has 30 clips named `t{task}_r{rep}.mp4`.
- Clips with the suffix `_p2` are a second person and are optional.
- `start_s` is when the hand first touches an object and `end_s` is when the task is done. Both are hand-marked to about 0.5 s.

## Query clips

- **Clip set.** The primary set is the 30 clips of person `p1`, one per (task, rep) with rep ∈ {1, 2, 3}.
  - **Gate L1.** All 30 clips exist.
  - Each task is at the scene given in the table above.
  - 0 < `start_s` < `end_s` ≤ the clip duration.
  - `start_s` ≥ 0.5 s, so the clip has an idle part.
  - If any of these fails, stop and report. Do not fix labels yourself.
- **Orientation check (gate O1).** Before any scoring, save a contact sheet to `/media/dhqg/d3/vlia_outputs/stage13/contact_sheet.png`, not to git. It holds the first frame and the frame at `start_s` of every clip. Look at it.
  - If the frames are rotated (phone rotation metadata not applied), fix the rotation for every clip.
  - Decide this from the picture only, never from a score. Record the decision in `config.json`.
- **Prefix p.** The observation window runs from the clip start (0 s) to `start_s + p·(end_s − start_s)`, with p ∈ {0, 25, 50, 75, 100}%.
  - Take 8 frames evenly spaced over the window, always including its last frame.
  - p = 0 is the idle part only: the table before the hand moves.
  - **The primary prefix is p = 50%.** A person's motion is slower and less direct than a robot demo, so half of the action is the fair point.
- **Preprocessing.** Resize with the same SigLIP image processor as the ladder preprocessing. Do not crop by hand.

## F (frozen, no change)

- **Model:** SigLIP so400m-patch14-384, L2-normalised image embeddings, mean over the 8 frames.
- **Candidates:** the 10 original LIBERO-10 task descriptions in template T1, the text side exactly as in Stage 12.
  - **Gate E1:** the candidate text embeddings equal the token table rows (max abs diff < 1e-6).
- **Scene-context removal, primary (C-scene).** Subtract the mean frame embedding of the **idle parts of the other clips of the same scene and the same person** (leave-one-out; 8 frames per idle part).
  - This uses the scene and person labels, which a robot knows (it knows which table it is at), but no task label.
  - For scene E (one task), this is still well defined: the other two reps of task 5.
- **Descriptive only** (no rule, never used to pick the token):
  - C0: no context removal;
  - C-libero: the fixed non-LIBERO-10 mean from Stage 12, reused from its saved file;
  - T0: the bare task text.
- **Outputs:** prediction = top-1 task; confidence margin = top-1 score minus top-2 score.

## Two accuracy measures

- **10-way accuracy:** top-1 among all 10 tasks, where a tie counts as wrong. Uses all 30 clips.
- **Within-scene accuracy:** top-1 among only the tasks of the clip's own scene. Uses the 27 clips of scenes A–D; scene E has one task and is excluded. Chance is 1/3 for scene A and 1/2 for B, C and D, which gives 11/27 = 40.7% on average.
  - This answers the question P2 could not: once the scene is known, does the motion tell the task?

## Robot conditions

All use the seed-0 VLIA checkpoint and an empty instruction, on the 300 units (task t, e = 0..29).

For unit (t, e), the query is Trang's clip of task t with rep r = (e mod 3) + 1, at p = 50%, under C-scene.

| condition | intention token |
|---|---|
| `E2E-human` (**new**) | T1 embedding of the **original description of the task F predicts** (10-way) for that clip |
| `E2E-human-ask` (computed, no new runs) | if the clip's margin < τ_h, the robot asks and the person names the task, so the token is correct (the `I-only` episode); otherwise the `E2E-human` episode |
| `I-only` (existing) | correct task's description |
| `I-wrong` (existing) | description of task (t+5) mod 10 |

- τ_h is the 25% quantile of F's margins over the 30 clips at p = 50%, C-scene. It uses no labels and is fixed before any rollout.
  - There is no separate calibration set of human clips, so this is a stated limitation.
- `z_int_task` in each `E2E-human` episode JSON must hold the predicted description string.

## Pre-registered rules

Recognition CIs come from a bootstrap over clips clustered by task (B = 10,000, seed 0), the same scheme as Stage 12. Robot deltas use `paired_bootstrap` over the 300 (task, episode) pairs.

- **H1 (F reads a real person's task).** 10-way accuracy at p = 50% on the 30 clips has CI lower bound > 10% (chance).
- **H2 (it uses the motion, not the scene).** Within-scene accuracy at p = 50% minus within-scene accuracy at p = 0, paired over the 27 clips of scenes A–D, has CI lower bound > 0.
- **H3 (the loop carries a real person's task).** success(`E2E-human`) − success(`I-wrong`) has CI lower bound > 0.
- **Descriptive**, with no pass/fail:
  - 10-way and within-scene accuracy for every p, under C-scene, C0 and C-libero, and with T0;
  - within-scene accuracy at p = 50% against its chance level (40.7%), with CI;
  - the 10 × 10 confusion matrix at p = 50%;
  - `E2E-human` − `I-only` and `E2E-human` − Stage 12 `E2E` (unpaired across query sets, so report both rates side by side);
  - `E2E-human-ask` − `E2E-human`, and the ask rate;
  - the success of `E2E-human` split by whether F was right;
  - H3 again with a bootstrap clustered by clip (30 clusters), because each clip's prediction is reused for 10 units, so the unit-level CI is too narrow;
  - **D1:** for units where F is right, how often `E2E-human` and `I-only` have the same success (run-to-run noise; Stage 12 found 92.8%).
- **Optional p2 clips**, if present: recognition only (10-way and within-scene at every p, with the LOO mean taken over p2's own idle parts). Descriptive; no robot runs.

## Optional: change-of-mind clips (descriptive only, no robot runs)

If `switches.csv` and its clips (`s{a}to{b}_r{1,2}.mp4`, columns `video_id,scene,task_first,task_final,start_s,switch_s,end_s`) exist, score them with the same F, C-scene and template. Use the leave-one-out idle mean from the 30 primary clips of the same scene and person. They never enter H1–H3 or τ_h.

- **Before the switch.** The window ends at `start_s + 0.5·(switch_s − start_s)`. Report how often the within-scene prediction is `task_first`.
- **After the switch.** The window is only the last 4 s before `end_s`; unlike the primary clips, it does not start at 0 s. Report how often the within-scene prediction is `task_final`.
- Also report the per-clip margin before and after. This tests whether F follows a change of intention, and whether its confidence drops near the switch.

## Implementation details fixed before any video (2026-11-10)

- **Frame times.** For a window [a, b] (in seconds), the 8 frame times are `linspace(a, b, 8)`, and frame index = min(round(t · fps), n_frames − 1). The last time is b, so the frame at the window end is always included. p = 0 gives the window [0, `start_s`].
- **Idle parts.** The idle part of a clip is its p = 0 window: 8 frames over [0, `start_s`].
- **Change-of-mind context.** It uses the idle parts of all 30 primary clips of the same scene and person. The change-of-mind clips are never part of the 30, so leave-one-out excludes nothing.
- **Decoder.** The first decoder that works in the order given by the plan: `cv2` in `PY`, then `cv2` in `VPY`, then `torchvision.io` / `av`, then the `ffmpeg` binary. Frames are read sequentially; there is no seeking.
