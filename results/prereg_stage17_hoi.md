# Pre-registration: Stage 17, hand-region (HOI) features on the 90 own clips (written 2026-11-18, before any number)

- **Source:** the Stage 17 prompt of 18/11.
- **Code and data** reused from Stage 16 (commit e41e72c, Stage 16 prereg sha256 4e041e2e):
  - `scripts/run_stage16_scene_motion.py`: frame grid, windows, S/M/F+, LOPO, C grid, bootstrap, onset;
  - the 90 clips in `/media/dhqg/d3/datasets/own_libero10`;
  - the manual t_on in `results/stage16_scene_motion/t_on_manual.csv`;
  - the Stage 16 full-frame SigLIP cache in `/media/dhqg/d3/vlia_outputs/stage16/emb/`.
- **Later changes** go to `results/stage17_hoi/DEVIATIONS.md`.

## What this stage can and cannot show (fixed now)

- **Hand timing:** the hand enters the frame only 0–0.75 s before first contact (median 0.25 s; Stage 16). At start_s − 0.5 s only about a third of the clips show a hand.
- **What HOI cannot show:** the hand region cannot help to anticipate the task before contact.
- **What HOI can show:** any gain can only be earlier or better recognition **after** contact.
- **The claim stays,** whatever the result: task recognition from human motion right after contact, not pre-contact anticipation.

## G0: can MediaPipe see the hand? (CPU only)

- **Detector settings (fixed; not tuned after results):**
  - MediaPipe Hands (legacy `solutions.hands`, mediapipe 0.10.21, in a separate venv `/media/dhqg/d3/venv_mp`);
  - `static_image_mode=True`, `max_num_hands=2`, `model_complexity=1`, `min_detection_confidence=0.5`.
- **Frames:** the Stage 16 4 fps grid (t = 0, 0.25, …, nearest video frame, read with cv2; all clips have rotation 0), RGB.
- **A frame is "hand detected"** if at least one hand is returned.
- **Windows per clip:**
  - **primary (gate):** [start_s, start_s + 2.0];
  - descriptive:
    - [t_on, start_s): hand visible before touch; empty when t_on = start_s;
    - [start_s, end_s]: the whole action;
    - [0, t_on − 0.25]: the idle table, as a false-positive check.
- **Output:** `results/stage17_hoi/g0_detection.csv` with columns video_id, table, task, person, rep, window, n_frames, n_detected, rate. Summaries by table, by person and by window.
- **Eye check:** 10 random clips (numpy `default_rng(0)`, choice of 10 without replacement from the sorted clip ids). Each gets a contact sheet of its primary-window frames with the boxes drawn (`/media/dhqg/d3/vlia_outputs/stage17_hoi/g0_sheets/`, not in git). The report counts boxes on a real hand, missed hands and false boxes.
- **Gate G0:**
  - **PASS** needs all three:
    - the pooled detection rate in the primary window is ≥ 60%;
    - every table A–E is ≥ 30%;
    - at most 2 of the 10 sheets have boxes mostly not on hands.
  - **FAIL:** stop and report. The thesis result is then "MediaPipe does not find the hand reliably on these top-down own videos, so HOI is not testable here". No other detector or setting is tried without Trang's ok.
- **STOP 1** after G0.

## Step 2 (only after G0 PASS and Trang's go)

### 2a. Crops (CPU)

- **Box per 4 fps frame:** the union of the landmark boxes of all detected hands. It is expanded to 2.0× its larger side around its centre, made square, clipped to the frame, with a minimum side of 128 px.
- **Missing detections:** if a frame has no detection, the last box is carried forward for at most 1.0 s; after that the frame has no crop.
- **Saving:** crops are resized to 384×384 and saved to `/media/dhqg/d3/vlia_outputs/stage17_hoi/crops/` (not in git). The share of frames with a crop is logged per clip.

### 2b. Embeddings (GPU, only while VLIA 100k is paused)

- **How VLIA is paused:** the chain's PAUSE file, exactly as Stage 16:
  - `touch /media/dhqg/d3/vlia_outputs/train_100k/vlia/PAUSE`;
  - the chain stops training right after the next complete checkpoint;
  - `rm` of the file resumes.
- **Pause check:** before the pause, write the last step and the last checkpoint to STATE. After resuming, confirm that training resumed from that checkpoint, with data-order fix B active. Report "steps before pause = X, resumed from = X, lost = 0".
- **Embeddings:** the same frozen SigLIP so400m-patch14-384, L2-normalised, cached to `/media/dhqg/d3/vlia_outputs/stage17_hoi/emb/`. The full-frame embeddings are the Stage 16 cache, not recomputed.

### 2c. Reproduction gate R0 (CPU)

The Stage 16 code and cache must reproduce, within 0.1 point: S = 42.0, M p50 = 67.9, F+ p50 = 74.1 (pooled shared tables). Otherwise stop and report.

### 2d. Features and protocol (identical to Stage 16 except the new features)

- **p levels:** p ∈ {25, 50, 100} of [t_on, t_end]. Observed frames are those of Stage 16 (at most 16, evenly spaced).
- **H (hand only):** the mean of the crop embeddings over the observed frames that have a crop.
  - If a query has no crop at all, H is imputed with the training-fold mean of H (computed inside each LOPO fold, training clips only). The query is counted in `n_no_crop`.
  - The same imputation applies to the H part of F+H.
- **F+H** = [S, mean full-frame z_t over the observed frames, H].
- **Baselines:** S, M, F+ exactly as in Stage 16.
- **Classifier and statistics, all as in Stage 16:**
  - within-table training, LOPO over persons;
  - C grid {0.01, 0.1, 1, 10, 100} with inner leave-one-rep-out and the one-time edge extension;
  - standardisation on the training fold only;
  - Wilson CIs;
  - paired bootstrap over clips within table (10,000 resamples, seed 0).
- **Tables:** the rules use the shared tables A–D only (pooled chance 0.444). Table E is excluded.

### Rules (fixed now)

- **H4-a (main):** pooled acc(F+H, p25) − acc(F+, p25), paired. The 95% CI lower bound must be > 0.
- **H4-b:** pooled acc(H, p50). The Wilson 95% lower bound must be > 0.444.
- **H4-c (earlier):**
  - **Per-clip stable-recognition time τ after contact:** on the grid start_s + c, c ∈ {0, 0.25, …, 2.0}, one LOPO model per cutoff and feature set (Stage 16 cut-based queries: reference [0, min(1, t_on − 0.25, cut)], observed (r, cut]). τ is the earliest c from which the prediction is correct at every later cutoff of the grid.
  - **Clips with no stable recognition** get τ = 2.0 (the end of the grid) and are counted separately.
  - **Test statistic:** the median over clips of τ(F+H) − τ(F+). The bootstrap 95% CI (clips resampled within table, 10,000, seed 0) must have an upper bound < 0.
- **Descriptive only:**
  - per-table results (especially A and B);
  - per-person results;
  - the confusion matrix at p50 for F+H;
  - accuracy against seconds after contact for S, M, F+, H, F+H, plotted with no video frames;
  - the pre-contact cutoffs start_s − 0.5 and − 0.25 for H and F+H, with the number of queries that actually have a crop. No pass/fail and no anticipation wording.
- **Causality:** no frame after the cutoff enters a query. The Stage 16 unit test is extended to the crop frames.

### Interpretation (fixed now)

- **H4-a PASS:** looking at the hand region helps recognise the task earlier after contact.
- **H4-a FAIL, H4-b PASS:** the hand region carries task information, but the full frame already has it.
- **H4-b FAIL:** on these clips the hand crop is not informative (crops too small, hand blur, or detection gaps). The crop coverage is reported next to it.

## Outputs

`results/stage17_hoi/`:
- `g0_detection.csv`
- `accuracy.csv`
- `rules.md`
- `per_person.csv`
- `confusion_p50.csv`
- `stable_tau.csv`
- `curve_seconds.png`
- `config.json`
- `DEVIATIONS.md`
- `pause_check.md`
