# Stage 17: deviations and notes

1. **MediaPipe version.** mediapipe 1.2.3, the current release, no longer has the legacy `solutions.hands.Hands` API with the pre-registered settings. Its Tasks API needs a separately downloaded model file. mediapipe **0.10.21** was installed instead, in the separate venv `/media/dhqg/d3/venv_mp` (Trang's ok, 18/11). That version still has the legacy API and its bundled model, and the settings are unchanged. The venv also holds opencv 4.11 (pulled in by mediapipe) and numpy 1.26. pip reports a numpy-version conflict with pandas and opencv-headless 5.1, which are installed but not used.
2. **VLIA was not training at the start of the session.** The machine had rebooted at 10:02. VLIA was restarted at 10:06 with Trang's ok. It resumed from checkpoint_002500 with data-order fix B active: the loss at step 2,510 equals the earlier runs, 0.471761.
3. **G0 CPU load.** G0 ran on 4 cores (taskset 12–15), nice 19, for about 2 minutes. The training speed was 1.85 s/step before and about 2.0 s/step during G0, read from the 60 s progress log with its 10-step granularity. That is within the 10% limit, up to this measurement precision.

## G0b: exploration outside the prereg (Trang, 18/11). Written before any G0b run.

G0 failed under the pre-registered settings. Everything below is exploratory: it is chosen after seeing the G0 numbers and is reported as such.

- **Thresholds (unchanged).** Primary window [start_s, start_s + 2]:
  - pooled detection rate ≥ 60%;
  - every table A–E ≥ 30%;
  - eye check of 10 clips (seed 0, same clips as G0): at most 2 sheets with boxes mostly not on hands.

  The idle-table false-positive count ([0, t_on − 0.25]) is reported for every configuration.
- **Configurations,** at most 4. All run on CPU only, in venv `/media/dhqg/d3/venv_mp`, nice 19, at most 4 cores (taskset 12–15), on the same 4 fps frames as G0.
  - **(i)** MediaPipe Hands (0.10.21 legacy): `static_image_mode=True`, `max_num_hands=2`, `model_complexity=1`, `min_detection_confidence=0.3`.
  - **(ii)** As (i), on a 2× zoom: the frame is cut into 2 × 2 overlapping tiles, each 0.6 W × 0.6 H, at the corners. Hands is run on each tile, and the boxes are mapped back to the frame. A frame is "detected" if any tile returns a hand.
  - **(iii)** MediaPipe Tasks HandLandmarker, IMAGE mode:
    - `num_hands=2`;
    - `min_hand_detection_confidence=0.3`, `min_hand_presence_confidence=0.3`, `min_tracking_confidence=0.3`;
    - model `hand_landmarker.task` (float16, the official MediaPipe model file, downloaded once to `/media/dhqg/d3/vlia_outputs/stage17_hoi/models/`).
  - **(iv)** Only if (i)–(iii) all fail the rate thresholds: one pretrained CPU hand detector with weights available without training. It is named here before it is run; nothing is trained.
- **Selection rule (fixed now).**
  - Among the configurations whose idle false-positive rate on tables A–C is ≤ 5%, the best one has the highest pooled primary-window detection rate on tables A–C. Ties go to the lower roman number.
  - Tables D–E are reported separately for the chosen configuration, as a check that was not used for selection.
  - The full gate (all of A–E) is then stated for the chosen configuration.
- **Load guard.** Training speed is read from the VLIA progress log during each run. If it is more than 10% slower than before G0b (1.85 s/step, so above 2.03 s/step over a 5-minute window), G0b is stopped at once.
- **Stop.** Step 2 (hand crops + SigLIP, GPU) is not started without Trang's ok, even if a configuration passes.

### G0b results (18/11, 12:27–12:37; CPU only; training speed 1.846 s/step before and after each configuration)

- **Primary-window detection rates:**

  | configuration | all | A–C | D–E | A | B | C | D | E | idle detections |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | G0 (prereg, conf 0.5) | 47.2 | – | 37.0 | 56.4 | 49.4 | 46.3 | 38.3 | 34.6 | 0/932 |
  | (i) conf 0.3 | 52.6 | 57.0 | 42.4 | 60.1 | 55.6 | 53.7 | 43.8 | 39.5 | 0/932 |
  | **(ii) conf 0.3, 2 × 2 tiles (2× zoom)** | **63.2** | **66.5** | **55.6** | 71.6 | 54.3 | 71.0 | 55.6 | 55.6 | 8/932 |
  | (iii) Tasks HandLandmarker 0.3 | 56.0 | 57.1 | 53.5 | 58.4 | 58.0 | 54.3 | 54.3 | 51.9 | 0/932 |

- **Chosen on A–C:** (ii), with A–C 66.5% and an idle rate on A–C of 0.8% (≤ 5%). On the held-out tables D–E it reaches 55.6% (D 55.6, E 55.6).
- **Full gate for (ii):** all 63.2% ≥ 60% and every table ≥ 54.3% ≥ 30%, so the rate part passes.
- **(iv) was not run,** because (ii) passed the rate thresholds.
- **Eye check of (ii),** the same 10 clips: every box is on a real hand, so 0 of 10 sheets have boxes mostly off hands, and the eye check passes.
  - n = 2 marks the same hand found in two overlapping tiles.
  - Misses remain with dark long sleeves (t2_r3_p3, t4_r3_p1) and with arms close to the camera (t7_r2_p3).
- **The 8 idle detections of (ii):**
  - 4 true false positives: the orange butter box (t1_r3_p2, 1 frame) and a shoe on the floor (t5_r1_p2, 3 frames);
  - 4 real but non-acting hands: another person's hand at the top edge (t7_r3_p3, 3 frames) and a resting hand at the bottom edge (t6_r3_p2, 1 frame).
- **Status:** this is an exploratory pass, not the pre-registered G0. Step 2 waits for Trang's ok, because it needs the GPU.

## Step 2 (exploratory; Trang's ok 18/11)

1. **Outside the prereg.** Config (ii) passed only in G0b, after the pre-registered G0 failed, and on the held-out tables D–E it reaches 55.6% < 60%. All Step 2 numbers are exploratory.
2. **Frames with no hand crop.** On such a frame the full-frame embedding of the same frame is used (Trang, 18/11), not the prereg's training-mean imputation. Share of observed frames that fell back to the full frame: pooled 15.9% at p25 and 15.2% at p50. By table at p25: A 10.9, B 37.4, C 11.5, D 6.8. By table at p50: A 8.8, B 29.3, C 11.5, D 15.2. Crops cover 61.9% of all grid frames (39.2% detected, 22.8% carried forward for at most 1 s).
3. **Scope.** Trang asked for F+ vs F+H at p25 and p50, so H4-c (the per-clip stable-recognition time) was not run. H alone is reported as H4-b.
4. **Pause.** VLIA paused right after checkpoint_010000 and resumed from step 10000 with data-order fix B active. Lost = 0; steps 10000–10010 were recomputed. See `pause_check.md`.
5. **Analysis run.** The first analysis run used one process with all torch threads, about 14 cores. VLIA slowed by about 4% (1.93 vs 1.85 s/step), within the 10% limit, but the run was slow, so I stopped it after 15 minutes. It was rerun with the same code in 6 single-thread processes (10 s). No result of the stopped run was used.
6. **Statistical note.** The H4-a paired bootstrap lower bound is exactly 0.000 (+9.9 points [0.0, 19.8]), so H4-a is "FAIL" by the rule (> 0 required). Exact McNemar: 12 clips correct only with F+H and 4 correct only with F+, p = 0.077.

## H4-c (exploratory, Trang's ok 18/11). Written before running.

- **Inputs, unchanged:** the existing embeddings (Stage 16 full-frame cache and the config (ii) hand-crop cache). No new configuration or threshold. The no-crop frame rule is the same as Step 2 (full-frame embedding of that frame). CPU only, nice 19, 4 processes; the GPU and VLIA are not touched.
- **Grid:** cut = start_s + c, c ∈ {0, 0.25, …, 2.0} (9 cutoffs). Queries are the Stage 16 cut-based ones: reference [0, min(1, t_on − 0.25, cut)], observed frames in (r, cut], at most 16. H and F+H use the same observed frames. One Stage 16 LOPO model (within table) per cutoff and per feature set (S, M, F+, H, F+H).
- **Per-clip τ (the prereg H4-c):**
  - τ is the earliest c from which the clip's prediction is correct at every later cutoff of the grid;
  - a clip with no such c gets τ = 2.0 and is counted as "no stable recognition";
  - the statistic is the median over clips of τ(F+H) − τ(F+), with a bootstrap 95% CI (clips resampled within table, 10,000, seed 0);
  - "earlier" means the CI upper bound is < 0.
- **Pooled / per-table onset (the Stage 16 definition):** the earliest cutoff at which the Wilson lower bound of accuracy is > chance at that and every later cutoff, with a bootstrap 95% interval (2,000 resamples of clips within table, seed 1). Pooled chance is 0.444; per table it is 1/k.
- **Reporting:** per table (A–D) and pooled; per-clip τ medians for each feature set, the share with no stable recognition, and the paired median difference with its CI.

### H4-c results (18/11; CPU, 4 processes, nice 19, 65 s)

- **Pre-registered statistic:** the pooled median per-clip τ(F+H) − τ(F+) is 0.00 s [0.00, 0.00], so **FAIL**. The median sits at 0 because many clips tie (the same τ for both, often 2.0, "no stable recognition").
- **Descriptive only, not pre-registered:**
  - the mean difference is −0.27 s;
  - F+H is stable earlier in 28 clips and later in 4 (sign test, two-sided p = 2 × 10⁻⁵);
  - clips with no stable recognition: F+ 34/81, F+H 21/81, H 17/81.
- **Stage 16 onset** (pooled, Wilson lower bound > chance, stable): F+ +1.25 s [0.5, 1.75], the same value as Stage 16; F+H +0.25 s [0, 0.75]; H 0.0 s [0, 0.25].
