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
