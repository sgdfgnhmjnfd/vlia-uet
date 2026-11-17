# Stage 16: deviations and notes

1. **t_on was labelled by hand, not by the planned detector** (Trang, 17/11, option a).
   - **The planned detector** (Step 0 item 6): SigLIP so400m embeddings at 4 fps; reference = mean of the frames in [0, 1] s; d_t = mean |z_t - z_ref|; threshold = idle mean + 3 idle sd.
   - **The eye check failed** on the 10-clip contact sheet (`check_t_on_10clips.jpg`). 8 of 10 detections were at 1.25 s, the first frame after the reference window, where no hand is visible. Camera motion (handheld / chest camera) moves the whole frame, and that change already crosses the threshold. 26 of 90 clips had t_on = 1.25 s.
   - **How the hand labels were made:**
     - one annotator (the agent), with frames on a 0.25 s grid from start_s − 1.25 s to start_s, at 420 px per frame;
     - only the video id and the time were shown; no model or detector output was shown;
     - the definition was fixed before looking: t_on = the first grid frame in which part of the acting hand or forearm is inside the frame;
     - body, legs and feet are ignored, and so are a resting other hand and another person's hand at the image edge (t7_r2_p3, t7_r3_p3);
     - in 2 clips no hand is visible before start_s (t0_r2_p3, t0_r3_p3), so t_on = start_s.
   - **Trang checks** 10 clips drawn with seed 0 (`check_manual_t_on_10random.jpg`) before "go".
   - **Files:** the labels are in `t_on_manual.csv`. The detector values stay in `inventory.csv` (`t_on`, flag columns) and are not used.
2. **Hand-visible time is very short.** start_s − t_on (manual) has min 0, median 0.25 s, max 0.75 s. The counts are 0 s: 16, 0.25 s: 42, 0.5 s: 26, 0.75 s: 6. The share of clips with a hand-visible query at each Part A2 cut-off (cut ≥ t_on) is:

   | H | share |
   |---|---:|
   | 2 s | 0% |
   | 1 s | 0% |
   | 0.5 s | 35.6% |
   | 0.25 s | 82.2% |

   So the 2 s and 1 s queries are no-hand (scene-only) by construction. Under the pre-registered 60% rule, H1-d at 0.5 s is "not testable". Both facts were known before any accuracy was computed.
3. **Embeddings for the Step 0 detector.** The first 7 clips were embedded on CPU while VLIA trained. When Trang paused VLIA, all 90 clips were re-embedded on the GPU. The CPU files were kept in `/media/dhqg/d3/vlia_outputs/stage16/step0_cpu_partial/` and are not used.
4. **Table C** is included (filmed 17/11). It is not a deviation. The substitute objects are described in `ban_C/GHI_CHU.md`: a broken induction stove (task 2 = press and hold the power button), a coffee phin on the left and a green cup on the right. These clips are not matched to LIBERO images.
5. **Viewpoint.** No clip differs clearly. Each person used one framing for every task of a table:
   - p1 near;
   - p2 higher and farther, with body and floor visible;
   - p3 mid distance; at table D, p3 used a wider angle with the oven door visible.

   Viewpoint therefore follows the person, not the task. A known small inconsistency remains: the pudding box lies flat in t4_r1_p1 and t6_r3_p1 and stands upright elsewhere.
6. **Datasets folder incident (no effect on Stage 16 data).** At 17:10 on 17/11, `/media/dhqg/d3/datasets` (all other data sets) was moved to the Trash while the clips were copied. With Trang's ok it was restored at about 17:32, before any Stage 16 analysis. `own_libero10` was not affected.
7. **Hand-visible share at H = 0.5: 33.3% here, 35.6% in the prereg.** The prereg share counted all 90 clips. The rule is computed on the 81 shared-table clips (27 of 81 are hand-visible). Both are below 60%, so H1-d is "not testable" either way.
8. **C at the grid edge almost everywhere.**
   - In 563 of 567 LOPO jobs, at least one fold chose an edge C. The grid was then extended once, as pre-registered: to 0.001 in 1,486 folds and to 1,000 in 29 folds.
   - The extension was not repeated. The strongest regularisation usually wins, which is expected with 12–18 training clips per fold.
9. **Excluded queries (fewer than 3 reference frames, or cut < 0): 42.** 36 of them are the start_s − 2.5 and − 2.25 cutoffs of the 18 clips with start_s ≤ 2.5 s; the other 6 are single clips at other early cutoffs. The pooled n per cutoff is in the curve files.
10. **Part B ask rate on held-out persons: 26% (p25) and 23% (p50).** τ was set to give 20% asking on the training persons. The test rate is higher; the random baseline uses the same test rate, so H3-a is unaffected.
11. **τ_gain inside the bootstrap.** In each resample, the per-cutoff paired CI uses the stratified normal approximation, not a nested bootstrap. The point τ_gain uses the bootstrap CI as pre-registered. The onset intervals are quantiles over the 0.25 s grid values, interpolated.
12. **The rules table prints H3-b as (ECE after, ECE before)** in the value and first CI columns. It is not a CI.
