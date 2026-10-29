# Pre-registration: anticipation in real seconds (written 2026-10-28, before any number)

Script: `scripts/run_time_horizon.py`. Data: EgoIntent train/val (27 videos); the locked test is never read.
Motivation: the 8-frames-per-step cache makes "H frames" depend on step length; the lab group asked for
time in seconds and a longer view than "8 frames".

- Frames re-sampled from every step clip at 4 frames/s (resized to 384x384, SigLIP so400m image tower,
  fp16 on GPU); a video's timeline is its steps concatenated in step order (gaps between clips ignored).
- Anchors: every 1 s from t = 16 s to T - 20 s; the SAME anchors for every context and horizon.
- Context C in {1, 2, 4, 8, 16} s: query = mean of the C2-centred frames in (t - C, t] (minus the mean of
  all earlier frames of the video; global mean if none). Horizon H in {0.5, 1, 2, 4, 8, 12, 16, 20} s.
- Target = WHAT (also WHY) of the step containing t + H; 5 candidates = target + 4 distinct labels of the
  same video, seeded by (video, anchor, H), hence identical across C. Strict top-1, ties wrong.
- Methods: ours (F scoring), copy_rec (recognise the current step with the same context, pick the
  candidate closest in SigLIP text space), copy_true (true current label).
- Reported: accuracy with video-bootstrap 95% CI (B = 2000); paired consecutive-H differences per C
  (plateau = CI contains 0) and paired consecutive-C differences per H ("does seeing longer help").
- Questions, no pass/fail: (Q1) is there a context length that beats C = 2 s (~one step) at H >= 2 s?
  (Q2) where does accuracy flatten in seconds? (Q3) does any C make ours beat copy_rec at H >= 4 s?
