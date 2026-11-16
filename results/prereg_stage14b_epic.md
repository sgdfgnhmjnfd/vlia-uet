# Stage 14b Part D pre-registration: writing the action on EPIC-KITCHENS-100 (subset)

Written and committed on 2026-11-16, before any video was downloaded and before any feature was extracted. EPIC-KITCHENS-100 is CC BY-NC 4.0. No video, frame or feature is ever committed; only small result files are.

## D0. Data

- **Annotations:** the clone of `github.com/epic-kitchens/epic-kitchens-100-annotations` in `/media/dhqg/d3/data/epic/annotations`.
  - Used: `EPIC_100_train.csv` and `EPIC_100_validation.csv` (segments), `EPIC_100_video_info.csv` (duration, fps), `EPIC_100_verb_classes.csv`, `EPIC_100_noun_classes.csv`.
  - The test file has no labels and is not used.
- **Video selection** (the fixed rule of the task, applied to the annotation files only):
  1. Keep the videos in train ∪ validation with a duration between 3 and 15 minutes.
  2. Take the participants with at least 3 such videos, sorted by id, first 8.
  3. Per participant, take the 3 videos with the most segments (ties: smaller video id).
- The resulting 24 videos (4,867 segments, 267.4 min, all 1920×1080, 50 or 59.94 fps):

| participant | videos (segments) |
|---|---|
| P01 | P01_106 (247), P01_11 (148), P01_02 (145) |
| P02 | P02_118 (290), P02_131 (273), P02_120 (221) |
| P03 | P03_101 (351), P03_117 (259), P03_120 (256) |
| P04 | P04_110 (401), P04_120 (376), P04_104 (294) |
| P05 | P05_07 (111), P05_09 (66), P05_04 (60) |
| P06 | P06_103 (177), P06_108 (171), P06_102 (129) |
| P07 | P07_113 (171), P07_04 (154), P07_117 (131) |
| P08 | P08_01 (156), P08_09 (147), P08_15 (133) |

- **Download.** Videos only, with `epic_video_downloader_verify.py --specific-videos <the 24 ids>` from the official download-scripts repository.
  - This is the README's video-only downloader. It does MD5 checks, resumes `.part` files and retries. It is used instead of `epic_downloader.py --videos` because of those three features; the data source and file layout are the same.
  - **Limits, as in the task:**
    - keep ≥ 60 GB free on `/media/dhqg/d3`;
    - stop after three failures, or 3 hours after the start;
    - Part D runs only if ≥ 12 complete videos from ≥ 6 participants are available. Otherwise it is skipped, with the reason in `DEVIATIONS.md`.
    - If fewer videos are complete, the rule above is applied to the complete videos only.

## D1. Units, split, how early

- **Units.** All segments of the downloaded videos.
  - If there are more than 4,000, 4,000 are drawn with seed 0, stratified by video. Video v keeps ⌊4000·n_v/N⌋ segments, and the remaining slots go to the largest fractional parts (ties: smaller video id). Within a video, segments are drawn with `numpy.random.default_rng(0).choice` without replacement.
- **Folds.** 4 folds over participants. Fold f tests participants 2f+1 and 2f+2 of the sorted list ((P01, P02), (P03, P04), (P05, P06), (P07, P08)) and trains on the other 6.
  - W's inner validation is 10% of the training videos (by video, seed 0, at least 1 video).
- **Frames.** Decoded with PyAV at the timestamps below: the frame whose timestamp is nearest, on the frame grid of the video. 8 frames per unit, frozen SigLIP so400m image embedding (1152-d, normalised; the encoder and preprocessing of the ladder cache).
  - **Early recognition**, p ∈ {25, 50, 75, 100}: 8 times evenly over [start, start + p%·(stop − start)], the last one included.
  - **Anticipation**, g ∈ {0, 0.5, 1.0} s: 8 times evenly over [start − g − 2, start − g − 1/fps].
    - The last frame is therefore strictly before the start of the segment, also at g = 0.
    - A unit is dropped for a g whose window starts before 0 s.
- **Seconds reported** next to every score:
  - early recognition: the mean seconds observed (p%·duration) and the mean seconds until the action ends ((1 − p%)·duration);
  - anticipation: 2 s observed and g s until the action starts.

## D2. Models

- **W**, as in Part B (`prereg_stage14b_writer.md`): connector plus the SmolVLM2-500M language model (or the from-scratch fallback named there), tag `act:` → the `narration` text.
  - A second tag token gives the setting (`early` / `ahead`), so the tag sequence is `early act:` or `ahead act:`.
  - One model per fold, trained on all 7 conditions at once.
  - Greedy decoding, at most 24 new tokens.
- **Reference classifier.** Linear squared-hinge one-vs-rest SVM in torch (the Stage 15 `LinearSVM`; no scikit-learn in either environment) on the mean of the 8 frame embeddings.
  - One SVM for `verb_class` and one for `noun_class`, per fold and per condition.
  - C = 1, not tuned.
  - Classes absent from the training part cannot be predicted.
- **Controls for W:**
  - *blank*: all visual tokens replaced by the mean training embedding;
  - *most frequent*: the most frequent training narration of the fold;
  - *nearest segment*: 1-NN by cosine of the mean frame embedding over training units of the same condition, which outputs that unit's narration.
- **Qwen (descriptive).** Qwen2.5-VL-3B, as in Stage 14: greedy, 40 new tokens, first sentence. Run on 500 test units drawn with seed 0 from the 4,000, at p = 100 and at g = 1.0 only. Prompts verbatim:
  - early: "These frames are from a first-person video in a kitchen. In two or three words, verb then object, say what the person is doing."
  - ahead: "These frames are from a first-person video in a kitchen. In two or three words, verb then object, say what the person is about to do next."

## D3. Scores and rules

- **Text normalisation:** lower-case, punctuation other than `-` removed, split on spaces.
- **Verb correct.** The first word w of the written text is correct when w equals an instance of the true `verb_class`, or the part before `-` of one (e.g. "pick" matches "pick-up"). A word in no class is wrong.
- **Noun correct.** True when some word or adjacent word pair in the rest of the text matches the true `noun_class`:
  - an instance without `:` matches a single word, compared directly or after removing a final "s" or "es";
  - an instance `a:b` matches the pair "b a" or "a b".
- **Action correct:** both are correct.
- **Token F1** with the true narration, on the normalised tokens.
- **Bootstrap:** clustered by participant (8 clusters), B = 10,000, seed 0.
- **E-1.** Action correct of W at p = 100 − the most-frequent control at p = 100: CI lower bound > 0.
- **E-2.** Action correct of W at p = 100 − W at p = 25 (same units, paired within cluster): CI lower bound > 0.
- **Descriptive** (no pass or fail):
  - every other p and g;
  - W against the SVM reference (its action is correct when both classes are right), the nearest-segment control, the blank control and Qwen;
  - verb and noun separately.

## Outputs (`results/stage14b_epic/`)

- `data_summary.md`, `epic_writer.csv` / `.md`, `sentences_epic.csv`, `config.json`, `DEVIATIONS.md`
- Code: `scripts/run_stage14b_epic.py`, `scripts/test_stage14b_epic.py`.
  - The tests check that no participant is in both train and test, that anticipation windows end before the segment starts, verb and noun scoring on hand examples, and that no video or frame file is staged.
