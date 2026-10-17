# Own-dataset recording protocol

This protocol is for building a small, consented egocentric pilot dataset. A
small convenience sample is useful for checking the recording and annotation
pipeline, but is too small to support quantitative generalization claims.

## Annotation CSV schema

Use exactly these columns, in this order:

```text
video_id,participant_id,task_id,rep_id,duration,observation_end,outcome_start,object,target,what,why,next
```

There is one row per video/recording. `video_id` is the video filename stem;
the validator resolves it as `<video_dir>/<video_id>.mp4`. Identifiers and
free-text fields must be non-empty. Times are seconds. The annotation must
satisfy:

```text
0 <= observation_end <= outcome_start - 0.5 <= duration
```

The recorded video duration, computed from frame count and frame rate, must be
within 0.5 seconds of `duration` in the CSV.

## Recording checklist

- Select tasks in which the same initial physical action can plausibly serve
  different goals. The competing goals should diverge after the initial
  observation window, rather than being identifiable from a distinct first
  action.
- Record at least two repetitions per task. Keep repetitions as separate
  videos and give each a unique `video_id` and `rep_id`.
- Before recording each scene, write down a fixed candidate-goal list. Preserve
  that list with the session notes; annotate `target` using that list while
  retaining a free-text `why` description. Do not force a goal into the list
  when the action does not fit; record the actual `why` and flag it for review.
- Record `observation_end` and `outcome_start` in seconds. The observation
  cutoff must precede the outcome by at least 0.5 seconds; do not annotate
  hindsight evidence into `what` or `why`.
- Use filenames `<video_id>.mp4`, with stable unique IDs such as
  `p001_sort_books_r01`. Keep participant identifiers pseudonymous and store
  any re-identification key separately with restricted access.
- Obtain informed participant consent before recording. Explain the intended
  research use, access, retention, and deletion process. Avoid capturing
  faces, screens, documents, or other identifying information when possible;
  restrict access to raw videos and remove or redact incidental sensitive
  content.

The included [`own_dataset_example_FAKE.csv`](own_dataset_example_FAKE.csv)
contains fabricated rows for schema illustration only. Its video files do not
exist and it is not a dataset or experimental result.
