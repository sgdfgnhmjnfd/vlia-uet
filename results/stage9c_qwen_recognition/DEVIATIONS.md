# Stage 9c: deviations

- The first `analyze` run (chain 2026-10-29 16:52) crashed after all three gates had passed:
  `ValueError: dict contains fields not in fieldnames: 'minus_F_hi', 'minus_F_lo', 'minus_F'`
  (`write_csv` used the keys of the first row only).
- Fix: `write_csv` now uses the union of keys over all rows, with empty cells where a row lacks a key. No computation, method, grid or threshold changed.
- `analyze` was re-run on CPU with the same inputs; the output is in `analyze.log`.
