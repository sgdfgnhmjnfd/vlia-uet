# Pre-registration: 100k training and final robot evaluation (written 2026-11-17, before any 100k training step)

Source: "kế hoạch cuối (17/11/2026)", Part 4, with Trang's answers at STOP 3 (17/11): (a) the lab trains VLIA, a rented GPU trains the baseline, and the final evaluation runs on the lab; (b) freeze 22:00 on 22/11/2026; (c) plain VLIA, one sentence per task. Nothing below is changed after results are seen; any change goes to `results/train_100k/DEVIATIONS.md`.

## Models

Both start fresh from `lerobot/smolvla_base`, with the 25k arguments. Only `--steps 100000` and `--save-freq 2500` change; `--output-root` is a location, not a training argument. The 25k `protocol.json` files of the two models differ only in `condition`.

| model | machine | command (inside `venv_env`, cwd `/home/dhqg/lerobot`, `VPY`) |
|---|---|---|
| VLIA (token, empty instruction) | lab, RTX 2000 Ada | `train_intention_only_LIBERO7D.py --condition intention_only --steps 100000 --batch-size 32 --save-freq 2500 --log-freq 10 --seed 0 --output-root /media/dhqg/d3/vlia_outputs/train_100k/vlia` |
| SmolVLA baseline (with instruction) | rented GPU | `train_matched_oracle_LIBERO7D_FIXED.py --condition baseline --steps 100000 --batch-size 32 --save-freq 2500 --log-freq 10 --seed 0 --output-root /media/dhqg/d3/vlia_outputs/train_100k/baseline` |

- **25k arguments**, from `run_v3_chain.sh` and `protocol.json`: `--steps 25000 --batch-size 32 --save-freq 1000 --log-freq 10 --seed 0`.
- **Script defaults**, the same at 25k and 100k: lr 1e-4, min lr 2.5e-6, warmup 1,000, weight decay 1e-4, 4 workers.
- **VLIA variant:** plain VLIA, the 25k recipe, `task_embeddings.npz`, one sentence per task. VLIA-aug is not run.
- **Where outputs go:** all 100k outputs are under `/media/dhqg/d3`, never `/media/dhqg/d1`.

## Gates

- **T0 (resume check)** runs before the real run on each machine. A: 40 steps in one go (`--save-freq 20`). B: 20 steps, stop, `--resume checkpoint_000020`, then 20 more. Compare loss and lr at steps 21–40.
  - If the lr schedule restarts, or the data order restarts, stop and report.
  - The largest differences are reported.
- **T1 (timing).** From the first 1,000 steps on each machine, record seconds per step and the projected finish time: 24 h/day, and 14 h/day for the lab.
- **Freeze.** If both models have not reached 100,000 steps by 22:00 on 22/11/2026, both are evaluated at the last milestone (25k/50k/75k) that both completed. The milestone is chosen by this rule only.
- **Checkpoints and pruning.** Save every 2,500 steps. After a new checkpoint is verified, delete older checkpoints of the same 100k run only. Keep 25k/50k/75k/100k and the two most recent.

## Final evaluation (lab, one machine)

- **Harness:** `scripts/libero_goal_ablation.py`. Episode e of task t uses `init_state_id = e`, seed `1000 + e`, a 520-step limit, and `copy_processors` from `PROC_SRC`.
- **Episodes:** 30 per task, e = 0..29, giving 300 units. E2E-type rows have 299 units: Stage 12 unit list, task 8 has 29 demos.
- **Baseline checkpoint:** copied back to the lab and verified by sha256 before its rows run.

| model | rows |
|---|---|
| VLIA | `I-only`, `I-wrong`, `I-para`, `I-goal`, `E2E` (F, `e2e_tokens.json`), `E2E-sup-p25` (C, `e2e_sup_tokens.json`), `E2E-gen-p25` (token of the G2 p25 sentence, Stage 14b a469873) |
| baseline | `correct`, `empty`, `L-para`, `L-goal`, `L-E2E`, `L-sup-p25`, `L-gen-p25` (the same sentence as the token partner, as the instruction text) |

- **Pairs:** `L-para`↔`I-para`, `L-goal`↔`I-goal`, `L-E2E`↔`E2E`, `L-sup-p25`↔`E2E-sup-p25`, `L-gen-p25`↔`E2E-gen-p25`. Each pair uses the same sentence for the same unit, byte for byte.
- **Reuse:** a row reuses `I-only` (or `correct`) only where its sentence equals the true description.
- **Missing G2 sentences:** if no per-unit G2 p25 sentence exists, both gen rows are skipped. No Qwen run is made.
- **Order:** `I-only`, `I-wrong`, `correct`, `empty`, `L-para`, `I-para`, `L-goal`, `I-goal`, `E2E`, `L-E2E`, `E2E-sup-p25`, `L-sup-p25`, `E2E-gen-p25`, `L-gen-p25`. If time runs short, the run stops only at the end of a row.
- **Learning curve (descriptive).** At each milestone before the final one, 10 episodes per task: baseline `correct` and VLIA `I-only`.

## Rules

Paired bootstrap over units (`paired_bootstrap`, B = 10,000, seed 0); Wilson 95% CIs for single rows.

- **T1:** `I-only` − `empty`, CI lower bound > 0.
- **T2:** `I-only` − `I-wrong`, CI lower bound > 0.
- **T3:** `I-goal` − `I-wrong`, CI lower bound > 0.
- **T4:** `I-para` − `I-wrong`, CI lower bound > 0.
- **T5:** `E2E` − `I-wrong` (299 units), CI lower bound > 0.
- **Token route vs instruction route** (descriptive, no direction fixed): `I-para` − `L-para`, `I-goal` − `L-goal`, `E2E` − `L-E2E`, `E2E-sup-p25` − `L-sup-p25`, `E2E-gen-p25` − `L-gen-p25`, each paired with a CI. The report states plainly which route is better where.
- **Other descriptive differences:**
  - `I-only` − `correct`;
  - `L-para` − `correct`;
  - `L-goal` − `correct`;
  - `E2E-ask` − `E2E` (Stage 12, τ = 0.0106, same asked units);
  - each condition at the final milestone minus the same condition at 25k;
  - learning curve;
  - per-task successes.

## Known differences, stated now

- **Different GPUs.** The baseline and VLIA train on different GPUs. Within-model comparisons are unaffected; cross-model differences include hardware nondeterminism, of the same order as a seed change.
- **Offline dataset load.** The training scripts call the Hub for the dataset version, which fails offline. On both machines `HF_LEROBOT_HOME/lerobot/libero` is a symlink to the cached snapshot `a1aaacb` (ref v3.0), the only revision in the cache, so the data is the same as at 25k. Loss at step 1 equals the 25k probe: 3.584585.
