# Final 100k evaluation: what runs and when (written 19/11/2026)

**Episodes to roll out per model.** Reused units are not rolled out (`unit_sentences.json`).

| model | row | episodes |
|---|---|---:|
| VLIA | I-only | 300 |
| VLIA | I-wrong | 300 |
| VLIA | I-para | 300 |
| VLIA | I-goal | 300 |
| VLIA | E2E | 299 |
| VLIA | E2E-sup-p25 | 1 |
| VLIA | E2E-gen-p25 | 299 |
| **VLIA total** | | **1,799 (about 5.0 h)** |
| baseline | correct | 300 |
| baseline | empty | 300 |
| baseline | L-para | 300 |
| baseline | L-goal | 300 |
| baseline | L-E2E | 90 |
| baseline | L-sup-p25 | 1 |
| baseline | L-gen-p25 | 299 |
| **baseline total** | | **1,590 (about 4.4 h)** |

- **Learning curve:** 10 episodes per task per model at each milestone before the final one, about 0.3 h per model per milestone.
- **Total GPU time:** about 9.4 h for the final rows, plus about 2 h for the curves (25k / 50k / 75k).

**Schedule.** These are times, not results; the milestone follows the freeze rule only.

- **VLIA training:** 100k at about 04:00 on 21/11 if run continuously (1.85 s/step). With about 10 h off each night, about 10:00 on 22/11.
- **Baseline (rented GPU):** not started yet (19/11 09:50). It must reach the common milestone before the freeze at 22:00 on 22/11. At the lab's speed, 100k needs about 51 h, so it would have to start before about 19:00 on 20/11.
- **Evaluation (lab GPU, after training stops; the script refuses while training runs):**
  - VLIA rows: about 5 h.
  - VLIA curve at 25k / 50k / 75k: about 1 h.
  - Baseline rows: about 4.4 h, after its checkpoint is copied back and its sha256 is checked.
  - Baseline curve: about 1 h.
- **Tables:** about 12 h after the later of (VLIA training done, baseline checkpoint on the lab). If the baseline arrives by 22:00 on 22/11, the tables are ready about 23/11 morning.
- **If the schedule passes 24/11:** cut rows from the end of the plan order (L-gen-p25, E2E-gen-p25, L-sup-p25, E2E-sup-p25, …), never by results.
- **Caution:** the VLIA rows should be run at the common milestone. If the baseline can only reach 50k / 75k by the freeze, VLIA is evaluated at that milestone too, so do not start the VLIA 100k rows before the milestone is known.
