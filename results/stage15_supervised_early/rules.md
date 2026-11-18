# Stage 15 rules (primary model, agent view, out-of-fold)

| rule | what | value | 95% CI | verdict |
|---|---|---:|---|---|
| R1 | full video: accuracy p=100 CI lower bound > 90% | 1.000 | [1.000, 1.000] | PASS |
| R2 | half video: accuracy p=50 CI lower bound > 50% | 1.000 | [1.000, 1.000] | PASS |
| R3 | not only the scene: tasks 0 vs 1, 2-way accuracy p=50 CI lower bound > 50% | 1.000 | [1.000, 1.000] | PASS |
| R4 | robot: success(E2E-sup, p=25) - success(I-wrong) CI lower bound > 0 (paired_bootstrap) | 0.164 | [0.120, 0.207] | PASS |

Primary model per outer fold: fold 0 svm-lin, fold 1 svm-lin, fold 2 svm-rbf, fold 3 svm-lin, fold 4 logreg

Confused pairs at p=25 (true -> predicted: count): 0->1: 1
