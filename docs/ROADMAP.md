# VLIA Research Roadmap: Early Robot-Relevant Goal Disambiguation

## 1. Research Focus & Problem Formulation

The central research problem is **Early Robot-Relevant Goal Disambiguation from Egocentric Manipulation Video**:

> **Core Research Question:**  
> *How early can a robot-relevant goal be disambiguated from egocentric manipulation video when multiple candidate goals share the same observed sub-actions, before physical trajectories reveal the intended outcome?*

### Motivation & One-to-Many Ambiguity
In egocentric human manipulation, the observed early sub-action (WHAT) is frequently compatible with several mutually exclusive future outcomes (WHY/goals):

```text
               ┌──► G1: Move cup to shelf
pick up cup ───┼──► G2: Put cup into box
  (WHAT)       └──► G3: Move cup to tray
```

- **Sub-action (WHAT):** Current atomic interaction (e.g., `pick up cup`).
- **Goal / Intention (WHY):** Intended terminal world state (e.g., `move cup to shelf`).
- **One-to-Many Ambiguity:** $|\mathcal{G}(w)| \ge 2$, where $\mathcal{G}(w) = \{G \in \mathcal{G} \mid \text{SubAction}(G) = w\}$.

### Mathematical Formulation
Given an observed video prefix $V_{1:t_{\mathrm{obs}}} = (I_1, I_2, \dots, I_{t_{\mathrm{obs}}})$, the goal is to estimate the posterior distribution over candidate goals:
$$\hat{P}\left(G \mid V_{1:t_{\mathrm{obs}}}\right), \qquad G \in \mathcal{G}(w)$$

### Temporal Exclusion Protocol (Outcome Exclusion Rule)
To ensure the model performs genuine **anticipation** rather than recognizing explicit post-hoc outcomes, evaluation observations must satisfy:
$$t_{\mathrm{obs}} \le t_{\mathrm{divergence}} - \Delta$$
- $t_{\mathrm{divergence}}$: The physical divergence point where the hand/object trajectory departs from the common grasp path (operationally approximated by object lift-off $t_{\mathrm{lift}}$).
- $\Delta \ge 0.5\,\text{s}$: Minimum reaction margin required for collaborative robotic assistance.

---

## 2. Research Hypothesis & Mechanism

> **Core Hypothesis:**  
> When multiple goals share the same early sub-actions, explicitly conditioning goal inference on the predicted sub-action ($\text{WHAT} \to \text{WHY}$) provides a useful inductive bias for fine-grained goal disambiguation.

```text
Egocentric Video Prefix V_{1:t_obs}
               │
               ▼
   Frozen Visual Backbone (SmolVLM2)
               │
               ├────────────────────────► WHAT (Observed sub-action)
               │                                    │
               ▼                                    ▼
        Direct Path                       Guided Reasoning Path
               │                                    │
               └────────────────► WHY ◄─────────────┘
                          (Intended Goal)
                                 │
                                 ▼
                        z_int ∈ R^256
                                 │
                                 ▼
                     Downstream Robot Interface
```

> [!IMPORTANT]
> The $\text{WHAT} \to \text{WHY}$ relationship is treated as a **research hypothesis / inductive bias to be rigorously verified**, NOT as an established or proven bottleneck.

---

## 3. Separation of Concerns

1. **Core Scientific Contribution:**
   - Formalization of early goal disambiguation under shared sub-actions.
   - Hypothesis testing of the computational conditioning path ($\text{WHAT} \to \text{WHY}$).
   - Formulation of same-task hard-negative objectives.
   - Rigorous evaluation protocol excluding trajectory divergence.
2. **Downstream Validation Testbed (SmolVLA):**
   - Software interface engineering (`Intention Adapter` $256 \to 960$).
   - Downstream skill dispatching proxy / policy integration.
   - *Implementation correctness (unit test pass, gradient flow) $\neq$ scientific policy improvement.*

---

## 4. Research Questions (RQ1–RQ4)

- **RQ1 (Goal Disambiguation):** Can a model distinguish between robot-relevant goals sharing identical early sub-actions from egocentric video significantly better than random chance?
- **RQ2 (Computational Path Inductive Bias):** Does explicit conditioning on predicted WHAT improve fine-grained goal inference compared to direct prediction and independent auxiliary supervision, when controlled for model capacity?
- **RQ3 (Strict Temporal Anticipation):** How does goal disambiguation accuracy behave as observation horizons vary under strict pre-divergence constraints ($t_{\mathrm{obs}} \le t_{\mathrm{divergence}} - \Delta$)?
- **RQ4 (Robot Relevance):** Can predicted goal representations reliably drive downstream robot skill selection and interface with a VLA policy?

---

## 5. Methodological Framework & Controls

### A. Conditioning Architectures
1. **Direct WHY:** $V \to z_{\mathrm{why}}$.
2. **Direct WHY (Capacity-Matched Control):** Direct model with widened/deepened layers matching the exact parameter count and FLOPs of the two-stage model ($\approx 1.7\text{M}$ params), ruling out capacity-driven gains.
3. **Auxiliary WHAT + WHY:** Shared trunk with independent WHAT and WHY heads ($\mathcal{L}_{\mathrm{why}} + \lambda \mathcal{L}_{\mathrm{what}}$).
4. **Predicted-WHAT Guided WHY:** $V \to \hat{z}_{\mathrm{what}} \to z_{\mathrm{why}}$ via a gated residual reasoner.
5. **Diagnostic Oracle WHAT:** Privileged setting using ground-truth WHAT at inference to measure semantic headroom (upper-bound only).

### B. Same-Task Hard-Negative Loss
To penalize confusions between distinct goals within the same overarching task:
$$\mathcal{L}_{\mathrm{same\text{-}task\text{-}HN}} = -\log \frac{\exp(\text{sim}(z, z^+) / \tau)}{\exp(\text{sim}(z, z^+) / \tau) + \sum_{z^- \in \mathcal{N}_{\mathrm{same\text{-}task}}} \exp(\text{sim}(z, z^-) / \tau)}$$

### C. Frozen Residual Reranker (Pha 2)
To eliminate gradient interference from multi-task training:
1. Freeze Phase-1 base predictor completely.
2. Train a lightweight residual head with **Soft-MRR** surrogate ranking loss:
   $$\hat{z}_t^{\mathrm{int}} = \mathrm{Normalize}\left(z_{t,\mathrm{base}}^{\mathrm{why}} + \beta \Delta z_t^{\mathrm{why}}\right)$$

---

## 6. Evaluation Protocol & Metrics

Evaluation must report three distinct metric slices:
1. **Overall Performance:** Standard multi-positive MRR, Top-1, Top-3 across the full benchmark.
2. **Same-Task Performance:** Accuracy/MRR restricted to candidate sets sharing the same high-level task (the definitive test of disambiguation).
3. **Cross-Task Performance:** Coarse task discrimination accuracy.

### Current Stage-A Empirical Results (EgoIntent Pilot, 3 Seeds)

| Configuration | Why MRR | Top-1 | Top-3 | Status / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Random Baseline** | `0.0420` | `0.0087` | `0.0259` | Multi-positive chance |
| **Direct Why (Standard)** | `0.0475 ± 0.0061` | `0.0186 ± 0.0047` | `0.0372 ± 0.0081` | Direct baseline |
| **Direct Why (Capacity-Matched)** | `0.0478 ± 0.0045` | `0.0186 ± 0.0038` | `0.0387 ± 0.0062` | Capacity control |
| **Auxiliary What + Why** | `0.0471 ± 0.0030` | `0.0155 ± 0.0054` | `0.0403 ± 0.0054` | Independent head fails |
| **Predicted-What Guided** | `0.0518 ± 0.0015` | `0.0217 ± 0.0027` | `0.0434 ± 0.0027` | **+8.9%** over Direct |
| **+ What Hard-Negative** | `0.0540 ± 0.0023` | `0.0279 ± 0.0047` | `0.0419 ± 0.0047` | Penalizes same-task errors |
| **+ Frozen Residual Reranker** | **`0.0551 ± 0.0009`** | **`0.0295 ± 0.0027`** | **`0.0434 ± 0.0027`** | **+15.9%** over Direct |
| *Mean-Pool 8-Frames Baseline* | `0.0589 ± 0.0016` | `0.0233 ± 0.0046` | `0.0419 ± 0.0059` | **Strongest static baseline** |
| *Diagnostic: Oracle What* | `0.1055 ± 0.0067` | `0.0527 ± 0.0054` | `0.0915 ± 0.0194` | Privileged headroom check |

> [!NOTE]
> **Scientific Transparency:** The proposed method ($0.0551$) does **not** beat the strongest static baseline (`Mean-pool 8 frames` = $0.0589$) on overall MRR. However, it significantly outperforms it on **Same-Task Top-1 accuracy** ($0.0295$ vs $0.0233$), where disambiguation matters most.

### Dominant Failure Mode
- **81.47% (510 / 626)** of Top-1 errors are within the same task.
- Only **18.53% (116 / 626)** are cross-task errors.
- Confirms that coarse task recognition is largely solved, while fine-grained within-task disambiguation is the true bottleneck.

---

## 7. Out-of-Distribution (OOD) Validation Dataset

### Pilot Collection
- **Scale:** 48 videos (2 participants $\times$ 12 tasks $\times$ 2 repetitions).
- **Setup:** Chest/neck-mounted egocentric camera, unconstrained manipulation, no verbalized intention.
- **Role:** Pure zero-shot OOD stress test / qualitative evaluation. Not used for ML training splits.

### 12 Task Taxonomy
- `T01`: Cup $\to$ Shelf
- `T02`: Bottle $\to$ Box
- `T03`: Block $\to$ Tray
- `T04`: Sort Blocks by Color
- `T05`: Sort Objects by Category
- `T06`: Object $\to$ Target Position
- `T07`: Peg $\to$ Hole
- `T08`: Object $\to$ Container
- `T09`: Simple Component Assembly
- `T10`: Component $\to$ Base Attachment
- `T11`: Open Container $\to$ Retrieve Object
- `T12`: Retrieve Object $\to$ Put Back

### Annotation Schema
```json
{
  "video_id": "P01_T01_R01",
  "participant_id": "P01",
  "task_id": "T01",
  "repetition_id": 1,
  "video_duration_s": 6.4,
  "observation_end_s": 1.6,
  "outcome_start_s": 3.8,
  "object": "cup",
  "target": "shelf",
  "what": "pick up cup",
  "why": "move cup to shelf",
  "next": "place cup on shelf",
  "intention_id": "INT_01"
}
```

---

## 8. Downstream Robot Interface (SmolVLA & Skill Proxy)

### Technical Integration Status
- `Intention Adapter`: Linear projection $256 \to 960$.
- `Prefix Injection`: Multimodal prefix $[I; L; E_{\mathrm{int}}; S]$, expanding prefix length by $+1$.
- `Verified Checks`: Forward/backward, finite loss, adapter gradient norm $> 0$, KV-cache consistency, baseline fallback.
- *Status:* **Implementation correctness PASS.** (Distinguished from policy-level scientific claims).

### LIBERO-10 Benchmark Role
- Fixed 7-D matched baseline: **24% success rate** over 100 episodes (`checkpoint_025000`).
- Role: Controlled engineering baseline, not a SOTA claim.
- Because LIBERO task language already explicitly specifies goals, the primary robot relevance validation is framed as **Skill Dispatching Proxy** and ambiguity-focused evaluation rather than full policy retraining.

---

## 9. Immediate Action Items

1. **Parameter-Matched Experiment:** Execute capacity-matched Direct WHY baseline over 3 seeds to formally establish capacity invariance.
2. **Disambiguation Benchmark Slice:** Produce the formal Overall vs. Same-Task vs. Cross-Task evaluation table.
3. **Temporal Exclusion Sweep:** Evaluate accuracy curves at pre-divergence horizons ($t_{\mathrm{obs}} \le t_{\mathrm{divergence}} - \Delta$).
4. **OOD Stress Test:** Run inference on the 48 pilot videos and document qualitative failure cases.
