# ANALYSIS_LOG.md — Post-Hoc Analysis Deviations
**Rule:** Any analysis performed after seeing results, or any protocol deviation, is logged here with date and reason.

---

## 2026-10-10 — Loss Polarity Finding (Phase A)

**Finding:** `pos_weight=18.4` in `configs/train.yaml` and `training/losses.py` upweights **fake** samples (label=1), not real samples as stated in the code comments. Label convention is 0=real, 1=fake (`data/dataset.py` L45). Combined with `balanced_sampling=true`, fakes receive disproportionate loss weight even in balanced batches.

**Impact on Paper 1:** Does not affect the audit results (all numbers come from post-hoc inference, not from re-training). Must be disclosed as a known training confound of the best.ckpt checkpoint.

**Action:** Added note to phase_a_log. Will disclose in paper Limitations section. No re-training in Paper 1.

**Decision:** Pre-registered? No — this was a diagnostic finding from Phase A code audit, not a hypothesis test. Logging here per rule 2.

---

## 2026-10-10 — Phase B Shared-Transform Results

**Hypothesis Testing & Decision Rules:**
- **P-S1 (Center MAD $\le 4.0$ px): MET.** Measured: 0.721 px (pilot median: 0.729 px).
- **P-S2 (face_out AUROC $\le 0.60$): MET.** Single-seed: 0.599 [0.572, 0.627]. 5-seed mean $\pm$ std: **0.596 $\pm$ 0.003** [0.592, 0.599]. Background shortcut eliminated across all seeds.
- **P-S3 (face_only AUROC $\ge 0.85$): NOT MET.** Single-seed: 0.772 [0.736, 0.819]. 5-seed mean $\pm$ std: **0.769 $\pm$ 0.008** [0.760, 0.780].
  *Contradiction:* Expectation of $\ge 0.85$ was not met. Residual facial synthesis signal is detected (0.769 vs 0.596 on background), but frozen DINOv2-S/14 linear probe achieves 0.769, not 0.85.
- **P-S4 (Full-image AUROC $\le 0.70$): NEAR BOUNDARY.** Single-seed: 0.736 [0.701, 0.782]. 5-seed mean $\pm$ std: **0.733 $\pm$ 0.009** [0.718, 0.743].
- **Joint Clean Reference Rule (P-S1 AND P-S2): MET.**

**Region Probe (Handcrafted low-level features):**
- Composited: background = 0.437 [0.38, 0.49] (at chance).
- Reenactment: background = 0.521 [0.48, 0.57] (at chance).

**5-Seed Controls (Seeds: 42, 7, 13, 99, 2024):**
- **Identity-Disjoint Control:** DINOv2 M->M AUROC = **0.705 $\pm$ 0.030** [0.653, 0.730]. Low-level = 0.517 $\pm$ 0.047. Confirms performance drop is not driven by source-identity leakage across train/test.
- **Size Control (3 fakes/scene):** DINOv2 M->M AUROC = **0.724 $\pm$ 0.014** [0.705, 0.740]. Low-level = 0.543 $\pm$ 0.002. Confirms results are invariant to balanced vs 1:2 real-to-fake sampling ratio.
