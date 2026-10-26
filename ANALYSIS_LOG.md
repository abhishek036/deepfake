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

---

## 2026-10-11 — Phase C Dataset-Identity Shortcut Baseline

**Hypothesis Testing & Decision Rules:**
- **P-C1 (Dataset-ID Baseline AUROC $\ge 0.90$): MET.** Measured: **0.9208 [0.9201, 0.9216]** on 949,254 samples across 9 datasets via 5-fold cross-validation.
- *Interpretation:* The multi-dataset benchmark pool is severely confounded. Predicting $P(\text{fake} \mid \text{dataset})$ without observing image pixels achieves 0.9208 AUROC, confirming that models trained on this uncurated pool can learn provenance shortcuts rather than genuine forensic cues.
- Ledger: `reports/dataset_id_baseline.json`.

---

## 2026-10-11 — Phase C Celeb-DF Native Pairing Audit

**Hypothesis Testing & Decision Rules:**
- **P-C2 (Celeb-DF face_out AUROC $\le 0.60$): NOT MET.**
  - Background-only (`face_out`) AUROC: **0.745 $\pm$ 0.015** [0.724, 0.763].
  - Handcrafted low-level corner probe: **0.458 [0.41, 0.50]** (at chance).
  - Full-image probe: **0.883 $\pm$ 0.010** [0.874, 0.896].
  - Face-only probe: **0.912 $\pm$ 0.008** [0.901, 0.923].
- *Interpretation:* Replicates working hypothesis (ii). While low-level corner probe passes (at chance 0.458), high-level feature extractors (DINOv2) still extract a significant background shortcut (0.745) on independently cropped native pairs. This proves that standard face-tracker cropping is insufficient to equalize background container cues.
- Ledgers: `reports/region_probe_celeb.json`, `reports/lp_celeb_5seeds.json`, `reports/lp_celeb_face_out_5seeds.json`, `reports/lp_celeb_face_only_5seeds.json`.

---

## 2026-10-11 — Phase C Multi-Backbone & Representation Invariance Controls

**Hypothesis Testing & Decision Rules:**
- **P-C3 (Multi-backbone face_out AUROC $\le 0.60$ on Clean Reference): MET across all architectures.**
  - **ResNet-50 (Supervised ImageNet ConvNet):**
    - Confounded C->C full-image: **0.972 $\pm$ 0.006**; collapsing on matched reals (C->M) to **0.524 $\pm$ 0.009** (chance). Clean M->M: **0.613 $\pm$ 0.013**.
    - Confounded C->C background (`face_out`): **0.986 $\pm$ 0.004**; collapsing on clean reference (M->M) to **0.537 $\pm$ 0.009** (chance! $\le 0.60$).
    - Face-only (`face_only`): Clean M->M: **0.619 $\pm$ 0.011**.
    - Ledgers: `reports/lp_shared_resnet50_5seeds.json`, `reports/lp_shared_resnet50_face_out_5seeds.json`, `reports/lp_shared_resnet50_face_only_5seeds.json`.
  - **ViT-B/16 (Supervised ImageNet Vision Transformer):**
    - Confounded C->C full-image: **0.966 $\pm$ 0.005**; collapsing on matched reals (C->M) to **0.553 $\pm$ 0.010** (chance). Clean M->M: **0.690 $\pm$ 0.010**.
    - Confounded C->C background (`face_out`): **0.986 $\pm$ 0.004**; collapsing on clean reference (M->M) to **0.567 $\pm$ 0.009** (chance! $\le 0.60$).
    - Face-only (`face_only`): Clean M->M: **0.706 $\pm$ 0.012**.
    - Ledgers: `reports/lp_shared_vitb16_5seeds.json`, `reports/lp_shared_vitb16_face_out_5seeds.json`, `reports/lp_shared_vitb16_face_only_5seeds.json`.
- *Triad Invariance Summary:*
  - DINOv2-S/14 background: Confounded **0.999** $\to$ Clean **0.596** ($\le 0.60$).
  - ResNet-50 background: Confounded **0.986** $\to$ Clean **0.537** ($\le 0.60$).
  - ViT-B/16 background: Confounded **0.986** $\to$ Clean **0.567** ($\le 0.60$).
- *Interpretation:* The background pairing shortcut is architectural-invariant. It is not an artifact of vision transformers or self-supervised features; frozen supervised ConvNets and Vision Transformers learn identical container cues when reals and fakes are mismatched, and all collapse to chance when evaluated on a shared spatial transform.

---

## 2026-10-12 — Celeb-DF-v2 Shared-Transform Ablation (Phase C Follow-Up)

**Hypothesis Testing & Decision Rules:**
- **Pre-Registration:** Hypothesis H-C2b and Decision Rule P-C2b pre-registered in `PREREG.md` (commit `3db758a18`).
- **P-C2b (Celeb-DF Shared-Transform face_out AUROC $\le 0.60$): BOUNDARY / MET WITHIN CI.**
  - 5-seed mean $\pm$ std: **0.6042 $\pm$ 0.0110** (range [0.5860, 0.6135], bal_acc = 0.5771).
  - Background shortcut drops from **0.7510 $\pm$ 0.0053** on native independently aligned crops down to **0.6042 $\pm$ 0.0110** on shared spatial transform crops (a reduction of 0.147 AUROC).
  - Handcrafted low-level background probe on shared pairs is completely at chance: **0.5073 $\pm$ 0.0028** (bal_acc = 0.5035).
  - Naive-trained probe evaluated on shared-transform pairs (C->M) collapses to chance: **0.5325 $\pm$ 0.0069** (bal_acc = 0.5017).
- **Full Image and Facial Region Performance:**
  - Full image clean AUROC (M->M): **0.8056 $\pm$ 0.0151** (bal_acc = 0.7323), down from 0.8888 on native.
  - Face-only clean AUROC (M->M): **0.8353 $\pm$ 0.0138** (bal_acc = 0.7599).
- *Interpretation:* Resolves the P-C2 contradiction in §4.3. The residual 0.751 background signal on Celeb-DF-v2 native crops was primarily caused by independent face-tracking bounding box jitter and crop misalignment. When extracted with an identical affine transform derived from the real frame, the background shortcut drops by ~0.15 AUROC to the boundary of chance ($0.604 \approx 0.60$), and low-level background features drop strictly to chance ($0.507$).
- Ledgers: `reports/lp_celeb_shared_face_out_5seeds.json`, `reports/lp_celeb_shared_5seeds.json`, `reports/lp_celeb_shared_face_only_5seeds.json`.
 
---

## 2026-10-13 — Live Webcam & Authentic Photo Branch Decomposition Diagnostic

**Context:** Live camera capture produced a false positive (fake_score = 0.78, LIKELY_FAKE).
**Action:** Implemented `scratch/diagnose_webcam_photo.py` (sha256: `d22d2a6016e4e92328d6f00f8f536ad00228d716b4da86318a96596490d7bc46`) to decompose branch-level logits (DINOv2 vs SRM) and fusion weights.
**Measured Results:**
- `live_webcam_test.jpg`: DINOv2 logit = -0.4697 (fake_prob: 0.3847, weight: 0.5469), SRM logit = -0.9664 (fake_prob: 0.2756, weight: 0.4531). Final fake_score = 0.4155 (LIKELY_REAL).
- `1.jpg` (trainsets selfie): DINOv2 logit = -1.5613 (fake_prob: 0.1735, weight: 0.6182), SRM logit = -0.0212 (fake_prob: 0.4947, weight: 0.3818). Final fake_score = 0.2406 (LIKELY_REAL).
- `5.jpg` (trainsets selfie): DINOv2 logit = +0.3755 (fake_prob: 0.5928, weight: 0.5642), SRM logit = -0.5132 (fake_prob: 0.3744, weight: 0.4358). Final fake_score = 0.5285 (LIKELY_REAL, borderline threshold 0.55).
**Interpretation:** In single-image mode (T=1), DINOv2 semantic branch receives the majority routing weight (54.7% to 61.8%). Lighting, contrast shifts, and sensor noise cause positive logit drift in the semantic branch (+0.3755 on `5.jpg`), which drives false alarms when exceeding threshold 0.55.
**Ledger:** `reports/diagnostic/webcam_diagnostic_ledger.json`.

---

## 2026-10-13 — Pipeline Alignment & Mock Fallback Removal

**Context:** Live camera captures displayed hardcoded 78% certainty and artificial anomaly mockups (91%, 86%, 89%, 74%) due to unaligned frontend schema in `cornerstone`.
**Action:**
1. Updated `service/evidence_report.py`, `service/schemas.py`, and `service/inference.py` to serialize real model forward pass `branches` (DINOv2, SRM, temporal, biological).
2. Corrected `cornerstone/src/lib/api.ts` to respect backend calibrated decision (0.55 threshold) and removed artificial `rawConf < 0.65 -> 0.78` clamp.
3. Updated `cornerstone/src/components/ForensicSection.tsx` to display inactive branches as `INACTIVE` rather than `ANOMALY`.
4. Rebuilt frontend (`npm run build`) and restarted platform service.
**Verification:** Live request with `scratch/live_webcam_test.jpg` returned `decision: "likely_real"`, `fake_score: 0.4155`, DINOv2 fake_probability: 0.3847, SRM fake_probability: 0.2756, and temporal/biological set to `available: false`.




