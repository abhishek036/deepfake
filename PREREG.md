# PREREG.md — Pre-Registration: Shared-Transform Pairing Experiment
**Date registered:** 2026-10-10
**Registered by:** Abhi + AI assistant (Antigravity / Claude)
**Rule:** Any analysis performed after seeing Phase B results goes into ANALYSIS_LOG.md with date and reason.

---

## Hypotheses

**H1 (primary):** Shared-transform pairing (same video source, same crop pipeline applied to both real frame and its deepfake) will reduce the DINOv2-S/14 linear probe AUROC substantially relative to the naive pairing (DF40 PNG fakes vs FF++ JPEG reals).

**H2:** On shared-transform pairs, the face-out (background-only) probe AUROC will be at or near chance (≤ 0.60), indicating that the background region does not carry a reliable shortcut signal once the container format is equalized.

**H3:** On shared-transform pairs, the face-only probe AUROC will remain higher than the full-image probe, indicating residual synthesis artifact signal in the face region even after pairing cleanup.

**H4 (exploratory):** If H1 fails (face_out probe remains ~0.74 on shared-transform pairs), the residual signal is likely from compression generation or face-region spill, not from geometric misalignment.

---

## Pre-Specified Decision Rules

| Symbol | Condition | Threshold | Interpretation if met |
|--------|-----------|-----------|----------------------|
| P-S1 | Paired MAD of face crop centers (pixels) | ≤ 4 px | Face alignment is equalized; geometry is not the confound |
| P-S2 | face_out linear probe AUROC (background only) | ≤ 0.60 | Background shortcut eliminated |
| P-S3 | face_only linear probe AUROC | ≥ 0.85 | Face region retains synthesis signal |
| P-S4 | Full-image probe AUROC on shared-transform pairs | ≤ 0.70 | Pairing substantially reduces confound |

**Joint decision rule for "clean reference" claim:**
Shared-transform pairing is declared a clean reference **only if P-S1 AND P-S2 are both met**.
If either fails, the pairing is confounded and the next diagnostic step applies (see H4).

---

## Tolerances and Sample Size

- Pairs: 600 (from `extract_shared_transform_pairs.py`, 4 frames per pair)
- Frames per pair: 4 (randomly sampled from shared video segment)
- Split: official FF++ split (720 train / 140 val / 140 test), scene-disjoint
- AUROC CI: bootstrap over video/scene groups, 5 seeds, report mean ± SD and 95% CI
- Equivalence margin for "at chance": AUROC ∈ [0.45, 0.55] (TOST with α=0.05)

---

## What will NOT be decided post-hoc

- Thresholds P-S1 through P-S4 are fixed here; not adjusted after seeing results.
- The choice of which probe mask (full / face_out / face_only) is primary is fixed: **face_out** is primary (tests P-S2), face_only is secondary (tests P-S3).
- Seed list for 5-seed runs: 42, 7, 13, 99, 2024.

---

## Phase C Pre-Registration: Cross-Domain & Dataset Shortcuts

### Hypotheses
**H-C1 (Dataset-ID Shortcut):** In uncurated multi-dataset benchmarks (e.g. `combined.jsonl`), categorical dataset provenance alone ($P(\text{fake} \mid \text{dataset})$) predicts forgery status with AUROC $\ge 0.90$ without accessing any image pixels.

**H-C2 (Celeb-DF Native Pairing):** In Celeb-DF-v2 native pairing (target video matched), independently aligned crops may still exhibit residual background or format separation. If `face_out` AUROC $\le 0.60$, the native pairing is clean; if $> 0.60$, it suffers from cropping/alignment shortcut leakage.

### Decision Rules
| Symbol | Condition | Threshold | Interpretation if met |
|--------|-----------|-----------|----------------------|
| P-C1 | Dataset-ID baseline 5-fold CV AUROC on combined training pool | $\ge 0.90$ | Dataset provenance constitutes an evaluation shortcut |
| P-C2 | Celeb-DF `face_out` DINOv2 linear probe AUROC | $\le 0.60$ | Celeb-DF native background is free of shortcut cues |

---

## Analysis Log (fill in after results)

*See ANALYSIS_LOG.md. Any post-hoc deviation from the above must be documented there with date and reason.*
