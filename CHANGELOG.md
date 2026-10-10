# FORGE-Lite Changelog

All model versions, architectural changes, and data integration milestones are
documented here. Every model version that reaches evaluation must be tagged in
this file before any retraining begins.

---

## v3.0 — Frozen Baseline Checkpoint
**Status:** FROZEN — do not retrain from this checkpoint  
**Tagged:** 2026-10-02  
**Policy file:** `configs/frozen_v3_policy.json`

### Architecture
- **7 branches:** DINOv2 ViT-S/14 (frozen) + LayerNorm/LoRA adapt, SRM+Haar forensic,
  temporal 3D-Conv, rPPG, AV-sync (SyncNet-style), geometry landmarks, DINOv2 context
- **Fusion:** 512-dim MLP with quality-aware branch weighting
- **Outputs:** `fake_score`, `uncertainty`, `branch_weights`, `abstention` flag,
  `insufficient_evidence` class

### Training Data (v3.0)
| Dataset | Videos | Status |
|---|---|---|
| FaceForensics++ C23 | 1,000 | ✅ Extracted |
| (others per combined.jsonl) | — | ✅ Manifests built |

### Known Limitations (v3.0)
- Trained exclusively on 2019–2021 GAN-era forgeries
- No diffusion-based generation in training data
- rPPG branch: 8fps sampling violates Nyquist (fixed in v3.1)
- SRM branch: structurally blind to fully-synthetic diffusion video
- AV branch: detects desync only, not voice-clone (fixed in v3.1)
- No fairness evaluation (no subgroup labels in training data)
- Frame-level bootstrap CI in celeb_df — must recompute at video level

### Calibration
- Temperature calibration performed on calibration partition
- Thresholds frozen — no recalibration without version bump

---

## v3.1 — FakeAVCeleb + DF40 Integration *(planned)*
**Status:** PENDING — data integration in progress  
**Target:** 2026-Q4

### Planned Changes vs v3.0
- **Data:** FakeAVCeleb v1.2 (21,544 videos) integrated — train + eval
- **Data:** DF40 images downloaded + integrated — eval cross-domain
- **Fix:** rPPG branch sampling raised to 25fps (Nyquist fix)
- **Fix:** AV branch trained on FakeAVCeleb Cat. B (voice-clone gap)
- **Fix:** `forgery_type` field added to all manifests
- **New metrics:** `tpr_at_0_1pct_fpr`, `per_skin_tone_eer`, `per_forgery_type_auroc`
- **New eval slots:** FakeAVCeleb Cat B/C/D, DF40 per-forgery-type breakdown

### Recalibration Required
- Full temperature calibration on new calibration partition after retraining

---

## v3.2 — AV-Deepfake1M AV Branch Retraining *(planned)*
**Status:** PENDING — dataset not yet downloaded  
**Target:** 2027-Q1

### Planned Changes vs v3.1
- AV branch retrained with LLM-driven word-level partial manipulation
- 1.1M videos, 2,068 subjects, localized phrase-level forgery
- New eval: partial manipulation localization (segment-level AUC)

---

## v3.3 — DINOv3 Backbone A/B Test *(planned)*
**Status:** PENDING — conditional on DINOv3 winning A/B test  
**Target:** 2027-Q1

### A/B Protocol
```
Model A: DINOv2 ViT-S/14 + LayerNorm adapt  (current)
Model B: DINOv3 ViT-S/16 + LayerNorm adapt  (same training protocol)
Metric:  DF40 cross-domain AUROC on fully_synthetic forgery_type
Decision: keep whichever wins on fully_synthetic + composited combined
```
**License caution:** DINOv3 has custom Meta AI Research license — review before distribution.

---

## v4.0 — Branch Pruning Ablation *(planned)*
**Status:** PENDING — run after DF40 integrated  
**Target:** 2027-Q1–Q2

### Decision Rule
Any branch contributing < 1% cross-domain AUROC delta at cost > 100MB VRAM
is a candidate for removal (per MASTER_PLAN §8).

| Ablation | Branches | Expected Delta |
|---|---|---|
| No rPPG | 6 | TBD |
| No geometry | 6 | TBD |
| No AV | 6 | TBD |
| No forensic (SRM) | 6 | Near-zero on fully_synthetic |
| DINOv2 only | 1 | Baseline |
| DINOv2 + temporal | 2 | Core minimal |
| DINOv2 + forensic + temporal | 3 | Classic comparison |

---

## Recalibration Cadence
- Every **6 months** regardless of new data
- Whenever a **new generator family** exceeds 10k publicly circulated examples
- All prior versions archived with frozen test sets after each recalibration
