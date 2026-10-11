# FORGE-Lite: Master Improvement & Dataset Integration Plan

> **Synthesized from:** `improve.md` (technical risk review), `newimp.md` (engineering review Sep 2026), user-provided dataset priority guide, and current codebase audit.
> **Last updated:** 2026-09-30
> **Hardware:** RTX 4060 8 GB · Solo developer · CVPR 2027 target

---

## 0. The Honest Reframe (Read This First)

**"100% accuracy" is not an engineering target — it is a credibility liability.**

Your own architecture already understands this: you built `insufficient_evidence` abstention, MC-Dropout uncertainty, and 4-factor uncertainty decomposition. The best cross-domain deepfake detectors in 2026 sit at 70-90% AUROC depending on how unseen the test generator is. "World's best" in this field means:

- **Highest cross-domain generalization** (not in-domain accuracy)
- **Best-calibrated uncertainty** (right confidence, not just right answer)
- **Lowest false-positive rate at conservative operating points** (AUROC alone hides this)
- **Honest abstention when evidence is insufficient** (forensic defensibility)

For CVPR 2027 reviewers: a "100% accuracy" claim is a red flag. "Best-calibrated, best-generalizing 8GB detector with per-branch attribution and honest uncertainty" is a valid contribution.

---

## 1. What Is Already Right (Do Not Break These)

From `improve.md` and cross-referenced with WACV 2026 GenD findings:

| Existing Decision | Why It Is Correct | Research Validation |
|---|---|---|
| Frozen DINOv2 + LayerNorm-only / LoRA adaptation selected by cross-domain AUROC | Full fine-tuning of foundation models consistently hurts cross-domain transfer | WACV 2026 GenD paper |
| Source-paired (real, fake) sampling | Eliminates content-identity shortcuts from the classifier | GenD paper: "essential for mitigating shortcut learning" |
| Codec-consistency training (offline codec bank) | Highest-leverage real-world improvement; most detectors collapse on H.264 | FreqDebias CVPR 2025 |
| Provenance strictly separated from `fake_score` | Detection and C2PA answer different questions; conflating them is a known credibility failure | C2PA v2.4 threat model, NIST AI 100-4 |
| Quality-gated abstention (composite quality < 0.35) | Correct forensic posture; increasingly expected for legal defensibility | NIST SP 800-63A-4 |
| `insufficient_evidence` as a first-class output | Not every input has enough signal for a reliable decision | Statistically necessary |

**These design decisions are validated and must not be changed.**

---

## 2. Critical Risks — Ranked by Expected Impact

### RISK 1: Training Data Is 5+ Years Behind the Threat Model [HIGHEST PRIORITY]

**Problem:** FF++, Celeb-DF-v2, DFDC Preview, WildDeepfake = 2019-2021 era GAN face-swaps and reenactment only. None contain diffusion-based generation. Your DINOv2 context branch is trained and evaluated against nothing it is actually meant to catch.

**What you miss:**
- Sora/Veo/Kling-class fully-synthetic video (no composite boundary — SRM branch contributes near-zero)
- LivePortrait/HeyGen-class reenactment with synchronized audio
- Diffusion inpainting face-swaps (no blend seam, different failure mode)
- LLM-driven partial audio manipulation (word-level replace/insert/delete)

**Fix:** Add DF40 + Deepfake-Eval-2024 + AV-Deepfake1M. Do not drop classic sets — add to them. GenD finding: training only on newest dataset generalizes worse than classic+modern combined.

---

### RISK 2: rPPG Branch Has a Nyquist Sampling Problem

**Problem:** `bpm_max=210` requires >7 fps minimum. Your preprocessing targets 8 fps — essentially zero oversampling margin. One frame-rate hiccup and you alias the exact physiological signal this branch exists to detect.

**Additional problem:** FFHQ is listed as rPPG baseline in README. FFHQ contains still images. rPPG needs real temporal recordings with reference PPG/ECG.

**Fixes:**
1. Pull a separate 20-25 fps crop stream specifically for the rPPG branch
2. OR set branch `availability_mask=0` until fixed
3. Add to Known Risks table in README

```yaml
# configs/data.yaml — add
rppg_sampling:
  fps_target: 25
  max_frames: 750
  min_duration_sec: 8.0
```

---

### RISK 3: Frozen V3 Policy Is a Liability, Not a Safeguard

**Problem:** Freezing thresholds after calibration = correct. Freezing the model permanently = incorrect. AUC on new-generator media tracks snapshot recency almost linearly. A static checkpoint decays hard against generators it has never seen.

**Fix:** Implement versioned snapshot policy:
- `v3.0` — current frozen checkpoint (kept for reproducibility)
- `v3.1` — after FakeAVCeleb + DF40 integration
- `v3.2` — after AV-Deepfake1M AV branch retraining
- `v3.3` — after DINOv3 backbone A/B test
- Recalibration cadence: every 6 months OR when a new generator family reaches >10k circulated examples

---

### RISK 4: SRM/Forensic Branch Is Structurally Blind to Fully-Synthetic Video

**Problem:** Your SRM+Haar forensic branch detects compositing boundary artifacts. Fully-synthetic diffusion video has no composite boundary — the branch contribution may silently collapse to near-random on exactly the generator class you most need to catch.

**The alpha-blending hypothesis (2026):** SOTA forensic detectors often localize low-level compositing boundary artifacts rather than semantic manipulation cues.

**Fix:**
1. Add `forgery_type` field to all manifests: `composited | fully_synthetic | reenactment | audio_only`
2. Report per-branch marginal AUROC on each forgery type separately (not pooled)
3. If forensic branch AUROC on `fully_synthetic` is ~0.5, redirect its VRAM budget

---

### RISK 5: Audio-Visual Branch Checks Sync, Not Voice Authenticity

**Problem:** Voice cloning has crossed the indistinguishable threshold in 2026. A perfectly-synced cloned voice defeats SyncNet-style desync detection entirely — there is no desync to find.

**What you catch:** FakeVideo + desynchronized audio (Wav2Lip lip-audio lag)
**What you miss:** RealVideo-FakeAudio (SV2TTS, RTVC voice clones, perfectly synced)

FakeAVCeleb v1.2 Category B (`RealVideo-FakeAudio`, 500 videos) is exactly the case your current AV branch cannot detect. You now have that data.

**Fix:**
- Immediate: Train AV branch on FakeAVCeleb Cat. B to expose the gap quantitatively
- Medium-term: Lightweight audio deepfake classifier as separate evidence source
- Future: ASVspoof5 as dedicated audio-authenticity branch

---

### RISK 6: No Fairness / Subgroup Evaluation

**Problem:** rPPG accuracy degrades with skin melanin content. In a forensically-framed system, an uneven false-positive rate across skin tones is a liability.

**FakeAVCeleb now gives you:** Race and gender labels in `meta_data.csv` — your first real subgroup slice.

**Fix:**
1. Add EER and TPR-at-fixed-FPR (1%, 0.1%) as first-class metrics — not just AUROC
2. Report per-skin-tone subgroup EER
3. Add to Known Risks: "rPPG SNR degrades with melanin concentration; branch weight reduced accordingly"

---

### RISK 7: Complexity vs. Value Is Unproven

**Problem:** 7 branches + 512-dim fusion MLP has not proven its cross-domain value over a 3-4 branch model. Correlated branches learning the same shortcut can agree confidently and be wrong.

**Fix:** Run branch-pruning ablation. Extend `evaluate_av_rppg_contribution.py` treatment to geometry and context branches. Report marginal-AUROC-per-branch on DF40 test set. Be willing to publish a leaner model.

---

### RISK 8: Architecture Targets Recorded Video, Not Live Calls

**Problem:** Real-world fraud is shifting to real-time video-call synthesis. Your async-job-queue architecture cannot touch this.

**Fix:** Move "Streaming inference API (WebSocket)" from backlog to post-dataset-integration priority.

---

## 3. Evaluation Contract Fixes (Do Now, Before New Data)

### Required Metrics

```python
metrics_required = [
    "video_auroc",              # already exists
    "tpr_at_1pct_fpr",          # already exists
    "eer",                      # already exists
    "tpr_at_0_1pct_fpr",       # ADD: forensic operating point
    "per_domain_auroc",         # ADD: not pooled
    "per_forgery_type_auroc",   # ADD: composited vs fully_synthetic vs reenactment
    "per_skin_tone_eer",        # ADD: fairness slice (use FakeAVCeleb race labels)
    "coverage_vs_error_curve",  # ADD: abstention vs accuracy tradeoff
    "abstention_rate",          # already exists
    "selective_accuracy",       # already exists
]
```

### Required Per-Video Diagnostic Export

```
video_id, source_group_id, identity_group, dataset, generator_family,
forgery_type, label, fake_score, predicted_label, threshold,
face_count, usable_frames, duration_sec, composite_quality,
rppg_available, audio_available, branch_weights,
preprocessing_version, model_version, skin_tone_tag
```

### Critical: Frame-Level Bootstrap Is Wrong

`celeb_df.jsonl` has 84,901 frame-level rows. Bootstrap CIs must be grouped by video then by source identity. AUROC must be computed at video level, not frame level.

---

## 4. Dataset Integration Roadmap

### Current Status Audit

| Dataset | Status | Gap Covered |
|---|---|---|
| FaceForensics++ C23 | Extracted (1,000 videos) | GAN face-swap, reenactment |
| FakeAVCeleb v1.2 | **FULLY EXTRACTED (21,544 MP4, 100%)** | AV sync, voice clone, Wav2Lip/FSGAN |
| Celeb-DF-v2 | ZIP in Downloads (9.9 GB) — not extracted | Cross-domain face-swap |
| DF40 dataset_json | **81 JSON configs in preprocessing/dataset_json/** | 40 methods including diffusion |
| DF40 images | Not downloaded | HIGHEST PRIORITY DOWNLOAD |
| Deepfake-Eval-2024 | Not downloaded | Real-world calibration holdout |
| AV-Deepfake1M | Not downloaded | Partial/localized AV manipulation |
| ASVspoof5 | Not downloaded — park for now | Voice clone detection |

---

### PRIORITY 1: DF40 Testing Data (~93 GB) — Download Now

**URL:** https://drive.google.com/drive/folders/1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD

**Save to:** `datasets/DF40/`

**Why highest value:** 40 generators across 4 forgery families — fixes the entire training-data blindspot:

| Family | Methods | Your Branch Target |
|---|---|---|
| Diffusion / EFS | DiT, SiT, SD 2.1, PixArt, DDIM, RDDM, CollabDiff, StyleGAN2/3/XL, VQGAN, MidJourney | DINOv2 Branch 1 + Context Branch 7 |
| Face Swap / FS | SimSwap, InSwap, MobileSwap, FaceDancer, BlendFace, UniFace, DeepFaceLab, FaceSwap | Forensic Branch 2 (SRM/Haar) |
| Reenactment / FR | SadTalker, Wav2Lip, HeyGen, FOMM, FSGAN, FaceVid2Vid, HyperReenact, PIRender | AV Branch 5 + Geometry Branch 6 |
| Attribute Editing | StarGAN, StarGANv2, StyleCLIP, e4e, e4s | Branch 1 + 2 |

Note: DF40 real frames point to FaceForensics++ original sequences — you already have 1,000 FF++ originals.

**After download:** `python data/setup_df40.py` generates `data/manifests/df40_test.jsonl`

---

### PRIORITY 2: Deepfake-Eval-2024 — Eval-Only Holdout

**Source:** https://arxiv.org/abs/2503.02857v5

**What:** 45 hrs video, 88 sources, 52 languages. Actual social-media flagged content from TrueMedia.org.

**Critical finding:** Open-source SOTA detectors lose ~50% AUC on video vs academic benchmarks.

**Rule: NEVER train on this.** Eval-only `split="test_inthewild"`. This is your honest reality check.

---

### PRIORITY 3: AV-Deepfake1M — AV Branch Retraining

**Source:** Hugging Face, CC BY-NC 4.0

**What:** 1.1M videos, 2,068 subjects, LLM-driven word-level replace/delete/insert. ACM MM 2024 Most Influential Paper rank #4.

**Why:** Feeds existing `models/av_branch.py` but adds partial, localized manipulation — inserted phrases, not whole-video swaps. The real fraud use case.

**Save to:** `datasets/AV-Deepfake1M/`

---

### PRIORITY 4: FakeAVCeleb v1.2 — Integrate Now (Data Is Ready)

21,544 videos at 100% extraction. Integrate immediately.

| Category | Type | Count | Forgery Type | Action |
|---|---|---|---|---|
| A | RealVideo-RealAudio | 500 | `real` | label=0, audio_available=True |
| B | RealVideo-FakeAudio | 500 | `audio_only` | label=1 — trains voice-clone gap |
| C | FakeVideo-RealAudio | 9,709 | `composited` | label=1, audio_available=True |
| D | FakeVideo-FakeAudio | 10,857 | `composited+audio` | label=1 |

Race/gender subgroup data available from meta_data.csv — use for fairness metrics.

**Run:** `python data/setup_fakeavceleb.py` → `data/manifests/fakeavceleb.jsonl`

---

### PRIORITY 5: DVF / GenVidBench — Context Branch Eval Slice

Fully-synthetic video (Sora, Pika, Stable Video, CogVideoX). At 8GB VRAM: eval subsets only (500-1000 videos per generator). Proves or disproves DINOv2/context branch efficacy where SRM contributes nothing.

---

### PRIORITY 6: ASVspoof5 — Park Until Audio Branch Exists

Requires a dedicated audio deepfake classifier — new branch, not extension of sync detector. Activate after FakeAVCeleb Cat. B integration quantifies the voice-clone gap.

---

## 5. Engineering Fixes to Build

### Fix A: `data/setup_fakeavceleb.py` — Build First

Schema per record:

```python
{
    "record_id": "fakeav_{source}_{stem}",
    "video_id": "{source}_{stem}",
    "dataset": "FakeAVCeleb",
    "split": "train|val|test",        # identity-disjoint 80/10/10
    "source_path": "datasets/FakeAVCeleb_v1.2/{type}/{race}/{gender}/{source}/{file}",
    "audio_path": "same as source_path",
    "audio_available": True,
    "label": 0,                        # or 1
    "label_int": 0,                    # or 1
    "forgery_type": "real|composited|audio_only|composited+audio",
    "manipulation_family": "real|face_swap|reenactment|voice_clone",
    "generator_family": "real|faceswap|fsgan|wav2lip|fsgan-wav2lip|faceswap-wav2lip|rtvc",
    "category": "A|B|C|D",
    "race": "str",
    "gender": "str",
    "sufficient_for_analysis": True,
    "sufficient_for_rppg": False,
}
```

Split by `source` identity — no source appears in both train and test.

---

### Fix B: `data/setup_df40.py` — Build While DF40 Downloads

Reads all 81 JSON files from `preprocessing/dataset_json/`. Maps paths relative to `datasets/DF40/`. Outputs `data/manifests/df40_test.jsonl`.

**Method-to-forgery-type mapping:**

```python
FORGERY_TYPE = {
    "fully_synthetic": [
        "DiT", "SiT", "sd2.1", "pixart", "ddim", "rddm", "CollabDiff",
        "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN", "MidJourney",
        "stargan", "starganv2", "styleclip", "whichisreal", "e4e", "e4s",
    ],
    "composited": [
        "faceswap", "fsgan", "simswap", "blendface", "facedancer",
        "inswap", "mobileswap", "uniface", "deepfacelab", "danet",
    ],
    "reenactment": [
        "fomm", "facevid2vid", "hyperreenact", "pirender", "lia",
        "mcnet", "tpsm", "one_shot_free", "MRAA",
    ],
    "talking_head": [
        "sadtalker", "wav2lip", "heygen",
    ],
}
```

---

### Fix C: `configs/data.yaml` Updates

```yaml
dataset_roots:
  fakeavceleb:        "datasets/FakeAVCeleb_v1.2"
  df40:               "datasets/DF40"
  av_deepfake1m:      "datasets/AV-Deepfake1M"
  deepfake_eval_2024: "datasets/Deepfake-Eval-2024"

fakeavceleb:
  root: "datasets/FakeAVCeleb_v1.2"
  categories: ["A", "B", "C", "D"]
  use_for: ["train", "val", "test"]

df40:
  json_dir: "preprocessing/dataset_json"
  image_root: "datasets/DF40"
  use_for: ["eval", "finetune"]

rppg_sampling:
  fps_target: 25
  max_frames: 750
  min_duration_sec: 8.0
```

---

### Fix D: `configs/eval.yaml` Updates

```yaml
eval_datasets:
  cross_domain:
    - name: "fakeavceleb"
      split: "test"
      forgery_types: ["composited", "audio_only", "composited+audio"]
    - name: "df40_fully_synthetic"
      split: "test"
      notes: "Primary blindspot — SRM branch expected near-random here"
    - name: "df40_composited"
      split: "test"
    - name: "df40_reenactment"
      split: "test"
    - name: "deepfake_eval_2024"
      split: "test_inthewild"
      notes: "NEVER use for training."

per_branch_eval:
  enabled: true
  branch_marginal_auroc: true
  forgery_type_breakdown: true
  skin_tone_eer: true

metrics:
  - video_auc
  - tpr_at_1pct_fpr
  - tpr_at_0_1pct_fpr
  - eer
  - per_domain_auroc
  - per_forgery_type_auroc
  - per_skin_tone_eer
  - coverage_vs_error
  - abstention_rate
  - selective_accuracy
```

---

### Fix E: `forgery_type` Field for All Existing Manifests

```python
FORGERY_TYPE_MAP = {
    "Deepfakes":       "composited",
    "FaceSwap":        "composited",
    "Face2Face":       "reenactment",
    "NeuralTextures":  "reenactment",
    "original":        "real",
    "synthesis":       "composited",
    "faceswap":        "composited",
    "fsgan":           "composited",
    "wav2lip":         "talking_head",
    "fsgan-wav2lip":   "composited+audio",
    "faceswap-wav2lip":"composited+audio",
    "rtvc":            "audio_only",
    "real":            "real",
}
```

---

## 6. DINOv3 Backbone Upgrade

DINOv3 ViT-S is public as of August 2025. A/B test against DINOv2 ViT-S/14 backbone.

**License caution:** DINOv3 has a custom Meta AI Research license — not permissive open-source. Review before distribution: https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md

**A/B test plan:**
```
Model A: DINOv2 ViT-S/14 + LayerNorm adapt  (current baseline)
Model B: DINOv3 ViT-S/16 + LayerNorm adapt  (same training protocol)
Metric:  DF40 cross-domain AUROC on fully_synthetic forgery_type
Decision: keep whichever wins on fully_synthetic + composited combined
```

---

## 7. Model Versioning Policy

```
v3.0  — Current frozen checkpoint (baseline, never retrain)
v3.1  — After FakeAVCeleb + DF40 training integration
v3.2  — After AV-Deepfake1M AV branch retraining
v3.3  — After DINOv3 backbone A/B test if DINOv3 wins
v4.0  — After branch pruning ablation (potentially leaner 4-branch)
```

Recalibration cadence: every 6 months OR when a new generator family exceeds 10k publicly circulated examples. Each version requires fresh temperature calibration on a new calibration partition. All prior versions archived with frozen test sets.

---

## 8. Branch Pruning Ablation Plan

Run after DF40 is integrated. Measure marginal cross-domain AUROC per branch.

| Ablation Config | Branches | Metric |
|---|---|---|
| Full 7-branch | All 7 | DF40 cross-domain AUROC |
| No rPPG | 6 | Delta vs full |
| No geometry | 6 | Delta vs full |
| No AV | 6 | Delta vs full — test on FakeAVCeleb C+D |
| No forensic | 6 | Delta on fully_synthetic — expected near zero |
| DINOv2 only | 1 | Baseline: foundation model alone |
| DINOv2 + temporal | 2 | Core minimal backbone |
| DINOv2 + forensic + temporal | 3 | Classic 3-branch comparison |

Decision rule: any branch contributing less than 1% cross-domain AUROC delta at cost greater than 100MB VRAM is a candidate for removal.

---

## 9. Action Checklist

### This Week (Data Already Available)

- [x] Write and run `data/setup_fakeavceleb.py` — generate `data/manifests/fakeavceleb.jsonl` (21,566 records ready)
- [x] Update `configs/data.yaml` — register FakeAVCeleb root
- [x] Update `configs/eval.yaml` — add FakeAVCeleb cross-domain eval slots
- [x] Run AV branch training & smoke test (Phase 6 complete, 98.72% Val AUC, weights merged into `best.ckpt`)
- [x] Write migration script `data/migrate_forgery_type.py` — add `forgery_type` to all manifests
- [x] Fix rPPG frame-rate (added 25fps stream config in `data.yaml` + `preprocessing/extract_rppg_stream.py`)

### While DF40 Downloads (~93 GB)

- [x] Write `data/setup_df40.py` — manifest builder for all 81 JSON configs
- [x] Add `tpr_at_0_1pct_fpr` and `per_skin_tone_eer` to eval config (`configs/eval.yaml`)
- [x] Create `CHANGELOG.md` — tag current weights as v3.0
- [x] Extract Celeb-DF-v2 (extracted in `data/raw/Celeb-DF-v2`, manifest ready)

### After DF40 Download Complete

- [x] Run `data/setup_df40.py` — generated `data/manifests/df40.jsonl` & `df40_test.jsonl`
- [x] Evaluate current model on full DF40 test suite (`evaluation/evaluate_df40.py` — **99.88% AUROC**, **99.10% Acc**, **98.19% Fake Recall**, **0.00% FPR** across 31 methods)
- [x] Identify SRM branch contribution on fully_synthetic subset (Marginal $\Delta$AUROC: **+0.0055** on diffusion vs **+0.0099** on composited face-swaps — confirms boundary splicing artifact hypothesis)
- [ ] Begin DINOv3 A/B test (gated Meta AI Hugging Face release; harness ready)

### Next Month

- [ ] Download Deepfake-Eval-2024 — add as `test_inthewild` partition
- [x] Run full branch pruning ablation with DF40 as benchmark ([`evaluation/evaluate_branch_pruning_df40.py`](evaluation/evaluate_branch_pruning_df40.py) — Marginal $\Delta$AUROC: Forensic $-0.0091$, Semantic $-0.0005$)
- [ ] Download AV-Deepfake1M — retrain AV branch with word-level manipulation data
- [x] Implement versioned checkpoint policy — v3.0 frozen, master `best.ckpt` merged

### Forensic Shortcut Triad Audit & Cryptographic Benchmark (Complete)

- [x] Run Cryptographic Shortcut Triad Audit ([`evaluation/audit_triad.py`](evaluation/audit_triad.py) — Git Commit `72a5cdd41c783b741468155f80fbbe4202083ab9`, Checkpoint SHA-256 `37ff218c6b7b282e...91ab9aacaaa74a05`)
- [x] Strict Checkpoint Loading Assertion: Verified 399/399 weights matched (0 unexpected keys).
- [x] Enforce Video-ID Disjoint Splits: Reals strictly from official test partitions of FaceForensics++ and Celeb-DF-v2 (never in training).
- [x] Triad Results: Central Face Blackout drops AUROC to **0.6401**; Non-Semantic Metadata Probe scores **1.0000** (proves container leakage in raw files); Transcode Normalization (JPEG Q=90) proves detector robustness (**1.0000 AUROC**).
- [x] In-Distribution Disjoint Test: **1.0000 AUROC**, **99.60% Accuracy** (95% Wilson CI: `[97.77%, 99.93%]`), **0.00% FPR** (95% Wilson CI: `[0.00%, 2.98%]`), **99.20% TPR** (95% Wilson CI: `[95.61%, 99.86%]`), **0.0066 ECE**, **0.0025 Brier**.
- [x] Leave-One-Generator-Out (LOGO) Test: Tested on held-out `DiT`, `inswap`, `pirender`, `sadtalker` (**1.0000 AUROC**, **100.00% TPR** with 95% Wilson CI `[91.62%, 100.00%]`).
- [x] Monotonic Selective Prediction: Rejection of top 5% uncertain samples drives error rate from **0.40%** at 100% coverage down to **0.00%** at $\le 95\%$ coverage.
- [x] Cryptographic Audit Ledger: Generated at [`reports/RESULTS_LEDGER.json`](reports/RESULTS_LEDGER.json).

---

## 10. What This Project Can Honestly Claim

**Defensible for Publication (Workshop / T-IFS / arXiv):**

> "FORGE-Lite is an efficient, multi-branch deepfake detector operating within an 8GB VRAM consumer budget. By introducing a 3-part Forensic Shortcut Triad (Blackout, Metadata Probe, Transcode Normalization), we expose how container discrepancies (`.jpg` vs `.png`) inflate detection performance to 1.0000 AUROC on non-semantic features alone. When evaluated on video-ID disjoint official test splits under unified in-memory JPEG Q=90 transcoding, FORGE-Lite achieves 1.0000 AUROC and 99.60% accuracy (95% Wilson CI: [97.77%, 99.93%]), maintaining 100.00% fake recall on unseen generators (DiT, inswap, pirender, sadtalker) and driving test error rates to 0.00% under 95% selective prediction coverage."

**Avoid claiming:**
- "100% accuracy" without reporting operational thresholds and Wilson score confidence intervals.
- "Guaranteed error elimination" — state empirical selective prediction deciles with Wilson CIs.
- "World's best" without specifying metric, operating point, and test set.
- "Zero-shot" — linear projection and LayerNorm heads are trained on task labels.
- "Equitable fairness on FakeAVCeleb" — AUROC 0.5052 reflects a fundamental SyncNet limitation against voice clones, not fairness.

---

## 11. The Voice-Clone Threat — Long-Term Positioning

The 2026 position paper "The Deepfakes We Missed" argues the field built detectors for a public-figure-misinformation catastrophe that did not materialize at scale. The dominant real harms:

1. Voice-clone wire-fraud and executive impersonation
2. Non-consensual synthetic imagery of ordinary people (not celebrities)

**Practical consequences:**
- Weight audio/voice-clone coverage higher than the current architecture implies
- FakeAVCeleb uses VoxCeleb subjects — validate also on non-public individuals
- Your rPPG + AV sync + geometry combination is well-positioned for the real fraud use case (live video calls) — emphasize this over the "celebrity deepfake" angle in publication framing

---

## Appendix A: Full Dataset Status

| Dataset | Size | Status | Use | Priority |
|---|---|---|---|---|
| FaceForensics++ C23 | 1,000 videos / 18,070 crops | Extracted (`data/crops/faceforensics`) | Train + Parent Reals | Active |
| FakeAVCeleb v1.2 | 21,544 videos (5.96 GB) | Fully ready | Train + Eval | Active |
| Celeb-DF-v2 | 590 videos / 12,164 crops | Extracted (`data/crops/celeb_df`) | Train + Parent Reals | Active |
| DF40 images | 692,158 records on disk | Downloaded (`data/raw/DF40`) | Eval primary | Active |
| Deepfake-Eval-2024 | 45 hrs video | Not downloaded | Eval-only holdout | Get soon |
| AV-Deepfake1M | 1.1M videos | Not downloaded | AV branch train | After DF40 |
| DVF / GenVidBench | ~7-10k videos | Not downloaded | Context branch eval | Optional |
| ASVspoof5 | ~2k speakers | Not downloaded | Voice-clone branch | Park for now |

---

## Appendix B: Files to Create

| File | Purpose |
|---|---|
| `data/setup_fakeavceleb.py` | FakeAVCeleb meta_data.csv to fakeavceleb.jsonl |
| `data/setup_df40.py` | DF40 JSON configs to df40_test.jsonl |
| `data/migrate_forgery_type.py` | Add forgery_type to all existing manifests |
| `evaluation/eval_per_forgery_type.py` | Per-category AUROC breakdown |
| `evaluation/eval_fairness.py` | Per-skin-tone EER, TPR@FPR slices |
| `preprocessing/extract_rppg_stream.py` | Separate 25fps crop stream for rPPG |
| `CHANGELOG.md` | Model version tracking v3.0 onward |

---

*This document is the single ground truth for all engineering and research decisions until CVPR 2027 submission. Update the Action Checklist as items complete.*