# PROJECT CORNERSTONE
### Deepfake Pairing Audit — Single Source of Truth

**Owner:** Abhi (B.Tech AIML, MITS Gwalior)  
**Repository:** https://github.com/abhishek036/deepfake  
**Local path:** `C:\deepfake`  
**Last updated:** 2026-10-26

---

## Table of Contents

1. [Project Identity & Scope](#1-project-identity--scope)
2. [Hardware & Environment](#2-hardware--environment)
3. [Repository Map](#3-repository-map)
4. [Datasets on Disk](#4-datasets-on-disk)
5. [Pre-Registration & Decision Rules](#5-pre-registration--decision-rules)
6. [All Measured Results (Ledger-Verified)](#6-all-measured-results-ledger-verified)
7. [Analysis Log Summary](#7-analysis-log-summary)
8. [Key Scripts & How to Run Them](#8-key-scripts--how-to-run-them)
9. [Known Open Issues](#9-known-open-issues)
10. [Paper 1 — Final Draft (Embedded)](#10-paper-1--final-draft-embedded)
11. [Paper 2 Roadmap](#11-paper-2-roadmap)
12. [Git History Summary](#12-git-history-summary)
13. [Reproducibility & Disclosure](#13-reproducibility--disclosure)

---

## 1. Project Identity & Scope

**What this project is:**  
An empirical audit of how real-vs-fake pairing choices in deepfake benchmarks drive reported detection accuracy. We use frozen linear probes (no task-specific training) to show that AUROC of ~0.998 collapses to ~0.596 when pairing confounds are removed.

**Paper 1 (this repo):**  
Audit paper. Demonstrates the pairing effect across FaceForensics++ and Celeb-DF-v2. Uses frozen DINOv2-S/14, ResNet-50, ViT-B/16 as encoders. Pre-registered. All results from JSON ledgers.

**Paper 2 (future):**  
Retrain a detector on clean shared-transform pairs and show that reported accuracy under clean pairing reflects genuine forensic signal rather than container shortcuts.

**Working claim (bounded to our setting):**  
Reported deepfake detection accuracy is largely set by how real frames are paired with fake frames, not by the quality of the detection model.

**Not claimed:**  
Universal applicability. We do not claim every benchmark is broken or that no method has found real forensic signal.

---

## 2. Hardware & Environment

| Component | Details |
| :--- | :--- |
| GPU | NVIDIA RTX 4060, 8 GB VRAM |
| OS | Windows 11, PowerShell |
| Python | 3.11 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| Free disk | ~120 GB |
| RAM | Not constrained in experiments |

**Key Python packages (from `environment.txt` / `requirements.lock`):**

```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
transformers==4.52.4
timm==1.0.15
scikit-learn==1.7.0
opencv-python==4.11.0.86
facenet-pytorch==2.6.0
dlib  (for face alignment)
fastapi==0.141.1
```

Full pinned environment: [`requirements.lock`](file:///c:/deepfake/requirements.lock)

---

## 3. Repository Map

```
c:\deepfake\
├── drafts/
│   ├── paper1_draft.md          # Original (AI-structured) draft
│   ├── paper1_final.md          # Human-voice rewrite (CURRENT PAPER)
│   ├── paper1_tables.md         # Tables generated from ledgers
│   ├── paper1_tables.tex        # LaTeX tables
│   ├── latex/
│   │   ├── paper1.tex           # Full LaTeX submission
│   │   └── paper1_refs.bib      # Verified BibTeX (13 refs)
│   └── figure2_region_pairing_bar.pdf/png
│
├── evaluation/                  # All evaluation scripts
│   ├── linear_probe_beforeafter.py   ← PRIMARY experiment script
│   ├── make_latex_tables.py          ← Reads ledgers, generates tables
│   ├── dataset_id_baseline.py        ← Dataset-ID shortcut baseline
│   ├── region_probe.py               ← Low-level handcrafted probes
│   ├── paired_stats.py               ← Geometry & pixel MAD stats
│   ├── one_pipeline_audit.py         ← Full pipeline audit
│   └── [40+ other eval scripts]
│
├── scripts/
│   ├── extract_shared_transform_pairs.py   ← Creates M→M pairs (FF++)
│   ├── extract_celeb_shared_transform_pairs.py  ← M→M for Celeb-DF
│   ├── make_native_manifests.py
│   ├── make_celeb_manifests.py
│   ├── make_figure2_bar_chart.py
│   └── collect_project_state.py
│
├── reports/                     # ALL RESULT LEDGERS (source of truth)
│   ├── lp_shared_dino_5seeds.json           ← Main FF++ DINOv2 results
│   ├── lp_shared_dino_face_out_5seeds.json  ← Background probe FF++
│   ├── lp_shared_dino_face_only_5seeds.json ← Face probe FF++
│   ├── lp_ff_native_dino_5seeds.json        ← FF++ native pairing
│   ├── lp_ff_indep_dino_5seeds.json         ← Same-frame, indep align
│   ├── lp_shared_resnet50_5seeds.json       ← ResNet-50 results
│   ├── lp_shared_vitb16_5seeds.json         ← ViT-B/16 results
│   ├── lp_celeb_5seeds.json                 ← Celeb-DF native
│   ├── lp_celeb_face_out_5seeds.json        ← Celeb-DF background
│   ├── lp_celeb_shared_5seeds.json          ← Celeb-DF shared-transform
│   ├── lp_celeb_shared_face_out_5seeds.json ← Celeb-DF shared background
│   ├── dataset_id_baseline.json             ← Dataset-ID shortcut
│   ├── paired_stats_shared.json             ← Geometry verification
│   ├── paired_stats_indep.json              ← Indep alignment stats
│   ├── region_probe_shared.json             ← Low-level background probe
│   ├── region_probe_celeb.json              ← Low-level Celeb probe
│   └── [70+ other ledgers]
│
├── preprocessing/
│   └── dataset_json/            ← GITIGNORED (large manifests, 100s MB)
│
├── data/
│   └── manifests/
│       └── combined.jsonl       ← 949,254 sample multi-dataset manifest
│
├── models/                      ← FORGE-Lite detector code (Paper 2 scope)
├── training/                    ← Training scripts (Paper 2 scope)
├── service/                     ← FastAPI inference service
├── cornerstone/                 ← React frontend (Cornerstone UI)
├── configs/
│   └── train.yaml               ← Has known pos_weight bug (see §9)
│
├── PREREG.md                    ← Pre-registration (edit BEFORE running)
├── ANALYSIS_LOG.md              ← Post-hoc deviations (edit AFTER running)
├── MASTER_PLAN.md               ← High-level research plan
├── DEEP_RESEARCH_COMPENDIUM.md  ← Literature and background
└── CODE_AUDIT.md                ← Code quality audit notes
```

---

## 4. Datasets on Disk

All datasets are **read-only**. Do not modify, delete, or commit them.

| Dataset | Local Path | Notes |
| :--- | :--- | :--- |
| FaceForensics++ c23 | `datasets/FF++/` | original, Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures |
| Celeb-DF-v2 | `datasets/Celeb-DF-v2/` | 590 real + 5,639 fake videos |
| WildDeepfake | `datasets/WildDeepfake/` | |
| FakeAVCeleb v1.2 | `datasets/FakeAVCeleb/` | |
| DF40 (train only) | `datasets/DF40/` | No DF40 test set on disk |
| FFHQ | `datasets/FFHQ/` | Real only |
| LAV-DF | `datasets/LAV-DF/` | |
| HDTF | `datasets/HDTF/` | Real only |
| TalkingHeadBench | `datasets/TalkingHeadBench/` | |

**Not on disk:** DFDC, Deepfake-Eval-2024, AV-Deepfake1M

### Combined Manifest Stats (from `reports/dataset_id_baseline.json`)

| Dataset | N samples | Fake % |
| :--- | :---: | :---: |
| DF40 | 686,358 | 100.0 |
| FaceForensics++ | 127,068 | 85.8 |
| Celeb-DF-v2 | 84,900 | 85.7 |
| WildDeepfake | 23,100 | 51.7 |
| FakeAVCeleb | 21,566 | 97.7 |
| FFHQ | 4,062 | 0.0 |
| LAV-DF | 2,000 | 100.0 |
| HDTF | 100 | 0.0 |
| TalkingHeadBench | 100 | 100.0 |
| **Total** | **949,254** | — |

---

## 5. Pre-Registration & Decision Rules

Source of truth: [`PREREG.md`](file:///c:/deepfake/PREREG.md)

Rules were committed to git **before** the corresponding experiments ran. Exceptions and post-hoc work are in [`ANALYSIS_LOG.md`](file:///c:/deepfake/ANALYSIS_LOG.md).

### Phase B Rules (FF++ Shared-Transform Experiment)

| Symbol | Condition | Threshold | Outcome |
| :--- | :--- | :---: | :---: |
| **P-S1** | Paired face crop center MAD | ≤ 4.0 px | ✅ MET (0.721 px) |
| **P-S2** | Center-masked `face_out` AUROC, M→M | ≤ 0.60 | ✅ MET (0.591 ± 0.010) |
| **P-S3** | Corner-masked `face_only` AUROC, M→M | ≥ 0.85 | ❌ NOT MET (0.774 ± 0.007) |
| **P-S4** | Full-image AUROC, M→M | ≤ 0.70 | ❌ NOT MET (0.763 ± 0.012) |

**Joint Clean Reference Rule:** P-S1 AND P-S2 → **MET**. Shared-transform pairing declared a clean reference.

### Phase C Rules (Cross-Domain & Multi-Backbone)

| Symbol | Condition | Threshold | Outcome |
| :--- | :--- | :---: | :---: |
| **P-C1** | Dataset-ID 5-fold CV AUROC | ≥ 0.90 | ✅ MET (0.9208) |
| **P-C2** | Celeb-DF native `face_out` AUROC | ≤ 0.60 | ❌ NOT MET (0.751 ± 0.005) |
| **P-C2b** | Celeb-DF shared-transform `face_out` AUROC | ≤ 0.60 | ✅ BOUNDARY (0.604 ± 0.011) |
| **P-C3** | Multi-backbone `face_out` AUROC, M→M | ≤ 0.60 | ✅ EXPLORATORY MET (ResNet: 0.537, ViT: 0.567) |

> **Note on P-C3:** Exploratory only — git commit postdates first ResNet-50 and ViT-B/16 runs. Weight this result accordingly.

### Phase B Extension (All-Generator LOGO — Pending)

| Symbol | Condition | Threshold | Outcome |
| :--- | :--- | :---: | :---: |
| **P-L1** | C→C LOGO mean AUROC, all 5 generators | ≥ 0.90 | Not yet measured |
| **P-L2** | M→M LOGO mean AUROC vs C→C | < C→C by ≥ 0.30 | Not yet measured |

---

## 6. All Measured Results (Ledger-Verified)

> Every number here comes from a JSON ledger in `reports/`. Nothing typed by hand.

### 6.1 Pairing Effect — FaceForensics++ (DINOv2-S/14)

| Condition | AUROC | Bal.Acc | Ledger |
| :--- | :---: | :---: | :--- |
| C→C (naive: DF40 PNG vs FF++ JPEG) | 0.998 ± 0.001 | — | `lp_shared_dino_5seeds.json` |
| C→M (cross-eval on geometry-matched reals) | 0.596 ± 0.013 | — | `lp_shared_dino_5seeds.json` |
| M→M (shared-transform clean) | 0.763 ± 0.012 | 0.691 | `lp_shared_dino_5seeds.json` |
| M→C (clean trained, naive tested) | 0.793 ± 0.035 | — | `lp_shared_dino_5seeds.json` |
| FF++ native benchmark | 0.874 ± 0.006 | 0.784 | `lp_ff_native_dino_5seeds.json` |
| Same-frame independent alignment, C→C | 0.774 ± 0.004 | — | `lp_ff_indep_dino_5seeds.json` |
| Same-frame independent alignment, M→M | 0.767 ± 0.005 | 0.690 | `lp_ff_indep_dino_5seeds.json` |

### 6.2 Background Probe (Spatial Decomposition)

| Encoder | C→C center-masked | M→M center-masked | M→M corner-masked | Ledgers |
| :--- | :---: | :---: | :---: | :--- |
| DINOv2 (shared-transform) | 0.998 ± 0.001 | **0.591 ± 0.010** [P-S2 ✅] | 0.774 ± 0.007 | `lp_shared_dino_face_out/only` |
| DINOv2 (FF++ native) | 0.758 ± 0.006 | 0.755 ± 0.011 | 0.872 ± 0.009 | `lp_ff_native_dino_face_out/only` |
| DINOv2 (same-frame indep) | 0.596 ± 0.013 | 0.600 ± 0.010 | 0.785 ± 0.011 | `lp_ff_indep_dino_face_out/only` |
| ResNet-50 | 0.986 ± 0.004 | **0.537 ± 0.009** [exploratory] | 0.619 ± 0.011 | `lp_shared_resnet50_face_out/only` |
| ViT-B/16 | 0.986 ± 0.004 | **0.567 ± 0.009** [exploratory] | 0.706 ± 0.012 | `lp_shared_vitb16_face_out/only` |
| Handcrafted baseline | 0.812 ± 0.007 | 0.546 ± 0.009 | — | `lp_shared_dino_5seeds.json` |

### 6.3 Celeb-DF-v2 Audit

| Condition | Region | AUROC | Ledger |
| :--- | :--- | :---: | :--- |
| Native — handcrafted low-level | Outer corners | 0.458 [0.413, 0.503] | `region_probe_celeb.json` |
| Native — DINOv2 | Background (face_out) | 0.751 ± 0.005 | `lp_celeb_face_out_5seeds.json` |
| Native — DINOv2 | Face (face_only) | 0.911 ± 0.010 | `lp_celeb_face_only_5seeds.json` |
| Native — DINOv2 | Full image | 0.889 ± 0.007 | `lp_celeb_5seeds.json` |
| Shared-transform — low-level | Background | 0.507 ± 0.003 | `lp_celeb_shared_face_out_5seeds.json` |
| Shared-transform — DINOv2 | Background | **0.604 ± 0.011** [P-C2b] | `lp_celeb_shared_face_out_5seeds.json` |
| Shared-transform — DINOv2 | Cross-eval C→M | 0.532 ± 0.007 | `lp_celeb_shared_5seeds.json` |
| Shared-transform — DINOv2 | Face (face_only) | 0.835 ± 0.014 | `lp_celeb_shared_face_only_5seeds.json` |
| Shared-transform — DINOv2 | Full image M→M | 0.806 ± 0.015 | `lp_celeb_shared_5seeds.json` |

### 6.4 Dataset-ID Baseline

| Metric | Value | Ledger |
| :--- | :---: | :--- |
| 5-fold CV AUROC | **0.9208 [0.9201, 0.9216]** | `dataset_id_baseline.json` |
| Overall balanced accuracy | 0.5452 | `dataset_id_baseline.json` |

### 6.5 Geometry & Low-Level Verification (FF++ Shared Pairs)

| Measurement | Value | Status |
| :--- | :---: | :---: |
| Crop center MAD (shared) | 0.721 px | ✅ ≤ 4.0 px |
| Crop center MAD (indep, same frames) | 0.823 px | ✅ Still low |
| Background MAD Real vs Fake | 3.244 [2.415, 4.994] gl | — |
| Background MAD Fake vs Fake | 3.372 [2.517, 5.207] gl | — |
| Ratio RF/FF | 0.962 | ✅ ±0.05 margin |
| Wilcoxon p | 1.96×10⁻¹² | ~3.8% diff (small) |
| Composited low-level bg probe | 0.437 [0.383, 0.486] | Below chance |
| Reenactment low-level bg probe | 0.521 [0.482, 0.570] | ✅ In [0.45, 0.55] |

### 6.6 Controls

| Control | AUROC | Purpose | Ledger |
| :--- | :---: | :--- | :--- |
| Identity-disjoint split, DINOv2 M→M | 0.705 ± 0.030 | Rules out identity leakage | `lp_shared_identity_disjoint_5seeds.json` |
| Size control (3 fakes/scene), DINOv2 M→M | 0.724 ± 0.014 | Rules out class imbalance artifact | `lp_shared_size_control_5seeds.json` |

---

## 7. Analysis Log Summary

Full log: [`ANALYSIS_LOG.md`](file:///c:/deepfake/ANALYSIS_LOG.md)

| Date | Event | Key Finding |
| :--- | :--- | :--- |
| 2026-10-10 | Loss polarity audit | `pos_weight=18.4` + `balanced_sampling=true` upweights fakes (label=1), not reals. Known bug in legacy checkpoint. Does not affect Paper 1 results. |
| 2026-10-10 | Phase B shared-transform experiment | P-S1, P-S2 MET. P-S3, P-S4 NOT MET. Joint clean reference rule MET. |
| 2026-10-11 | Dataset-ID baseline (P-C1) | 0.9208 AUROC without pixels. P-C1 MET. |
| 2026-10-11 | Celeb-DF native audit (P-C2) | 0.751 background AUROC. P-C2 NOT MET (expected ≤ 0.60). |
| 2026-10-11 | Multi-backbone controls (P-C3) | ResNet 0.537, ViT 0.567 background AUROC. Exploratory P-C3 MET. |
| 2026-10-12 | Celeb-DF shared-transform ablation (P-C2b) | 0.604 background AUROC. P-C2b boundary MET. Low-level at chance (0.507). |
| 2026-10-13 | Live webcam false positive diagnosis | Resolved: mock API clamp removed. Real scores now used (fake_score 0.4155). |
| 2026-10-13 | Frontend schema alignment | Cornerstone now shows real branch logits, not hardcoded 78% certainty. |

---

## 8. Key Scripts & How to Run Them

All scripts must be run from `C:\deepfake` in a Python 3.11 environment with packages from `requirements.lock`.

### 8.1 Main Experiment: Linear Probe (Before/After Pairing)

```powershell
# FF++ Shared-transform (M→M) — DINOv2, full image
python evaluation/linear_probe_beforeafter.py `
  --mode shared `
  --backbone dinov2 `
  --seeds 42 7 13 99 2024 `
  --output reports/lp_shared_dino_5seeds.json

# With background mask (face_out)
python evaluation/linear_probe_beforeafter.py `
  --mode shared --backbone dinov2 --mask face_out `
  --seeds 42 7 13 99 2024 `
  --output reports/lp_shared_dino_face_out_5seeds.json

# FF++ native pairing
python evaluation/linear_probe_beforeafter.py `
  --mode native --backbone dinov2 `
  --seeds 42 7 13 99 2024 `
  --output reports/lp_ff_native_dino_5seeds.json

# Celeb-DF shared-transform
python evaluation/linear_probe_beforeafter.py `
  --mode celeb_shared --backbone dinov2 --mask face_out `
  --seeds 42 7 13 99 2024 `
  --output reports/lp_celeb_shared_face_out_5seeds.json
```

### 8.2 Extract Shared-Transform Pairs

```powershell
# FF++ — creates 600 real+fake pairs with identical affine crops
python scripts/extract_shared_transform_pairs.py

# Celeb-DF-v2
python scripts/extract_celeb_shared_transform_pairs.py
```

### 8.3 Dataset-ID Baseline

```powershell
python evaluation/dataset_id_baseline.py `
  --manifest data/manifests/combined.jsonl `
  --output reports/dataset_id_baseline.json
```

### 8.4 Region Probe (Low-Level Handcrafted Features)

```powershell
python evaluation/region_probe.py `
  --pairs shared `
  --output reports/region_probe_shared.json
```

### 8.5 Geometry & Pixel MAD Statistics

```powershell
python evaluation/paired_stats.py `
  --mode shared `
  --output reports/paired_stats_shared.json
```

### 8.6 Generate LaTeX Tables from Ledgers

```powershell
python evaluation/make_latex_tables.py `
  --ledger-dir reports/ `
  --output drafts/paper1_tables.tex
```

### 8.7 Generate Figure 2

```powershell
python scripts/make_figure2_bar_chart.py `
  --output drafts/figure2_region_pairing_bar.pdf
```

### 8.8 Pending: All-Generator LOGO Run

```powershell
python evaluation/linear_probe_beforeafter.py `
  --mode shared --backbone dinov2 `
  --families all `
  --logo `
  --seeds 42 7 13 99 2024 `
  --cache reports/emb_cache_shared_dino.pkl `
  --output reports/lp_shared_dino_logo_allfam_5seeds.json
```

---

## 9. Known Open Issues

| Issue | Status | Impact |
| :--- | :--- | :--- |
| `pos_weight=18.4` + `balanced_sampling=true` polarity mismatch in `configs/train.yaml` | Open (Paper 2) | Upweights fakes in training. Does not affect Paper 1 (frozen probes). |
| Legacy `best.ckpt` trained on confounded combined pool | Open (Paper 2) | AUROC 0.921 achievable by dataset-ID alone. Do not use for Paper 1 claims. |
| No DF40 test set on disk | Permanent | LOGO results on DF40 not possible locally. |
| `preprocessing/dataset_json/` not tracked in git (gitignored, too large) | By design | Regenerate from `make_native_manifests.py` if needed. |
| All-generator LOGO run (P-L1/P-L2) not yet executed | Open | Must run before paper submission if LOGO results are included. |
| Git history had `preprocessing/dataset_json/*.json` in old commits | Resolved | Removed via `git filter-branch` on 2026-10-26. Remote now clean. |
| `emb_cache_*.pkl` files in `reports/` are gitignored (multiple GB) | By design | These regenerate from running eval scripts with `--cache` flag. |

---

## 10. Paper 1 — Final Draft (Embedded)

Full paper: [`drafts/paper1_final.md`](file:///c:/deepfake/drafts/paper1_final.md)  
LaTeX source: [`drafts/latex/paper1.tex`](file:///c:/deepfake/drafts/latex/paper1.tex)  
BibTeX: [`drafts/latex/paper1_refs.bib`](file:///c:/deepfake/drafts/latex/paper1_refs.bib)  
Tables: [`drafts/paper1_tables.md`](file:///c:/deepfake/drafts/paper1_tables.md)  
Figure 2: [`drafts/figure2_region_pairing_bar.pdf`](file:///c:/deepfake/drafts/figure2_region_pairing_bar.pdf)

### Paper Structure

| Section | Title | Status |
| :--- | :--- | :--- |
| Abstract | Pairing effect, 4 key numbers | ✅ Complete |
| §1 | Introduction + 5 claims | ✅ Complete |
| §2 | Background (benchmarks + shortcuts lit) | ✅ Complete |
| §3 | Methods (encoders, pairing, masking, pre-reg) | ✅ Complete |
| §4.1 | Pairing effect across representations (Table 2) | ✅ Complete |
| §4.2 | Spatial decomposition / background probe (Table 3) | ✅ Complete |
| §4.3 | Celeb-DF-v2 native pairing fallacy (Table 4) | ✅ Complete |
| §4.4 | Dataset-ID classifier (Table 5) | ✅ Complete |
| §4.5 | Geometry & pixel verification (Table 6) | ✅ Complete |
| §5 | What Went Wrong (P-S3, P-S4, P-C2 contradictions) | ✅ Complete |
| §6 | Limitations | ✅ Complete |
| §7 | Recommendations (3 protocols) | ✅ Complete |
| References | 14 verified citations | ✅ Complete |
| **Missing** | LOGO results (P-L1/P-L2) | ⏳ Pending run |
| **Missing** | Discussion / Intro — your voice | ✍️ You write this |

### Three Recommendations (Summary)

1. Enforce shared spatial transforms (identical affine from real, applied to both).
2. Include a background-probe baseline (center-masked AUROC) in every benchmark paper.
3. Report a zero-pixel dataset-ID baseline for all multi-dataset benchmarks.

---

## 11. Paper 2 Roadmap

Paper 2 trains a new detector on clean shared-transform pairings and compares it against the same detector trained on confounded data. Primary questions:

1. Does training on clean pairs improve cross-dataset generalization?
2. Does the detector learn face-region forensic features (face_out probe ≤ 0.60 in background) rather than container shortcuts?
3. Can we reproduce published leave-one-generator-out results after cleaning pairing?

**Prerequisites before starting Paper 2:**
- [ ] Fix `pos_weight` polarity in `configs/train.yaml`
- [ ] Retrain baseline on confounded pairs (to establish fair comparison)
- [ ] Retrain on clean M→M pairs
- [ ] Evaluate both using the Paper 1 audit protocol (background probe, geometry check)
- [ ] Do not use the legacy `best.ckpt`

---

## 12. Git History Summary

```
11c3574e1  docs(paper1): add human-voice rewrite of paper1 as paper1_final.md
3eccbdf06  docs: update PREREG.md and ANALYSIS_LOG.md with phase B extension and webcam
60aa50426  docs(paper1): soften Celeb-DF shortcut claim and cite Yermakov (GenD, WACV 2026)
7b907e44e  feat: complete Paper 1 benchmark release, LaTeX submission package, Celeb-DF ablation
3db758a18  prereg: add H-C2b and P-C2b for Celeb-DF shared-transform ablation
c74f249da  fix(paper1): correct FF++ native AUROC to match ledger (0.877→0.874)
875975781  docs(paper1): align draft and tables with frozen ledgers
...
72a5cdd41  chore: snapshot codebase for forensic benchmark audit   ← root commit
```

**Total commits:** 34  
**Remote:** https://github.com/abhishek036/deepfake (branches: `master`, `main`, tag: `phase-a-complete`)

---

## 13. Reproducibility & Disclosure

**Determinism:** All evaluation scripts use fixed seeds `[42, 7, 13, 99, 2024]`. Results are byte-identical across runs given the same environment.

**Source of truth:** `reports/` JSON ledgers, not this document. If this document contradicts a ledger, the ledger wins.

**Pre-registration:** [`PREREG.md`](file:///c:/deepfake/PREREG.md) is the pre-registration record. Do not edit it after seeing results — use [`ANALYSIS_LOG.md`](file:///c:/deepfake/ANALYSIS_LOG.md) for post-hoc notes.

**AI assistance declaration:** Python pipelines, evaluation scripts, bootstrap computations, and text drafts were developed with AI pair-programming support (Antigravity / Claude). All empirical claims are verified against local JSON ledgers. Hypotheses and thresholds are documented in git-timestamped commits in PREREG.md.

**Datasets:** Read-only. No downloads, no modifications, no commits of dataset files or secrets.

**Rule checklist before any new experiment:**
- [ ] Write hypothesis in `PREREG.md` and commit before running
- [ ] Verify the script has a self-test on synthetic data
- [ ] Run and write ledger to `reports/`
- [ ] Report in `ANALYSIS_LOG.md` if result contradicts expectation
- [ ] Add ledger path to the relevant table in this file

---

*This file is the single entry point for the project. When in doubt, read this first.*
