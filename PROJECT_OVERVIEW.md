# realitycheck.fr — Complete Project Overview

> **realitycheck.fr (formerly FORGE-Lite) — Forensic, Robust, Generalizable Evidence Detection for Facial Deepfakes and Synthetic Media**
>
> Version: `1.0.0` | Target: CVPR 2027 | Hardware: RTX 4060 8 GB | Python >= 3.10


---

### ⚡ What Was Optimized

1. **Integrated `make_dataloader` with Balanced Sampling**:
   - Updated [`training/train_temporal.py`](training/train_temporal.py), [`training/train_fusion.py`](training/train_fusion.py), and [`training/train_av.py`](training/train_av.py) to use `make_dataloader` instead of raw un-sampled `DataLoader`.
   - Capped training at `samples_per_epoch: 40000` with 50% real / 50% fake `WeightedRandomSampler`.
   - **Training batches dropped from 26,125 $\rightarrow$ 2,500 batches per epoch (10.5× reduction!)**.

2. **Doubled Batch Size**:
   - In [`configs/train.yaml`](configs/train.yaml), increased `batch_sizes.temporal: 16` (up from 8), taking advantage of the frozen spatial backbones without exceeding 4.2 GB VRAM.

3. **Stratified Representative Validation Cap**:
   - In [`data/dataset.py`](data/dataset.py), added deterministic uniform random sampling across all datasets when `max_samples` is specified.
   - Set `max_val_samples: 5000` for sequence evaluation.
   - **Validation time per epoch dropped from 25–30 minutes $\rightarrow$ ~45 seconds!**

---

### ⏱️ Performance Comparison

| Metric | Previous Un-optimized Phase 4 | **Optimized Phase 4 (Current)** | Improvement |
|---|---|---|---|
| **Training Samples / Epoch** | 209,004 records | **40,000 balanced samples** | **5.2× less redundant I/O** |
| **Batch Size** | 8 | **16** | **2× GPU throughput** |
| **Training Batches / Epoch** | 26,125 batches | **2,500 batches** | **10.5× faster** |
| **Validation Batches** | 5,327 batches (~25 min) | **312 batches (~45 sec)** | **33× faster** |
| **Estimated Epoch Duration** | ~2 to 2.5 hours | **~7 to 8 minutes** ⚡ | **~15× overall speedup** |

---

### 🚀 Current Live Status (as of 03-Oct-2026 08:43 IST)

- **Phase 4 (Temporal Branch)**: **COMPLETED ✅** (`val/auc = 0.9957` / 99.57%)
- **Phase 5 (Full Fusion + Localization)**: **RECORD PERFORMANCE ✅** (`val/auc = 0.9960` / 99.60%)
- **Phase 6 (Audio-Visual Branch)**: **COMPLETED ✅**
  - **Dataset**: `talkingheadbench.jsonl` (talking head lip-sync desynchronization)
  - **Best Val AUC**: **`0.9872` (98.72%)**
  - **Saved Checkpoint**: [`forge-lite-v1_av_branch_best.ckpt`](outputs/checkpoints/forge-lite-v1_av_branch_best.ckpt)
- **Master Checkpoint Status**: **ALL BRANCHES TRAINED & MERGED 🏆**
  - **File**: [`outputs/checkpoints/best.ckpt`](outputs/checkpoints/best.ckpt) (124 MB)
  - **Tensors Loaded**: **100% matched (0 missing keys, 0 unexpected keys)**
  - Combines: DINOv2 Semantic (99.48%), SRM Forensic (99.48%), Bi-GRU Temporal (99.57%), Multimodal Fusion Head (99.60%), and SyncNet Audio-Visual Branch (98.72%).
- **Hardware Status**: GPU idle (0% utilization, 49°C, 0/8,188 MiB VRAM allocated). Ready for calibration and inference/eval benchmarking.
---

## Table of Contents

1. [Ek Line Mein Kya Hai Ye](#1-ek-line-mein-kya-hai-ye)
2. [Ye Project Alag Kyun Hai](#2-ye-project-alag-kyun-hai)
3. [Project Ka Pura Structure](#3-project-ka-pura-structure)
4. [Model Architecture — 7 Branches](#4-model-architecture--7-branches)
5. [Training Pipeline — 5 Phases](#5-training-pipeline--5-phases)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Data and Datasets](#7-data--datasets)
8. [Service Layer — FastAPI](#8-service-layer--fastapi)
9. [Web Frontend](#9-web-frontend)
10. [Gradio Interactive Demo](#10-gradio-interactive-demo)
11. [Evaluation Framework](#11-evaluation-framework)
12. [Configuration System](#12-configuration-system)
13. [Key Innovations and Design Decisions](#13-key-innovations--design-decisions)
14. [Dependencies at a Glance](#14-dependencies-at-a-glance)
15. [Current State of the Project](#15-current-state-of-the-project)
16. [How to Run — Step by Step](#16-how-to-run--step-by-step)
17. [Output Schema](#17-output-schema)
18. [Forensic Explainability Layer](#18-forensic-explainability-layer)
19. [Provenance and C2PA](#19-provenance--c2pa)
20. [Known Risks and Guardrails](#20-known-risks--guardrails)

---

## 1. Ek Line Mein Kya Hai Ye

FORGE-Lite ek **multi-modal neural deepfake video detector** hai jo ek saath **7 alag-alag evidence branches** ko combine karta hai — visual forensics, semantic understanding, temporal dynamics, facial geometry, physiological signals (heartbeat), audio-visual sync, aur full-frame context — taaki ye decide kar sake ki koi face video **real hai, fake hai, ya evidence insufficient hai**.

Ye ek "detect-and-explain" system hai — sirf ek number nahi deta, balki har branch ka contribution, uncertainty, quality breakdown, aur suspicious time windows bhi explain karta hai.

---

## 2. Ye Project Alag Kyun Hai

| Feature | Typical Deepfake Detectors | FORGE-Lite |
|---|---|---|
| **Architecture** | Single CNN/ViT | 7-branch multimodal fusion |
| **Foundation Model** | None / ImageNet CNN | DINOv2 ViT-S/14 (self-supervised) |
| **Codec Robustness** | Fails on H.264 compression | Offline codec bank + consistency loss |
| **Uncertainty** | Just a score | MC-Dropout + 4-factor uncertainty decomposition |
| **Abstention** | Always gives verdict | `insufficient_evidence` when quality is bad |
| **Physiological Signal** | Not present | rPPG (remote photoplethysmography) heartbeat detection |
| **Audio-Visual** | Not present | SyncNet-style lip-audio desync detection |
| **Explainability** | Black box | Directional evidence, counterfactual attribution, branch weights |
| **Provenance** | Not present | C2PA credential verification (separate from fake score) |
| **Hardware Target** | Cloud GPU | RTX 4060 8 GB consumer GPU |
| **Output** | Float 0-1 | Structured JSON: decision + evidence + uncertainty + mask |

---

## 3. Project Ka Pura Structure

```
deepfake/
|
├── models/                  ← Neural network branches (core)
│   ├── forge_lite.py        ← Main FORGELite nn.Module (top-level composer)
│   ├── dino_branch.py       ← Branch 1: DINOv2 ViT-S/14 semantic
│   ├── forensic_branch.py   ← Branch 2: SRM + Haar CNN (pixel forensics)
│   ├── temporal_branch.py   ← Branch 3: GRU temporal encoder
│   ├── rppg_branch.py       ← Branch 4: Remote photoplethysmography
│   ├── av_branch.py         ← Branch 5: Audio-Visual sync (SyncNet-style)
│   ├── geometry_branch.py   ← Branch 6: 16-dim facial landmark geometry
│   ├── context_branch.py    ← Branch 7: Full-frame context (DINO shared)
│   ├── fusion.py            ← Reliability-Aware Fusion Head
│   ├── localization.py      ← 56×56 manipulation mask decoder
│   ├── calibration.py       ← Temperature scaling + abstention
│   ├── srm_filters.py       ← SRM + Haar fixed filter bank
│   └── metadata_baseline.py ← Metadata-only baseline (ablation)
│
├── service/                 ← Production serving layer
│   ├── api.py               ← FastAPI app (REST endpoints)
│   ├── inference.py         ← Full pipeline: video to decision (879 lines)
│   ├── forensic_explainer.py← Attribution + uncertainty explanation (1196 lines)
│   ├── evidence_report.py   ← Typed EvidenceReport dataclass
│   ├── forensic_analyzer.py ← High-level analyzer wrapper
│   ├── provenance.py        ← C2PA credential verification
│   └── schemas.py           ← Pydantic API schemas
│
├── training/                ← Training scripts (multi-phase)
│   ├── run_training.py      ← Master orchestrator (5 phases)
│   ├── trainer.py           ← Base Trainer class
│   ├── train_spatial.py     ← Phase 1+2: spatial branches
│   ├── train_temporal.py    ← Phase 4: GRU temporal
│   ├── train_fusion.py      ← Phase 5: full fusion
│   ├── train_av.py          ← Audio-Visual branch training
│   ├── adversarial_train.py ← PGD adversarial training
│   ├── calibrate_fusion_heads.py ← Temperature calibration
│   ├── finetune_generative.py    ← Fine-tune on new generators
│   ├── train_v3_codec_invariant.py ← Codec invariance training
│   └── losses.py            ← Composite loss functions
│
├── preprocessing/           ← Video to face crops pipeline
│   ├── preprocess_all.py    ← Full pipeline runner
│   ├── decode.py            ← Video decoding (OpenCV + PyAV)
│   ├── detect_faces.py      ← RetinaFace/MTCNN face detection
│   ├── track_faces.py       ← Multi-frame face tracking
│   ├── align_faces.py       ← 5-pt landmark affine alignment
│   ├── quality.py           ← Quality analysis (blur, pose, occlusion)
│   └── build_manifest.py    ← Dataset manifest builder
│
├── data/                    ← Dataset loading & augmentation
│   ├── dataset.py           ← ForgeDataset (main PyTorch Dataset)
│   ├── augmentation.py      ← Online photometric augmentation
│   ├── codec_bank.py        ← Offline codec re-encoding (H.264 variants)
│   ├── splits.py            ← Train/val/test split management
│   ├── source_paired_sampler.py ← Paired (real, fake) sampling
│   └── synthetic_test.py    ← Synthetic test video generator
│
├── evaluation/              ← 36 evaluation scripts
│   ├── evaluate_final_system_v3.py  ← Main eval script
│   ├── evaluate_per_generator_v3.py ← Per-generator breakdown
│   ├── evaluate_cross_domain.py     ← Cross-dataset generalization
│   ├── evaluate_robustness.py       ← Codec/perturbation robustness
│   ├── ablation_e0_e7.py            ← Full ablation study (E0-E7)
│   ├── bootstrap_ci.py              ← Bootstrap confidence intervals
│   ├── test_trainsets.py            ← Custom trainset benchmark
│   └── ... (30+ more scripts)
│
├── configs/                 ← YAML configs
│   ├── model.yaml           ← All branch hyperparameters
│   ├── train.yaml           ← Training hyperparameters
│   ├── data.yaml            ← Dataset paths & preprocessing params
│   ├── eval.yaml            ← Evaluation settings
│   └── frozen_v3_policy.json ← Frozen V3 ablation policy
│
├── utils/                   ← Shared utilities
│   ├── visualization.py     ← Heatmap overlays, anomaly timelines
│   ├── forensic_html_report.py ← Full HTML forensic report generator
│   ├── logging_utils.py     ← Structured logging
│   └── vram_monitor.py      ← GPU memory monitoring
│
├── app/
│   └── gradio_app.py        ← Interactive Gradio demo (389 lines)
│
├── web/                     ← Standalone web frontend
│   ├── index.html           ← Dark-theme cyberpunk dashboard
│   ├── style.css            ← ~18KB CSS with glassmorphism
│   └── app.js               ← JavaScript frontend logic
│
├── docker/                  ← Docker configs (deployment)
├── tests/                   ← Pytest unit tests
├── reports/                 ← Auto-generated evaluation reports
├── outputs/                 ← Model checkpoints, logs
├── trainsets/               ← Custom evaluation dataset
|
├── pyproject.toml           ← Package definition + CLI entry points
├── requirements.txt         ← Pip requirements
├── FINAL_DATASET.csv        ← Curated dataset manifest (1.5 MB)
├── CODE_AUDIT.md            ← Code review notes
├── improve.md               ← Improvement backlog
└── newimp.md                ← V3 implementation notes (30KB)
```

---

## 4. Model Architecture — 7 Branches

The main model is `FORGELite` (`models/forge_lite.py`). It is a single `nn.Module` that sequentially processes all branches to manage VRAM on a consumer GPU (RTX 4060 8 GB).

```
Video Input
    |
Preprocessing (decode -> detect -> track -> align -> quality)
    |
+----------------------------------------------------------+
|                  FORGELite Forward Pass                  |
|                                                          |
|  Branch 1 -- DINOSemanticBranch (DINOv2 ViT-S/14)       |
|              -> CLS embedding (384-dim)                  |
|              -> Patch tokens (256 x 384) for localization|
|                                                          |
|  Branch 2 -- ForensicResidualBranch (SRM+Haar CNN)       |
|              -> Forensic embedding (256-dim)             |
|              -> FFT auxiliary vector (20-dim)            |
|              -> Feature map for localization             |
|                                                          |
|  Branch 3 -- TemporalEncoder (GRU)                       |
|              -> Temporal embedding (256-dim)             |
|              -> Per-window anomaly scores                |
|                                                          |
|  Branch 4 -- RPPGBranch (physiological, quality-gated)   |
|              -> rPPG feature vector (16-dim)             |
|              -> Availability mask                        |
|                                                          |
|  Branch 5 -- AudioVisualBranch (SyncNet-style)           |
|              -> AV sync score (scalar)                   |
|              -> Availability-gated (needs audio)         |
|                                                          |
|  Branch 6 -- GeometryFeatureExtractor (landmarks)        |
|              -> Geometry features (16-dim per frame)     |
|                                                          |
|  Branch 7 -- ContextBranch (full-frame DINO, shared)     |
|              -> Context embedding (384-dim)              |
|                                                          |
|  EvidenceAwareModalityRouter                             |
|  -> quality-conditioned branch weights (softmax)         |
|                                                          |
|  ReliabilityAwareFusion MLP                              |
|  -> fake logit + uncertainty                             |
|                                                          |
|  LocalizationHead (DINO patches + forensic feature map)  |
|  -> 56×56 manipulation mask                              |
|                                                          |
|  CalibrationLayer (temperature scaling + MC-Dropout)     |
|  -> calibrated_risk + Decision (real/fake/abstain)       |
+----------------------------------------------------------+
```

### Branch 1 — DINOSemanticBranch (`models/dino_branch.py`)

- **Backbone**: DINOv2 ViT-S/14 (`facebook/dinov2-small`, embed_dim=384)
- **Input**: 224×224 face crop
- **Output**: CLS embedding (384-dim) + 256 patch tokens (16×16 grid)
- **4 Adaptation Modes** (chosen by cross-domain AUROC, NOT in-domain accuracy):
  - `frozen_linear` — frozen backbone + linear head only
  - `layernorm` *(default)* — frozen backbone + trainable LayerNorm params
  - `lora` — frozen + LoRA adapters (rank=8, alpha=16) via `peft`
  - `last_block` — unfreeze last transformer block + head
- **Why DINOv2?**: Self-supervised pre-training on diverse images gives semantic features that generalize better across deepfake generators than supervised ImageNet CNNs.

### Branch 2 — ForensicResidualBranch (`models/forensic_branch.py`)

- **Input**: 35 channels — 1 (luminance) + 30 (SRM filters) + 3 (Haar wavelet subbands) + 1 (noise residual)
- **Architecture**: Depthwise-separable Conv blocks (4 blocks, base_channels=64), GroupNorm, GELU
- **Output**: 256-dim forensic embedding + 20-dim FFT feature vector
- **SRM (Steganalysis Rich Model)**: 30 fixed high-pass filters extracting noise residuals invisible to the human eye but revealing of generation/compression artifacts
- **Haar Wavelet**: 1-level decomposition extracting LH, HL, HH subbands (edge and texture irregularities)
- **Key Risk**: Spectral bias — frequency features can overfit to specific compression levels. Codec consistency training is the primary mitigation.

### Branch 3 — TemporalEncoder (`models/temporal_branch.py`)

- **Architecture**: GRU (hidden_dim=256, 2 layers, bidirectional=False)
- **Input**: Concatenation of DINO + Forensic + FFT + Geometry features (660-dim) per frame
- **Output**: 256-dim temporal embedding aggregated over the video
- **Aggregation modes**: Mean + max + top-10% anomaly windows + temporal dispersion + attention
- **Purpose**: Captures temporal inconsistencies — deepfakes often have frame-level artifacts that flicker inconsistently over time (boundary effects, texture flickering)
- **Window size**: 8 frames (~320ms at 25fps)

### Branch 4 — RPPGBranch (`models/rppg_branch.py`)

- **Measures**: Physiological heart rate signal from skin color micro-variations
- **6 Facial ROIs**: Forehead, left cheek, right cheek, nose bridge, chin, periorbital
- **BPM Range**: 42–210 BPM (0.7–3.5 Hz bandpass)
- **Output**: 16-dim feature vector (SNR, dominant frequency, BPM plausibility, cross-ROI coherence, temporal stability, illumination variation, motion contamination, face visibility)
- **Quality Gating** — ALL must pass:
  - Track duration >= 8 seconds
  - Face area ratio >= threshold
  - Track continuity >= threshold
  - Motion contamination <= threshold
- **Critical Rule**: NEVER output "no rPPG detected therefore fake". Absence = unavailable, NOT = fake.

### Branch 5 — AudioVisualBranch (`models/av_branch.py`)

- **Detects**: Phonetic, temporal, and visual desynchronization between speech audio and lip motion
- **Primary Targets**: Talking-head deepfakes (Wav2Lip, SadTalker, LivePortrait, reenactment synthesis)
- **Architecture**: SyncNet-style dual-stream contrastive encoder:
  - Audio Encoder: 2D ConvNet on 80-mel Mel-spectrogram chunks
  - Lip Visual Encoder: 3D/2D spatiotemporal ConvNet on mouth crops (48×96px, 5-frame windows)
  - Joint projection space → cosine similarity + cross-correlation lag estimation
- **Output**: AV sync score [0=synchronized, 1=desynchronized] + availability mask
- **Gating**: Silently disabled when no audio is present

### Branch 6 — GeometryFeatureExtractor (`models/geometry_branch.py`)

- **Input**: 5-point landmarks per frame (left_eye, right_eye, nose, mouth_left, mouth_right)
- **Output**: 16-dim geometry descriptor per frame, fed into the GRU temporal encoder
- **16 Features per frame** (all normalized by inter-ocular distance for scale invariance):
  - Inter-ocular distance, Eye aspect ratio, Mouth width/height
  - Jaw asymmetry, Nose-to-chin distance, Face width proxy
  - Pose roll, pitch, yaw proxies
  - Eye-to-nose and mouth-to-eye distances
  - Temporal first derivatives: Δ inter-ocular, Δ pose_roll, Δ mouth_width
- **Why**: Deepfakes show subtle geometric anomalies — rigid jaw, abnormal blink rates, inter-ocular discontinuities

### Branch 7 — ContextBranch (`models/context_branch.py`)

- **Input**: Full video frame downsampled to 112×112 (not just face crop)
- **Backbone**: Shared DINOv2 weights with the semantic branch
- **Output**: 384-dim context embedding
- **Gate Weight**: 0.3 (down-weighted in face-localized mode)
- **Purpose**: Captures scene-level inconsistencies (lighting, background, body-face boundary artifacts in fully synthetic videos)

---

### Fusion and Decision

#### EvidenceAwareModalityRouter

```
branch_weight = softmax(quality_encoder(face_quality, branch_quality, availability))
```

- Learns quality-conditioned weights from data — NOT hand-crafted rules
- Face quality vector: 10-dim (blur, brightness, contrast, pose_yaw, pose_pitch, occlusion, face_area, compression, track_continuity, landmark_jitter)
- Video quality vector: 5-dim (fps, resolution_score, duration, audio_present, codec_quality)
- Automatically suppresses low-quality or unavailable branches

#### ReliabilityAwareFusion MLP

- Hidden dim: 512, 2 layers, LayerNorm + GELU, dropout=0.3
- Input: Weighted combination of all branch embeddings
- Output: Single fake logit + aleatoric uncertainty estimate

#### LocalizationHead (`models/localization.py`)

- Decodes DINOv2 16×16 patch tokens + forensic CNN feature maps
- Bilinear upsample → depthwise conv blocks → 56×56 manipulation probability mask
- Only reported on samples with documented ground-truth pixel masks

#### CalibrationLayer (`models/calibration.py`)

- Temperature scaling (learnable scalar T, initialized=1.0)
- MC-Dropout: 10 forward passes → predictive entropy → uncertainty
- **Abstention** (output = `insufficient_evidence`) when ANY of:
  - Evidence coverage < 30% (too many branches unavailable)
  - Uncertainty > 0.35
  - Branch disagreement > 0.4
  - Face quality < 0.2
  - Score margin |score - 0.5| < 0.15 (too close to decision boundary)

---

## 5. Training Pipeline — 5 Phases

All phases orchestrated by `training/run_training.py`.

```
Phase 1 — Semantic Branch (8 epochs)
  Script: train_spatial.py
  Trains DINOv2 + head on face crops
  LR: 5e-4 (layernorm mode), cosine schedule
  Loss: FocalLoss (alpha=0.25, gamma=2.0)

Phase 2 — Forensic Branch (25 epochs)
  Script: train_spatial.py
  Trains SRM+Haar CNN independently
  LR: 1e-3
  Loss: FocalLoss + CodecConsistencyLoss
  Codec variants: clean, h264_crf23, h264_crf30, h264_crf40

Phase 3 — Joint Spatial (continued)
  Joint fine-tune from pre-trained Phase 1 & 2
  LR: 1e-4 (last_block) or 5e-4 (layernorm)

Phase 4 — Temporal Branch (10 epochs)
  Script: train_temporal.py
  Trains GRU on frozen spatial branches
  LR: 5e-4
  Loss: FocalLoss + TemporalConsistencyLoss
  Window: 8 frames

Phase 5 — Full Fusion (10 epochs)
  Script: train_fusion.py
  Trains EvidenceAwareModalityRouter + ReliabilityAwareFusion MLP
  LR: 3e-4
  Loss: All composite losses combined
  Includes: localization loss when masks exist

Additional Training Phases:
  Audio-Visual branch    (train_av.py, 10 epochs)
  Adversarial hardening  (adversarial_train.py, PGD-5, epsilon=0.03)
  Temperature calibration (calibrate_fusion_heads.py, 3 epochs)
  LoRA generative fine-tune (finetune_generative.py)
  Codec-invariant V3 training (train_v3_codec_invariant.py)
```

### Composite Loss Function (`training/losses.py`)

```
Total Loss =
    1.0  × FocalLoss              (classification, imbalance-robust)
  + 0.3  × CodecConsistencyLoss   (compression != forgery)
  + 0.2  × SourcePairMetricLoss   (contrastive, margin=0.5)
  + 0.5  × LocalizationLoss       (binary BCE on pixel masks, when available)
  + 0.1  × TemporalConsistencyLoss
  + 0.15 × GeometryConsistencyLoss
  + 0.20 × AVContrastiveLoss
```

### Hardware Optimizations

- **FP16 mixed precision** (AMP) throughout all phases
- **Gradient accumulation**: 2 steps (effective batch doubles without extra VRAM)
- **Batch sizes**: spatial=32, temporal=8, fusion=8 (all fit in 8 GB VRAM)
- **Windows fix**: `torch.multiprocessing.set_sharing_strategy("file_system")` — prevents page-file limit errors (Error 1455) before any DataLoader
- **Sequential branch processing** in FORGELite.forward() — no parallel branch execution, controls peak VRAM
- **cuDNN autotuner**: `torch.backends.cudnn.benchmark = True`
- **Early stopping**: patience=3 on `val/cross_domain_auroc` — stops if no cross-domain gain for 3 epochs
- **WandB logging**: experiment tracking, image logs every 5 epochs

---

## 6. Preprocessing Pipeline

`preprocessing/preprocess_all.py` runs the full pipeline once before training:

```
Raw Video
    |
VideoDecoder (decode.py)
  - OpenCV/PyAV frame extraction at 8 FPS target
  - Max 300 frames per video, min 8 frames

FaceDetector (detect_faces.py)
  - RetinaFace (primary) or MTCNN (fallback)
  - Min face size: 80px, confidence >= 0.9
  - Max 3 faces per frame, dominant face only

FaceTracker (track_faces.py)
  - Multi-frame track continuity
  - IoU-based track association across frames

FaceAligner (align_faces.py)
  - 5-point affine alignment (left eye, right eye, nose, mouth-left, mouth-right)
  - 224×224 JPEG output with margin=0.4

QualityAnalyzer (quality.py)
  - Blur score (Laplacian variance)
  - Pose angles (yaw: max 45 deg, pitch: max 30 deg)
  - Occlusion score
  - Face area ratio
  - Track continuity score
  - Compression estimate

build_manifest.py
  - Creates JSON manifests per dataset/split
  - Consumed by ForgeDataset at training time
```

### Codec Bank (offline, runs ONCE)

`data/codec_bank.py` creates 6 compressed variants of every training video using FFmpeg:

| Config Name | Codec | CRF | Scale | Blur | Notes |
|---|---|---|---|---|---|
| `clean` | copy | — | 1.0 | — | Original, no re-encoding |
| `h264_crf23` | libx264 | 23 | 1.0 | — | Standard broadcast quality |
| `h264_crf30` | libx264 | 30 | 1.0 | — | Moderate social-media compression |
| `h264_crf40` | libx264 | 40 | 1.0 | — | Heavy compression (messaging apps) |
| `resize_compress` | libx264 | 28 | 0.75 | — | 75% resize + compress |
| `blur_compress` | libx264 | 30 | 1.0 | 1.0 | Pre-blur sigma=1.0 + compress |

During training, ForgeDataset randomly selects one variant per video per epoch, so the model sees all compression levels.

---

## 7. Data & Datasets

### Training Datasets

| Dataset | Type | Identity Split | Notes |
|---|---|---|---|
| **FaceForensics++** | Multi-method | 720/140/140 | Deepfakes, Face2Face, FaceSwap, NeuralTextures; c23+c40 compressions |
| **Celeb-DF-v2** | Face swap | Custom | Celeb-real, Celeb-synthesis, YouTube-real |
| **DFDC Preview** | Multi-method | Custom | Facebook DFDC preview dataset |
| **WildDeepfake** | In-the-wild | Custom | fake_test, real_test (tar.gz archives) |
| **FFHQ** | Real faces | Train only | High-quality real face thumbnails (128px) |

### FINAL_DATASET.csv

A curated 1.5 MB manifest at `FINAL_DATASET.csv` — the primary dataset reference for all training runs.

### Custom Trainsets (`trainsets/`)

A local evaluation set with `selfie_and_video.csv` mapped to actual files in `trainsets/files/`. Used by `evaluation/test_trainsets.py` for quick local benchmarking without access to the full public datasets. Supports filtering by type (video/photo) and limiting sample count.

### ForgeDataset (`data/dataset.py`)

PyTorch Dataset (26KB) that:
- Loads face crop frames from preprocessed directories
- Randomly selects a codec bank variant per video during training (codec augmentation)
- Returns temporal windows (8 frames per sample)
- Applies online photometric augmentation (see below)
- Supports paired (real, fake) from same source video for contrastive learning
- Handles missing optional branches gracefully (sets availability mask = 0)

### Online Augmentation (`data/augmentation.py`)

Cheap image-level transforms ONLY — never online codec re-encoding:
- **Geometric**: Horizontal flip (real videos only — fakes may have chirality artifacts that flipping destroys)
- **Photometric**: Brightness (0.7–1.3×), Contrast (0.7–1.3×), Saturation (0.8–1.2×), Hue (±0.05), Gamma (0.7–1.4×)
- **Degradation**: Gaussian blur (p=0.2, sigma=0.1–1.0), Gaussian noise (p=0.2, std=0–0.03)
- **Color**: Grayscale (p=0.05)
- Always applied consistently to (face crop, mask) pairs to preserve mask alignment

---

## 8. Service Layer — FastAPI

`service/api.py` — 534-line Production REST API:

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard (index.html) |
| `POST` | `/api/v1/analyze` | Synchronous analysis (small videos, immediate response) |
| `POST` | `/api/v1/analyze/async` | Async job queue (large videos, poll for result) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll async job status: queued/processing/complete/failed |
| `GET` | `/api/v1/mask/{job_id}` | Retrieve 56×56 localization mask as PNG |
| `GET` | `/api/v1/health` | System health: GPU name, VRAM, queue depth, model loaded |
| `GET` | `/api/docs` | Swagger UI auto-documentation |
| `GET` | `/api/redoc` | ReDoc documentation |

CORS is enabled for all origins to support multi-device access (browser on different machine).

### FORGEInference Engine (`service/inference.py`) — 879 lines

The full end-to-end inference pipeline:
1. **SHA-256 content fingerprint** computation at entry (audit trail)
2. **C2PA provenance verification** (completely separate from fake_score)
3. **Hard quality routing**:
   - Composite quality < 0.35 → **conservative path** (skip expensive DINOv2, save ~40% VRAM)
   - Composite quality >= 0.35 → **deep forensic path** (all 7 branches)
4. Video decode → face detect → multi-frame track → affine align → quality analyze
5. Geometry feature computation (16-dim per frame from 5-pt MediaPipe landmarks)
6. rPPG quality gating + signal extraction (6 skin ROIs)
7. Audio extraction + 80-mel Mel-spectrogram computation (when audio stream present)
8. FORGELite forward pass in **fp16** on CUDA
9. Temperature calibration → Decision (real/fake/abstain)
10. **EvidenceReport** construction — typed dataclass, fully JSON-serializable, backward-compatible

---

## 9. Web Frontend

`web/` — Standalone web dashboard automatically served by FastAPI at the root `/`:

- **Dark cyberpunk theme** with glassmorphism panels, animated glow orbs (cyan + purple), cyber-grid background
- **Fonts**: JetBrains Mono + Outfit + Space Grotesk (Google Fonts)
- **Full feature set**:
  - Drag-and-drop video/image upload with preview
  - Real-time analysis with animated progress indicator
  - Verdict badge (FAKE / REAL / REVIEW) with calibrated risk percentage
  - Per-branch evidence attribution bars
  - Suspicious time window markers (temporal timeline)
  - 56×56 localization mask overlay on face region
  - 10-metric quality score breakdown panel
  - One-click exportable forensic HTML report
- **API communication**: Calls `/api/v1/analyze` (sync) and `/api/v1/health` (polling)
- **Asset sizes**: ~14KB HTML, ~18KB CSS, ~16KB JS — fast loading, no JS frameworks

---

## 10. Gradio Interactive Demo

`app/gradio_app.py` — 389-line Gradio interface:

- Independent from the web dashboard — separate tool for interactive exploration
- Upload video or face image from browser
- **Outputs**:
  - Styled HTML verdict badge with decision and calibrated risk
  - Markdown evidence breakdown (branch by branch)
  - Raw JSON branch metrics dict
  - Spatial heatmap overlay image (numpy → Gradio Image component)
  - Exportable forensic report file path
- Configurable thresholds via sliders: uncertainty abstention threshold (default 0.35), evidence coverage minimum (default 0.40)
- CLI entry point: `forge-demo` or `python app/gradio_app.py [--share]`

---

## 11. Evaluation Framework

`evaluation/` — 36 scripts covering every evaluation dimension:

### Main Evaluation Scripts

| Script | Purpose |
|---|---|
| `evaluate_final_system_v3.py` | Full V3 system benchmark — primary eval |
| `evaluate_per_generator_v3.py` | Per-generator AUROC/ACC breakdown |
| `evaluate_cross_domain.py` | Train-on-FF++, test-on-Celeb-DF/DFDC generalization |
| `evaluate_robustness.py` | H.264 codec + Gaussian noise robustness curve |
| `evaluate_calibration.py` | ECE, Brier score, reliability diagram |
| `evaluate_localization.py` | Pixel IoU on documented ground-truth masks |
| `evaluate_av_rppg_contribution.py` | Marginal AUROC gain from AV and rPPG branches |
| `evaluate_multimodal_active.py` | Active branch combination sweep |
| `bootstrap_ci.py` | Bootstrap 95% confidence intervals on all metrics |
| `ablation_e0_e7.py` | Full E0–E7 ablation study (8 ablation conditions) |
| `run_9_phase_ablations.py` | 9-phase ordered ablation study |
| `test_trainsets.py` | Local custom trainset benchmark (CSV-driven) |
| `audit_split_leakage.py` | Detects train/test identity overlap (data hygiene) |
| `uncertainty_v3.py` | Uncertainty decomposition analysis (4 factors) |
| `evaluate_v3_vs_v2.py` | V3 vs V2 comparison (model versioning) |
| `evaluate_v2_router_vs_v1_baseline.py` | Router vs baseline comparison |
| `diagnostic_export.py` | Full diagnostic export (26KB of metrics) |

### Primary Metrics Tracked

- **AUROC** (macro, per-generator, cross-domain — primary metric)
- Accuracy, Precision, Recall, F1 (at operating threshold)
- **Expected Calibration Error (ECE)** and Brier Score
- **Abstention Rate** + Coverage-Risk Curve
- **Pixel IoU** (manipulation localization, on ground-truth-masked samples only)
- Bootstrap 95% CI on all metrics
- Marginal AUROC gain per branch (ablation)
- **Cross-domain generalization gap** (in-domain AUROC vs out-of-domain AUROC)

---

## 12. Configuration System

All hyperparameters centralized in `configs/` as YAML files, loaded via `omegaconf`. No magic numbers in source code.

### `configs/model.yaml` — Full Branch Hyperparameters (157 lines)

Key sections:
```yaml
semantic_branch:
  backbone: "dinov2_vits14"       # HuggingFace: facebook/dinov2-small
  adaptation_mode: "layernorm"    # frozen_linear | layernorm | lora | last_block
  output_dim: 384                 # CLS embedding dim

forensic_branch:
  srm_num_filters: 30
  in_channels: 35                 # 1 lum + 30 SRM + 3 Haar + 1 noise
  num_blocks: 4
  output_dim: 256

temporal_branch:
  hidden_dim: 256
  num_layers: 2
  aggregation: {use_mean: true, use_top_k_percent: 0.10, use_max: true,
                use_dispersion: true, use_attention: true}

rppg_branch:
  num_rois: 6
  bpm_min: 42  |  bpm_max: 210
  min_duration_s: 8.0  # quality gate

av_branch:
  mel_n_mels: 80
  lip_crop: 48×96
  window_frames: 5

calibration:
  mc_dropout_samples: 10
  abstention:
    min_evidence_coverage: 0.3
    max_uncertainty: 0.35
    max_branch_disagreement: 0.4
    min_face_quality: 0.2
    score_margin: 0.15
```

### `configs/train.yaml` — Training Hyperparameters (159 lines)

Key settings: seed=42 (deterministic), fp16, cosine LR, early_stopping patience=3, checkpoint top-3 by `val/cross_domain_auroc`, WandB project=`forge-lite`, smoke_test mode for fast setup validation.

### `configs/data.yaml` — Dataset & Preprocessing (127 lines)

Dataset roots, face detection settings (RetinaFace, min_face=80px, confidence=0.9), frame sampling (8 FPS target, max 300 frames), temporal windows (size=16, stride=8), quality thresholds for rPPG gating, codec bank definitions.

### `configs/frozen_v3_policy.json`

Freezes all V3 model weights and evaluation thresholds post-calibration. Ensures no test-set leakage — thresholds are locked and cannot be re-tuned after any test-set access.

---

## 13. Key Innovations & Design Decisions

### 1. "Evidence, Not Score" Philosophy

The system does not output a bare probability. It outputs a structured report: which branches were active, their directional contributions (toward real or fake), why abstention was triggered (if applicable), and the calibrated uncertainty. This makes analysis forensically defensible and auditable.

### 2. Quality-Gated Abstention Instead of Forced Verdict

Standard deepfake detectors always output a number. FORGE-Lite outputs `insufficient_evidence` when face quality, uncertainty, or branch coverage fall below thresholds — refusing to make a claim when the evidence is weak. This is critical for real-world forensic use cases.

### 3. Codec Consistency Training (Core Innovation)

Most deepfake detectors fail catastrophically on H.264-compressed social media videos because their frequency artifacts get confused with compression artifacts. FORGE-Lite's solution:
- Offline codec bank creates 6 compression variants of every training video
- `CodecConsistencyLoss` (weight 0.3) penalizes different scores for the same video at different compression levels
- Model explicitly learns: compression level != fakeness signal

### 4. DINOv2 Adaptation — Not Full Fine-Tuning

Full fine-tuning of DINOv2 on deepfake data risks catastrophic forgetting of generalizable features and causes in-domain overfitting. The 4-mode adaptation system allows the minimum necessary modification. Mode selection is done by cross-domain AUROC, never in-domain accuracy alone.

### 5. Strict Provenance Separation

C2PA credential verification is architecturally separated from `fake_score`. These are two independent outputs. Key rules:
- A valid C2PA credential does NOT make the video real (credential may describe AI-generated media)
- A missing credential does NOT make it fake (most real videos have no credentials)
- Never mix `validation_state` into `fake_score`

### 6. Hard Quality Routing (Conservative vs Deep Forensic Path)

If composite quality < 0.35, the expensive DINOv2 forward pass is skipped entirely, saving ~40% VRAM and inference latency. This is critical for real-world deployment where many uploaded videos are low quality. The system still gives a result via the simpler forensic path.

### 7. Frozen V3 Policy

After final system calibration, all thresholds and weights are frozen via `frozen_v3_policy.json`. The forensic explainer is architecturally read-only — it can observe and explain V3's decisions but is structurally prevented from modifying them.

### 8. Source-Pair Contrastive Learning

Training pairs (real, fake) are sampled from the **same source video** where possible. This forces the model to learn what manipulation *adds* rather than learning identity-specific or scene-specific features that would cause identity bias.

### 9. rPPG as Auxiliary Only

The heartbeat signal (rPPG) is never treated as an oracle. It is gated by strict quality conditions (8+ seconds, sufficient face area, low motion), and its absence is always logged as "unavailable" not "suspicious." This prevents false positives on short clips or high-motion content.

---

## 14. Dependencies at a Glance

| Category | Key Libraries | Min Version |
|---|---|---|
| **Core DL** | PyTorch, torchvision, torchaudio | >= 2.3.0 |
| **Foundation Model** | timm, transformers | >= 1.0.3, >= 4.40 |
| **Face Detection** | facenet-pytorch, retinaface-pytorch | >= 2.5.3 |
| **Landmark & Physiology** | mediapipe | >= 0.10.14 |
| **Signal Processing** | PyWavelets, scipy, scikit-image | >= 1.6, >= 1.13, >= 0.23 |
| **Audio** | librosa | >= 0.10.2 |
| **Computer Vision** | opencv-python, Pillow | >= 4.10, >= 10.3 |
| **Data Storage** | lmdb, webdataset | >= 1.4.1, >= 0.2.86 |
| **ML Utilities** | numpy, pandas, scikit-learn | >= 1.26, >= 2.2, >= 1.5 |
| **Experiment Tracking** | wandb | >= 0.17.0 |
| **Explainability** | captum | >= 0.7.0 |
| **API** | FastAPI, uvicorn, python-multipart | >= 0.111, >= 0.30, >= 0.0.9 |
| **Demo** | gradio | >= 4.36.0 |
| **Visualization** | plotly, matplotlib, seaborn | >= 5.22, >= 3.9, >= 0.13 |
| **Config** | omegaconf, pydantic, pydantic-settings | >= 2.3, >= 2.7, >= 2.3 |
| **Utilities** | tqdm, rich, click, httpx | >= 4.66, >= 13.7, >= 8.1, >= 0.27 |
| **Optional: LoRA** | peft | >= 0.11.0 |
| **Optional: Adversarial** | torchattacks | >= 3.5.1 |
| **Dev** | pytest, ruff, mypy, pre-commit | >= 8.2, >= 0.4, >= 1.10, >= 3.7 |

---

## 15. Current State of the Project

### Fully Implemented and Working

| Component | File | Lines | Status |
|---|---|---|---|
| FORGELite (top-level model) | `models/forge_lite.py` | 569 | Done |
| DINOSemanticBranch | `models/dino_branch.py` | 253 | Done |
| ForensicResidualBranch | `models/forensic_branch.py` | 279 | Done |
| TemporalEncoder | `models/temporal_branch.py` | ~300 | Done |
| RPPGBranch | `models/rppg_branch.py` | 398 | Done |
| AudioVisualBranch | `models/av_branch.py` | 332 | Done |
| GeometryFeatureExtractor | `models/geometry_branch.py` | 218 | Done |
| EvidenceAwareModalityRouter | `models/fusion.py` | 379 | Done |
| LocalizationHead | `models/localization.py` | 212 | Done |
| CalibrationLayer | `models/calibration.py` | 331 | Done |
| FastAPI service (all endpoints) | `service/api.py` | 534 | Done |
| FORGEInference engine | `service/inference.py` | 879 | Done |
| ForensicExplainer | `service/forensic_explainer.py` | 1196 | Done |
| ProvenanceVerifier (C2PA) | `service/provenance.py` | 588 | Done |
| EvidenceReport | `service/evidence_report.py` | ~500 | Done |
| ForgeDataset | `data/dataset.py` | ~670 | Done |
| CodecBank | `data/codec_bank.py` | 154 | Done |
| Augmentation pipeline | `data/augmentation.py` | 197 | Done |
| Full preprocessing pipeline | `preprocessing/preprocess_all.py` | ~600 | Done |
| 5-phase training orchestrator | `training/run_training.py` | 283 | Done |
| Composite losses | `training/losses.py` | 326 | Done |
| 36 evaluation scripts | `evaluation/` | ~300K total | Done |
| Web dashboard | `web/` | ~50KB | Done |
| Gradio demo | `app/gradio_app.py` | 389 | Done |
| 5 YAML configs | `configs/` | ~35KB total | Done |
| CLI entry points | `pyproject.toml` | 118 | Done |

### In Progress / Partially Complete

- **Actual model training**: Architecture is complete but no trained `best.ckpt` exists. Requires dataset download + multi-day GPU training run.
- **Real dataset download**: `data/download_datasets.py` and `data/download_av_datasets.py` exist but public datasets must be requested and downloaded manually.
- **LoRA adaptation**: Code complete, requires `pip install peft` separately (optional dependency).
- **rPPG in production**: Logic fully implemented but quality gating requires real long-form videos (>=8 seconds of continuous face track) to activate.
- **Adversarial hardening**: Training script exists but PGD training adds ~2× training time; run after base training is complete.

### Backlog (from improve.md and newimp.md)

- DINOv3 backbone upgrade when publicly available
- Video-level ensemble over multiple temporal windows
- TalkingHeadBench evaluation
- Per-generator fine-tuning for newest generators (Stable Video Diffusion, CogVideoX, etc.)
- ONNX/TorchScript export for edge deployment
- Camera fingerprinting integration (PRNU-based)
- Streaming inference API (WebSocket)
- Batch inference CLI for dataset-scale processing

---

## 16. How to Run — Step by Step

### Install

```bash
cd c:\deepfake
pip install -e ".[dev]"

# Optional extras:
pip install -e ".[lora]"         # LoRA adaptation support
pip install -e ".[adversarial]"  # Adversarial training
```

### Verify Setup (Smoke Test — No Dataset Needed)

```bash
# Runs 5 batches, 1 epoch with synthetic data — verifies everything imports and runs
python -m training.run_training --smoke-test
```

### Preprocess Datasets (After Dataset Download)

```bash
# After downloading raw datasets to data/raw/
python -m preprocessing.build_manifest --config configs/data.yaml

# Build codec bank (runs once, takes several hours)
python -m data.codec_bank --config configs/data.yaml
```

### Train

```bash
# Full training (all 5 phases sequentially)
python -m training.run_training --config configs/train.yaml

# Resume from checkpoint
python -m training.run_training --resume-from outputs/checkpoints/last.ckpt

# Single phase
python -m training.run_training --phase 4 --config configs/train.yaml
```

### Evaluate

```bash
# Main system evaluation
python evaluation/evaluate_final_system_v3.py

# Per-generator breakdown
python evaluation/evaluate_per_generator_v3.py

# Custom trainset benchmark (local files, no full dataset needed)
python evaluation/test_trainsets.py --type video --limit 50

# Full ablation study
python evaluation/ablation_e0_e7.py
```

### Serve API

```bash
# Via CLI entry point
forge-serve

# Or directly
uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload

# Dashboard: http://localhost:8000
# API docs:  http://localhost:8000/api/docs
```

### Gradio Demo

```bash
forge-demo
# Or with public link:
python app/gradio_app.py --share
```

### CLI Entry Points (pyproject.toml)

| Command | Module | Description |
|---|---|---|
| `forge-preprocess` | `preprocessing.build_manifest:main` | Build dataset manifests |
| `forge-train` | `training.trainer:main` | Train model |
| `forge-eval` | `evaluation.evaluate_video:main` | Evaluate on video |
| `forge-serve` | `service.api:main` | Start API server |
| `forge-demo` | `app.gradio_app:main` | Launch Gradio demo |

---

## 17. Output Schema

Every analysis returns a structured `AnalysisResult` Pydantic model (`service/schemas.py`):

```json
{
  "decision": "likely_fake | likely_real | insufficient_evidence",
  "fake_score": 0.87,
  "calibrated_risk": 0.83,
  "uncertainty": 0.12,
  "evidence_coverage": 0.80,
  "quality": {
    "face_visibility": 0.92,
    "track_continuity": 0.88,
    "blur": 0.75,
    "pose": 0.15,
    "occlusion": 0.05,
    "compression": 0.60,
    "audio_available": true,
    "rppg_available": false,
    "rppg_quality": 0.0
  },
  "active_evidence": ["semantic", "forensic", "temporal", "av_sync", "geometry"],
  "unavailable_evidence": ["rppg"],
  "suspicious_windows": [
    {"start_seconds": 1.2, "end_seconds": 2.4, "score": 0.91}
  ],
  "localization_available": true,
  "abstention_reason": null,
  "model_version": "forge-lite-1.0",
  "preprocessing_version": "prep-1.0",
  "processing_time_seconds": 2.34
}
```

### Health Endpoint Response (`/api/v1/health`)

```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4060",
  "vram_total_gb": 8.0,
  "vram_free_gb": 5.2,
  "queue_depth": 0,
  "model_loaded": true,
  "model_version": "forge-lite-1.0"
}
```

---

## 18. Forensic Explainability Layer

`service/forensic_explainer.py` — 1196 lines. One of the most unique and detailed components of the project.

### What It Provides

1. **Directional Evidence** per branch:
   ```
   C_i = w_i × (p_i - 0.5)
   ```
   Positive = evidence for fake, Negative = evidence for real. Also computes decision margin: `2 × |p_i - 0.5|`

2. **Counterfactual Branch Effect**:
   ```
   Δz_i = z_full - z_without_branch_i
   ```
   Measures how much the final logit changes when each branch is removed.

3. **4-Factor Uncertainty Decomposition**:
   - **Predictive Entropy**: MC-Dropout variance across 10 forward passes
   - **Decision Margin**: `|score - 0.5|` (how close to decision boundary)
   - **Quality Defect**: Combined face + video quality degradation signal
   - **Branch Disagreement**: Variance of individual branch predictions

4. **Video Track Aggregation**:
   Per-frame → per-window → per-track using robust medians, 75th percentiles, and peak anomaly interval detection.

5. **Neutral, Accurate Phrasing**:
   Quality degradation triggers "Evidence reliability reduced by [factor]" — NEVER "Evidence of manipulation detected due to quality issues."

6. **Provable Checkpoint Fingerprint**:
   SHA-256: `904E8A91391A337B5BEBD1B95EAFD45A8F5F819F2FA024067B68CF96B3BE5B03`

### Critical Architectural Mandate

> *"The explanation system may observe and explain V3, but it must never modify the V3 decision."*

The ForensicExplainer is strictly a **read-only observer**. It will never change `fake_score`, `calibrated_risk`, or `decision`. This separation is enforced architecturally by not passing mutable state to the explainer.

---

## 19. Provenance & C2PA

`service/provenance.py` — 588 lines. Implements **FORGE Verify** — completely separate from deepfake detection.

### What It Does

Verifies whether media was captured through a trusted path using C2PA v2.4 Content Credentials.

**Output schema** (per `newimp.md §7`):
```
validation_state: valid | invalid | missing | unsupported | unavailable
signer_trust:     trusted | untrusted | unknown
applicable_assertions: list[str]  # what the credential claims
```

### 3 Operating Modes

1. **Arbitrary uploaded / internet video**:
   Check credentials if present; report with explicit uncertainty. NEVER mix `validation_state` into `fake_score`. Evidence is forensic, not credential-based.

2. **Video captured through your own application**:
   - Server issues a fresh, short-lived, session-bound challenge
   - Approved capture software binds the session challenge and media to a verifiable capture record
   - Verify: freshness, media binding, signer/device trust, revocation status
   - Add liveness/PAD and anti-injection checks appropriate to threat model

3. **High-stakes transaction** (identity verification, legal, financial):
   NEVER let a face fake-score be the sole authorization control. Must combine authenticated channels, transaction-specific approval, and independent confirmation.

### Core Rule (Repeated in Source)

> *"A valid credential may explicitly describe AI-generated media. Missing credentials do NOT make a genuine video fake. Invalid credentials mean provenance validation FAILED — not necessarily that the facial content was generated."*

**References**: C2PA v2.4, NIST SP 800-63A-4, NIST FATE Part 10.

---

## 20. Known Risks & Guardrails

| Risk | What Can Go Wrong | Guardrail in Place |
|---|---|---|
| **Spectral bias** | SRM/FFT learns codec-specific artifacts, not forgeries | CodecConsistencyLoss (0.3 weight) + offline codec bank |
| **In-domain overfitting** | Model learns identity/scene features of FF++ | DINOv2 frozen backbone + cross-domain AUROC as primary metric |
| **rPPG as oracle** | "No heartbeat = fake" (wrong) | Strict quality gating; unavailable != suspicious |
| **Threshold leakage** | Tuning decision thresholds on test set inflates reported accuracy | Thresholds frozen ONLY on separate calibration set, never test set |
| **Provenance confusion** | Mixing credential status with fake probability | C2PA and fake_score are completely separate outputs, architecturally |
| **High-stakes abuse** | Using the system as sole decision for identity/legal | `insufficient_evidence` abstention + explicit uncertainty quantification |
| **VRAM OOM** | 8 GB GPU overflow during inference | Sequential branch processing + fp16 + conservative path routing |
| **Windows DataLoader crash** | Shared memory page-file limit (Error 1455) | `file_system` sharing strategy set before any DataLoader creation |
| **Adversarial attacks** | Pixel-level perturbations that fool the model | PGD-5 adversarial training (ε=0.03 L-inf), tested with PGD-20 |
| **Class imbalance** | More real than fake (or vice versa) biases learning | FocalLoss (alpha=0.25, gamma=2.0) |
| **Localization metric inflation** | Reporting IoU on attention maps (not pixel masks) | Only pixel IoU on documented ground-truth masks is reported |
| **Generator-specific overfitting** | Good on FF++ generators, bad on newer ones | Cross-domain eval on held-out datasets, LoRA generative fine-tune |

---

## Summary

FORGE-Lite is a research-grade, production-aware deepfake video detection system that uniquely combines:

- A **DINOv2 foundation model** (ViT-S/14, self-supervised) for semantic face understanding
- **Classical forensic signals** — SRM pixel residuals (30 filters), Haar wavelets, FFT radial spectra
- **Physiological signals** — rPPG heartbeat detection from skin micro-color changes (6 ROIs, quality-gated)
- **Audio-visual desync detection** — SyncNet-style lip-audio contrastive encoder
- **Facial geometry kinematics** — 16-dim landmark descriptors with temporal derivatives, scale-invariant
- **Quality-aware abstention** — "insufficient_evidence" output instead of always forcing a verdict
- **Full explainability** — directional evidence contributions, counterfactual branch effects, 4-factor uncertainty decomposition
- **Provenance verification** — C2PA credential checking, strictly separated from fake_score
- All running on a **consumer RTX 4060 8 GB GPU** via sequential branch processing, fp16, and quality routing

**Architecture is 100% complete.** All 7 branches, fusion head, localization, calibration, inference engine, FastAPI service, web frontend, Gradio demo, evaluation framework (36 scripts), and training pipeline (5 phases + adversarial + calibration) are fully implemented.

**Remaining work**: Dataset acquisition (FF++, Celeb-DF-v2, DFDC, WildDeepfake) and running the multi-day training pipeline.

---

*Last updated: September 2026 | Model Version: forge-lite-1.0 | Target: CVPR 2027*
*Total codebase: ~40 source files, ~20K+ lines of Python*
