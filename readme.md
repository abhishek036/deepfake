This is an excellent, rigorous research document. Here is the complete final PRD, cleaned and structured:

---

# realitycheck.fr: Final Product Requirements Document
### Forensic, Robust, Generalizable Evidence Detection for Facial Deepfakes & Synthetic Media
**Version 1.0 | AIML Cornerstone Project 2026 | RTX 4060 8GB Optimized**

---

## Document Metadata

| Field | Detail |
|---|---|
| Project Name | realitycheck.fr (formerly FORGE-Lite) |
| Full Name | Forensic, Robust, Generalizable Evidence Detection for Facial Deepfakes & Synthetic Media |
| Project Code | AIML-CORNER-2026-FORGE |
| Version | 1.0 — Final |
| Hardware Target | RTX 4060 8GB VRAM · 16GB RAM · 512GB SSD |
| Status | APPROVED FOR DEVELOPMENT |
| Target Venue | CVPR 2027 Workshop on Media Forensics / IEEE TIFS Journal |

---

## Official Research Claim

> FORGE-Lite is a resource-efficient face deepfake video detector that combines foundation-model visual features, low-level forensic residuals, temporal dynamics, and quality-aware optional physiological evidence. It is evaluated under unseen manipulation methods, source-disjoint splits, codec transformations, and selective prediction.

### Executive Engineering Synthesis

> **The system combines semantic, forensic, temporal, and reliability-aware evidence, and explicitly abstains when evidence is insufficient. Evaluation identified significant sensitivity to JPEG compression and demonstrated that branch disagreement is not itself a monotonic confidence indicator.**

### Validated Headline Benchmarks (Source-Disjoint 50/50 Balanced Test)

| Metric | Score | Validation Standard |
|---|---|---|
| **AUROC** | **0.8297** | Source-Disjoint Clean Split (Zero Leaked Identities) |
| **AUPRC** | **0.7988** | Balanced 50% Real / 50% Fake |
| **Balanced Accuracy** | **74.88%** | Equal Class Weighting ($N_{\text{Real}} = N_{\text{Fake}}$) |
| **F1-Score** | **0.7500** | Balanced Harmonic Mean |
| **Matthews Corr (MCC)** | **0.4975** | True Non-Trivial Binary Correlation |
| **Platt Calibrated ECE** | **0.0654** | Held-Out Calibration Partition ($ECE < 0.10$ Target Met) |
| **Brier Score** | **0.1694** | Sub-0.20 Probabilistic Error |
| **Selective Accuracy (@ 15% Abstention)** | **77.93%** | Composite Uncertainty Gating |

---

## What This System Will NOT Claim

These statements are scientifically invalid and must never appear in the paper, demo, or README:

- Reverse-engineers the original generator
- Cannot be fooled by adversarial attacks
- Absence of heartbeat proves synthetic content
- Works on all future video generators
- Produces legally valid proof of forgery
- World-best accuracy before locked cross-domain evaluation

---

---

# SECTION 01 — Problem Statement

---

## 1. Problem Statement

### 1.1 Why This Problem Matters

Deepfake video technology — powered by GANs and diffusion models — has reached a level of realism where human observers can no longer reliably distinguish synthetic from authentic media. Current benchmarks cover dozens of modern generation models: AIGVDBench reports 31 generation models and over 440,000 videos; FVBench reports 42 synthesis and editing models across 120,000+ videos.

The field has moved decisively toward foundation-model adaptation, parameter-efficient tuning, broader generator coverage, codec robustness, and strict cross-dataset evaluation. Any system built in 2026 must be designed with this context.

### 1.2 Why Existing Systems Fail

| Failure Mode | Root Cause | Consequence |
|---|---|---|
| Single-frame CNN only | No temporal reasoning | Misses inter-frame motion anomalies |
| Dataset bias | Trained only on FaceForensics++ | AUC drops from 95% → 60% on unseen generators |
| GAN-only training | Zero diffusion-model coverage | Completely blind to Stable Diffusion fakes |
| Discriminative boundary only | Learns statistical pattern, not physics | Single adversarial perturbation flips output |
| rPPG used as oracle | Treats signal absence as proof | High false positive on low-quality real video |
| No explainability | Binary output only | Unusable for forensic or legal purposes |
| Frequency overfitting | Spectral shortcuts to compression artifacts | Collapses on re-encoded video |

### 1.3 The Correct Objective

No honest researcher can guarantee a system will be world-best before it survives a locked, cross-domain evaluation. The correct objective is:

> Build the strongest credible, reproducible, resource-efficient system and make "world-best" an empirical result, not a marketing claim.

### 1.4 Scope

**In scope:**
- Face swapping (DeepFaceLab, SimSwap, FaceSwap)
- Face reenactment and talking-head synthesis
- Partial facial manipulation
- Synthetic facial identity replacement
- AI-generated facial video
- Videos with one dominant tracked face

**Out of scope — v1.0:**
- Audio-only deepfakes (separate system)
- Full-body or background manipulation
- Non-facial scene generation
- Real-time streaming (planned v2.0)
- Multi-face tracking at scale (planned v2.1)
- Legal provenance verification — never in scope

---

---

# SECTION 02 — Goals and Success Metrics

---

## 2. Goals and Success Metrics

### 2.1 Primary Goals

**G1 — Cross-domain generalization:** Improve cross-domain macro-AUROC by at least 3 percentage points over the strongest baseline. No more than 1 percentage point in-domain degradation.

**G2 — Codec robustness:** Compression causes less than 10% relative degradation at the operating point. The model must learn compression is not equivalent to forgery.

**G3 — Calibrated selective prediction:** ECE below 0.05. The system returns INSUFFICIENT_EVIDENCE rather than a forced binary label when quality or evidence is inadequate.

**G4 — Spatial localization:** Produce a learned 56×56 manipulation probability mask. Report IoU only on samples with valid ground-truth pixel masks.

**G5 — Resource efficiency:** Full pipeline runs on RTX 4060 8GB. Peak VRAM measured and reported. P50 and P95 latency reported.

**G6 — Publication readiness:** Ablations, locked evaluation, failure cases, and calibration curves all present. Credible workshop paper submission.

### 2.2 Quantitative Targets

| Metric | Target |
|---|---|
| FaceForensics++ C23 Video AUC | ≥ 0.96 |
| FaceForensics++ C40 Video AUC | ≥ 0.90 |
| Celeb-DF v2 Video AUC (cross-domain) | ≥ 0.88 |
| DFDC Preview AUC (cross-domain) | ≥ 0.82 |
| Cross-domain macro-AUROC | ≥ +3pp over best baseline |
| TPR @ 1% FPR | ≥ 0.75 |
| ECE | ≤ 0.05 |
| Localization IoU (valid masks only) | ≥ 0.55 |
| Compression degradation | ≤ 10% relative |
| Peak VRAM — inference | ≤ 7.5GB |
| P50 inference latency (30-sec clip) | ≤ 60 seconds |

### 2.3 Non-Goals

- Real-time streaming (v2.0)
- Mobile on-device inference (v2.1)
- Detecting every type of body or background manipulation
- Identifying the exact generator without a separate attribution experiment
- Guaranteeing detection of future generators

---

---

# SECTION 03 — System Architecture

---

## 3. System Architecture

### 3.1 Core Design Principle

This system does NOT ask: "does this look fake?"

It asks: "what evidence exists across semantic, forensic, temporal, and physiological dimensions — and how reliable is each piece of evidence given the quality of this input?"

This is fundamentally different from discriminative boundary learning. The result is a system that degrades gracefully on hard inputs rather than producing confidently wrong answers.

### 3.2 Architecture Pipeline

```
Input video
    |
    v
Video metadata + quality analysis
    |
    v
Face detection + tracking (MTCNN / RetinaFace)
    |
    v
Aligned face windows (224x224, 16 frames per window)
    |
    +─────────────────────────┬─────────────────────────+
    |                         |                         |
    v                         v                         v
RGB Semantic Branch     Forensic Residual Branch   Context Branch
DINOv2 ViT-S/14         Lightweight CNN (1-6M)     (Optional)
    |                         |                         |
    |                 SRM filters                       |
    |                 Haar wavelets                     |
    |                 HF residuals                      |
    |                 Compact FFT stats                 |
    |                         |                         |
    +─────────────────────────+─────────────────────────+
                              |
                              v
                    Temporal Encoder
                    2-layer GRU / TemporalConv
                    (per-frame DINO + forensic features,
                     aligned frame difference,
                     landmark velocity,
                     face quality sequence)
                              |
                +─────────────+──────────────+
                |                            |
                v                            v
         rPPG Branch                Audio-Visual Branch
         (quality-gated)            (availability-gated)
         SNR, frequency,            Lip-sync score
         coherence, stability       (only when audio + face present)
                |                            |
                +─────────────+──────────────+
                              |
                              v
               Reliability-Aware Fusion Head
               (2-layer MLP + quality gates +
                missing-modality masks +
                branch disagreement)
                              |
                              v
               Calibration + Abstention Layer
                              |
               ┌──────────────┼──────────────┐
               v              v              v
          likely_real    likely_fake   insufficient_evidence
```

---

### 3.3 Branch Definitions

---

#### Branch 1 — RGB Semantic Branch (DINOv2 ViT-S/14)

**Purpose:** Extract high-level semantic visual features from face crops using a foundation vision model.

**Backbone:** DINOv2 ViT-S/14 (~22M parameters, Meta AI `facebookresearch/dinov2`). Selected for proven pretrained weights, open reproducibility, and RTX 4060 VRAM efficiency. (Note: Early design documents explored DINOv3, but DINOv2 ViT-S/14 is the locked, active implementation).

**Input:** 224×224 aligned RGB face crop.

**Extracted features:**
- CLS token embedding
- Patch-token features (for localization head)
- Patch-token mean and variance
- Optional context-crop feature

**Adaptation strategy — test in this order, select by cross-domain macro-AUROC:**

| Config | Description |
|---|---|
| A | Frozen backbone + linear classifier |
| B | Frozen backbone + trainable LayerNorm only |
| C | Frozen backbone + low-rank adapters (LoRA) |
| D | Last-block fine-tuning only |

Default candidate: **LayerNorm-only adaptation (Config B).** Recent WACV 2026 work shows parameter-efficient foundation adaptation with source-matched pairs outperforms more complex systems on cross-domain generalization.

**Selection rule:** Choose the version with the best cross-domain macro-AUROC, subject to acceptable VRAM. Never select by FaceForensics++ in-domain score alone.

> **License note:** DINOv3 uses a custom Facebook Research license with redistribution and publication requirements. Review before releasing publicly. If it prevents intended release, use DINOv2 ViT-S/14 instead.

---

#### Branch 2 — Forensic Residual Branch (Lightweight CNN)

**Purpose:** Capture low-level pixel-domain artifacts that semantic models cannot see — compression traces, generator upsampling patterns, noise residuals.

**Architecture:** Small custom CNN, 1–6M trainable parameters.

**Input channels:**
- Luminance channel
- SRM high-pass residuals (30 fixed filters)
- Haar LH, HL, HH subbands
- Local noise / compression estimate

**Structure:**
- Fixed SRM/Haar preprocessing (not trained)
- Depthwise-separable convolution blocks
- BatchNorm or GroupNorm
- GELU / SiLU activations
- Global average pooling
- 256-dimensional forensic embedding

**Compact FFT feature vector (auxiliary, not a full classifier):**
- Radial frequency energy bins
- High-frequency energy ratio
- Low-to-high frequency ratio
- Spectral flatness
- Local spectral variance

**Critical warning:** Frequency features cause generalization failure when the model overfits to generator-specific spectral patterns. FreqDebias (CVPR 2025) identifies spectral bias as a key generalization problem. Use consistency training across codec variations to prevent this. The forensic branch is retained only if ablations show measurable cross-domain improvement beyond the semantic branch.

---

#### Branch 3 — Temporal Encoder

**Purpose:** Detect inter-frame motion inconsistencies — synthetic faces generated frame-independently without modeling full-body kinematics exhibit subtle, consistent motion anomalies.

**Do NOT use:** RAFT optical flow + ViT-B/16. Too heavy for 8GB VRAM and not justified by accuracy improvement.

**Use instead:**
- 8–16 aligned face frames per temporal window
- Shared spatial encoders (semantic + forensic)
- 2-layer GRU or temporal convolution network
- Hidden dimension: 128–256

**Inputs to temporal encoder:**
- Per-frame DINO feature
- Per-frame forensic feature
- Aligned frame difference (not raw — aligned to remove camera motion)
- Normalized landmark velocity
- Face quality values per frame
- Track continuity score

**Outputs:**
- Window-level fake score
- Frame-level suspiciousness score
- Temporal embedding
- Temporal quality score

**Video-level aggregation — not a naive mean:**
- Mean window score
- Top-10% window score
- Maximum window score
- Temporal attention weighted score
- Score dispersion

A fake may appear in only part of a video. Naive mean suppresses real manipulation windows.

---

#### Branch 4 — rPPG Branch (Quality-Gated, Optional)

**Purpose:** Measure whether a plausible physiological heartbeat signal is present in the face region. Absence or spatial incoherence of rPPG is a supporting signal, not proof of fakeness.

**Critical rule:** rPPG is an auxiliary signal, not an authenticity oracle. Synthetic face videos CAN be produced with physiologically realistic signals (demonstrated in CVPR 2022). Never output "no rPPG detected, therefore fake."

**Valid rPPG outputs:**
- rPPG available — quality sufficient, signal plausible
- rPPG unavailable — insufficient duration or quality
- rPPG low quality — signal present but SNR below threshold
- rPPG spatially inconsistent — cross-ROI coherence failed

**Extracted features per stable face track:**
- Signal-to-noise ratio (SNR)
- Dominant frequency (BPM range check: 42–210 BPM)
- Band power (0.7–3.5 Hz)
- Cross-ROI coherence (6 ROIs: forehead, cheeks, nose bridge, chin, periorbital)
- Temporal stability
- Illumination variation
- Motion contamination estimate
- Face visibility score

**Minimum requirements for rPPG to be used:**
- ≥8–12 seconds of stable face track
- Face area ratio ≥ threshold
- Track continuity ≥ threshold
- Motion contamination ≤ threshold

If any condition fails: rppg_available = false. The fusion head receives the availability mask and must learn what to do without rPPG.

**Add rPPG only after the core model works.** Run this ablation:

| Config | Notes |
|---|---|
| FORGE without rPPG | Baseline |
| FORGE with rPPG, no quality gate | Shows why gating matters |
| FORGE with quality-gated rPPG | Target |

Keep rPPG only if the quality-gated version improves cross-domain performance without unacceptable false positives on real low-quality videos.

---

#### Branch 5 — Audio-Visual Branch (Availability-Gated, Optional)

**Purpose:** Detect lip-speech desynchronization in face-swap videos where the face generator was applied without access to the audio track.

**This is NOT part of the primary video-only claim.** It is a separately gated optional module.

**Required before this branch activates:**
- audio_available = true
- speech_detected = true
- face_visible = true
- mouth_visible = true
- synchrony_quality ≥ threshold

If any condition is false: audio_visual_available = false. The fusion head must learn to ignore this branch when unavailable.

**Do not include Celery + Redis in v1.0.** Use a local background task queue. Distributed infrastructure is for v2.0 with multiple GPU workers.

---

#### Branch 6 — Optional Context Branch

**Purpose:** Handle fully synthetic videos where the manipulation is not confined to the face region.

**Input:** Letterboxed full-frame thumbnail processed through DINOv3 with low-resolution temporal aggregation.

**Use primarily for:**
- Fully synthetic videos (Track B evaluation)
- Global scene inconsistencies
- Face-not-detected cases

**Critical constraint:** Do not allow context features to dominate face-swap detection. Backgrounds and identities become dataset shortcuts. Context evidence must be gated and down-weighted in face-localized manipulation cases.

---

### 3.4 Reliability-Aware Fusion Head

**Replace XGBoost post-processing with a jointly trained fusion MLP.**

**Inputs:**
- Semantic embedding (DINO branch)
- Forensic embedding (residual branch)
- Temporal embedding + aggregated temporal scores
- rPPG features + availability mask + quality score
- Audio-visual sync score + availability mask
- Face quality vector (blur, pose, occlusion, compression, area)
- Video quality vector
- Branch availability masks
- Branch disagreement score

**Architecture:**
- 2-layer MLP
- Feature normalization
- Reliability gate (softmax over branch quality scores)
- Missing-modality mask
- Dropout
- LayerNorm

**Reliability gate concept:**

```python
branch_weight = softmax(
    quality_encoder(face_quality, branch_quality, availability)
)
```

This automatically reduces reliance on rPPG when the face is blurred and reduces reliance on audio-visual evidence when audio is absent. The model learns these weights from data — not from hand-crafted rules.

---

### 3.5 Localization Head

**Purpose:** Produce an interpretable spatial map of the manipulated region.

**Architecture:** Decoder from DINOv3 patch tokens + forensic CNN feature maps + temporal attention.

**Output:** 56×56 manipulation probability map.

**Loss:**
- Binary cross-entropy
- Dice loss
- Optional boundary loss

**Critical reporting rules:**
- Report pixel-level IoU only on samples with valid documented ground-truth masks
- Mark uncertain masks as weak supervision — never mix bounding boxes and pixel masks in one IoU table
- Report supervised mask performance and weak-label performance separately
- Attention rollout and integrated gradients are supporting visualizations only — never reported as IoU metrics

---

---

# SECTION 04 — Final Output Contract

---

## 4. Output Schema

The model must never output only a binary label.

```json
{
  "decision": "likely_fake",
  "fake_score": 0.91,
  "calibrated_risk": 0.87,
  "uncertainty": 0.12,
  "evidence_coverage": 0.84,
  "quality": {
    "face_visibility": 0.94,
    "track_continuity": 0.91,
    "blur": 0.18,
    "pose": 0.13,
    "occlusion": 0.07,
    "compression": 0.42,
    "audio_available": false,
    "rppg_available": true,
    "rppg_quality": 0.66
  },
  "active_evidence": [
    "spatial_residual",
    "temporal_consistency",
    "rppg_spatial_consistency"
  ],
  "unavailable_evidence": [
    "audio_visual_sync"
  ],
  "suspicious_windows": [
    {
      "start_seconds": 4.0,
      "end_seconds": 7.5,
      "score": 0.94
    }
  ],
  "localization_available": true,
  "model_version": "forge-lite-1.0",
  "preprocessing_version": "prep-1.0"
}
```

**Valid decisions:**
- `likely_real`
- `likely_fake`
- `insufficient_evidence`

**Return INSUFFICIENT_EVIDENCE when:**
- Face tracking is unstable
- Face area is too small
- Heavy occlusion or motion blur
- Clip is too short
- Branch disagreement is high
- Input quality is outside training distribution
- rPPG duration or quality is insufficient
- Audio requested but unavailable

---

---

# SECTION 05 — Technical Requirements

---

## 5. Technical Requirements

### 5.1 Tech Stack

| Component | Technology |
|---|---|
| Deep learning framework | PyTorch 2.3 + torchvision |
| Primary backbone | DINOv2 ViT-S/14 (Meta AI `facebookresearch/dinov2`) |
| Forensic preprocessing | SRM filters, PyWavelets (Haar), scikit-image |
| Temporal encoder | 2-layer GRU or TemporalConv (PyTorch) |
| rPPG extraction | MediaPipe Face Mesh + SciPy bandpass + NumPy FFT |
| Audio-visual sync | Custom SyncNet — PyTorch + torchaudio + librosa |
| Localization decoder | Custom head from DINO patch tokens |
| Calibration | Temperature scaling + uncertainty estimation |
| Experiment tracking | Weights & Biases (wandb) |
| Inference API | FastAPI (local queue — no Celery/Redis in v1.0) |
| Demo interface | Gradio 4.x |
| Visualization | Integrated gradients (captum) + Plotly heatmaps |
| Containerization | Docker + Docker Compose |
| Precision | AMP fp16 throughout |
| Gradient accumulation | Enabled for adapter/forensic CNN training |

### 5.2 What NOT to Use in v1.0

These are explicitly excluded from the core pipeline due to VRAM constraints or insufficient justification:

- RAFT optical flow (too heavy, 8GB VRAM insufficient)
- ViT-B/16 temporal encoder (use GRU instead)
- e4e / StyleGAN inversion (scientifically overclaims)
- DDIM inversion (overclaims)
- Large β-VAE (50-step diffusion inversion — not feasible)
- Full SyncNet training in primary pipeline (gated module only)
- Celery + Redis (unnecessary for v1.0 single-machine)

### 5.3 Datasets

| Dataset | Details | Role |
|---|---|---|
| FaceForensics++ C23 | 1,000 real + 4,000 fake videos, 4 methods | Primary training + in-domain eval |
| FaceForensics++ C40 | Same videos, heavy compression | Compression robustness eval |
| Celeb-DF v2 | 590 real + 5,639 high-quality fakes | Cross-domain benchmark |
| DFDC Preview | 1,131 videos (Kaggle, free) | Cross-domain benchmark |
| WildDeepfake | 3,805 real + 3,509 in-the-wild fakes | Zero-shot generalization |
| FFHQ (70K faces) | Real face images (NVIDIA, free) | rPPG baseline, forensic CNN pretraining |
| Custom codec bank | Offline re-encoded versions of training videos | Compression robustness training |
| AIGVDBench / FVBench subset | Modern generator samples (face-centric only) | Modern generator evaluation |

### 5.4 Data Pipeline Rules

**Do NOT decode and store every raw frame.** Store:
- Sampled face crops (224×224, JPEG/WebP)
- Face bounding boxes and landmarks
- Track IDs and continuity scores
- Quality metadata per frame
- Optional pixel masks
- Frame timestamps and video-level metadata

**Use LMDB, WebDataset, or sharded tar files.** Use fp16 cached DINO features where useful.

**Keep raw videos only for:** final evaluation archive, codec re-encoding experiments, debugging, qualitative demos.

### 5.5 Hardware Policy

| Setting | Value |
|---|---|
| Backbone | DINOv2 ViT-S/14 |
| Input resolution | 224×224 face crops |
| Temporal window | 8–16 frames |
| Batch size | 1–8 depending on branch |
| Precision | AMP fp16 |
| Workers | 2–4 maximum initially |
| Training strategy | Sequential branches — not all models resident simultaneously |

---

---

# SECTION 06 — Threat Model

---

## 6. Threat Model

Evaluate separately across all four threat categories. Never pool them.

### Category A — Distribution Shift
- Unseen generator, unseen identity, unseen dataset
- Unseen camera, lighting, compression, resolution

### Category B — Benign Transformations
- H.264 / H.265 CRF recompression
- JPEG recompression
- Resize (down and up), frame-rate conversion, frame dropping
- Gaussian blur, motion blur, brightness, contrast, gamma, color temperature
- Additive and Poisson noise
- Social-media-style processing pipeline

### Category C — Adaptive Digital Attack
- White-box PGD-10 / PGD-20 against the complete fused model
- Transfer attacks from other published detectors
- Temporal perturbation attacks
- Perturbations applied before face extraction (not just to the crop)

### Category D — Failure Conditions (must return INSUFFICIENT_EVIDENCE)
- No face detected
- Multiple ambiguous faces
- Very low resolution
- Severe occlusion
- No speech or audio
- Very short video (< 2 seconds)
- Heavy motion blur

---

---

# SECTION 07 — Build Plan (16 Steps)

---

## 7. Implementation Plan

Follow this order strictly. Do not add complexity before each prior step is validated.

### Step 1 — Freeze the scientific claim
Write the official claim. Review it against the literature. Lock it. Do not let it expand during implementation.

### Step 2 — Build the canonical dataset manifest
Every sample gets a metadata record:

```json
{
  "dataset": "FaceForensics++",
  "video_id": "video_001",
  "source_id": "source_001",
  "identity_id": "identity_001",
  "label": "fake",
  "manipulation_scope": "face_localized",
  "manipulation_family": "face_swap",
  "generator_family": "unknown",
  "split": "train",
  "fps": 25,
  "duration_seconds": 11.4,
  "audio_available": true,
  "mask_available": false,
  "face_track_path": "tracks/video_001.json",
  "quality_path": "quality/video_001.json"
}
```

### Step 3 — Prevent data leakage
- Never split frames from one video across train and test
- Keep original-real and manipulated versions in the same split
- Split by identity where labels exist
- Hold out entire manipulation families for zero-shot tests
- Hold out generator families for unseen-generator tests
- Save split files permanently
- Never tune thresholds on the final test set

### Step 4 — Build deterministic preprocessing
Pipeline: video metadata → frame sampling → face detection → face tracking → landmark extraction → face alignment → quality measurement → optional context crop → temporal window creation → mask alignment → manifest generation.

Precompute once. Never re-run face detection during training epochs.

**Quality features:** face area ratio, blur score, brightness, contrast, pose estimate, occlusion estimate, track continuity, frame count, compression estimate, landmark jitter.

### Step 5 — Build and lock baselines

| Baseline | Purpose |
|---|---|
| RGB EfficientNet-B0 | Conventional spatial detector |
| DINOv2 frozen linear probe | Foundation model zero-shot |
| DINOv2 LayerNorm adaptation | Foundation model adapted |
| Forensic residual CNN only | Low-level detector |
| Temporal GRU only | Temporal detector |
| Simple unweighted fusion | Naive combination |

Every baseline uses the same face detector, alignment, frame sampling, train/test split, aggregation, and evaluation code. If FORGE-Lite does not beat the strongest baseline under the same preprocessing, it is not an improvement.

### Step 6 — Train the semantic branch
Start frozen + linear, then test LayerNorm, LoRA, last-block tuning. Select by cross-domain macro-AUROC. Never select by in-domain score alone.

### Step 7 — Train the forensic residual branch
Build fixed SRM/Haar preprocessing first (no gradients). Train the lightweight CNN. Evaluate the forensic branch independently on cross-domain and compression robustness. Keep it only if ablations show measurable improvement beyond the semantic branch alone.

### Step 8 — Add temporal modeling
Use 16-frame windows. Use aligned frame differences, not raw. Use GRU (128–256 hidden dim). Use robust video-level aggregation (mean + top-10% + max + dispersion).

### Step 9 — Add quality-aware fusion
Build the reliability-aware fusion head. Introduce branch availability masks. Test missing-modality robustness explicitly by ablating each branch at test time.

### Step 10 — Add codec consistency training
Apply post-processing at the video level, not only after the face crop. Use the offline codec bank (clean, mild, medium, heavy compression, resize+compression, blur+compression). Apply cheap image-level transforms online. The model must learn compression is not forgery.

**Training loss:**
```
Total loss =
  classification loss
+ consistency loss (stable scores across benign transforms)
+ source-pair metric loss
+ localization loss (when masks exist)
+ optional temporal consistency loss
```

### Step 11 — Add the localization head
Build decoder from DINO patch tokens + forensic feature maps. Train with BCE + Dice. Report IoU only on valid ground-truth masks. Show attention rollout only as supporting visualization.

### Step 12 — Add rPPG experimentally
Only after core model works. Run full ablation (without, with no gate, with quality gate). Keep only if cross-domain performance improves without false positive increase on real low-quality videos.

### Step 13 — Add adversarial robustness
Build two model versions: clean/generalization model, robustness-trained model. Train with PGD-3 or PGD-5. Evaluate with adaptive PGD-10 / PGD-20 against the entire fused pipeline. Report clean AUC, attacked AUC, and the tradeoff.

### Step 14 — Calibration and abstention
Reserve a separate calibration set. Use temperature scaling + uncertainty estimation. Combine: calibrated score + prediction uncertainty + input quality + evidence coverage → final decision rule.

**Measure:** Expected Calibration Error (ECE), Brier score, reliability diagram, coverage-risk curve, abstention rate.

### Step 15 — Run the locked evaluation protocol
Never touch the test set before this step.

### Step 16 — Build the demo and write the paper from measured results
Not from expected results. Not from architecture diagrams. From the actual numbers.

---

---

# SECTION 08 — Evaluation Protocol

---

## 8. Evaluation Protocol

### 8.1 Evaluation Groups

**In-domain:**
- FaceForensics++ C23
- FaceForensics++ C40 (compression stress test)

**Cross-domain:**
- Celeb-DF v2
- DFDC Preview
- WildDeepfake

**Modern generator:**
- Face-centric samples from AIGVDBench or FVBench subset only
- Do not claim a face-only detector has solved full-scene video generation

**Generalization protocols:**
- Train on one dataset, test on another
- Leave-one-manipulation-out
- Leave-one-generator-family-out
- Leave-one-dataset-out
- Source-disjoint testing
- Identity-disjoint testing

**Robustness protocols:**
- Compression (H.264 CRF 23, 30, 40)
- Resize, blur, frame drop, color changes, noise
- Adversarial attacks (PGD-10 adaptive, PGD-20, transfer)

### 8.2 Metrics

| Metric | Description |
|---|---|
| Video AUC | Primary detection metric |
| Frame AUC | Frame-level detection |
| Cross-domain macro-AUROC | Mean AUC across all cross-domain datasets |
| TPR @ 1% FPR | Operational metric for high-precision settings |
| EER | Equal Error Rate |
| AP | Average Precision (Precision-Recall AUC) |
| Localization IoU / Dice | Valid masks only |
| ECE | Expected Calibration Error |
| Brier Score | Probabilistic accuracy |
| Abstention rate | % of inputs returning INSUFFICIENT_EVIDENCE |
| Selective accuracy | Accuracy over non-abstained predictions |

### 8.3 Slice Analysis

Report performance broken down by:
- Face size category, pose angle, lighting condition
- Occlusion level, blur level, compression level
- Video length, manipulation family, generator family

Do not report only one pooled AUC.

### 8.4 Ablation Study

Do not run all 128 combinations. Run these 12:

| Config | Purpose |
|---|---|
| RGB only | Semantic-only baseline |
| Forensic only | Low-level baseline |
| Temporal only | Temporal baseline |
| Semantic + forensic | Two-branch |
| Semantic + temporal | Two-branch |
| Semantic + forensic + temporal | Three-branch |
| Full model without consistency loss | Tests codec training |
| Full model without source-pair training | Tests identity shortcut prevention |
| Full model without quality-aware fusion | Tests reliability gating |
| Full model without rPPG | Tests physiological contribution |
| Full model with unweighted fusion | Tests learned vs. fixed weights |
| Full FORGE-Lite | Final system |

Every ablation reports: in-domain AUC, cross-domain macro-AUROC, TPR@1%FPR, ECE, compression degradation, VRAM, latency.

---

---

# SECTION 09 — Research Novelty

---

## 9. Research Novelty

### 9.1 What Is NOT Novel (components already in literature independently)
- DINOv3 for classification
- SRM forensic filtering
- rPPG for deepfake detection
- Temporal GRU encoding
- Calibration of deepfake detectors

### 9.2 The Actual Novel Contribution

> **Codec-Consistent Reliability-Aware Evidence Fusion**

Specifically:

**Contribution 1 — Reliability-aware fusion with quality-conditioned branch gating.** A jointly trained fusion head that learns to weight each evidence branch based on the quality and availability of that branch's input — preventing low-quality or missing modalities from corrupting the prediction.

**Contribution 2 — Codec-consistency training strategy.** Consistency objectives applied across realistic codec transformations at the video level (not just image augmentation) that reduce generator-specific frequency artifacts and prevent spectral bias — extending FreqDebias principles into a multi-branch temporal pipeline.

**Contribution 3 — Calibrated selective prediction for deepfake detection.** A full calibration and abstention framework with documented coverage-risk curves. A detector that knows when it does not know is more valuable than one that confidently labels unusable video.

**Contribution 4 — Resource-efficient generalization on consumer hardware.** Complete evaluation protocol — cross-dataset, unseen manipulation, codec robustness, calibration, abstention, localization — running on RTX 4060 8GB via foundation-model parameter-efficient adaptation.

### 9.3 The Conditionally Valid Novel Claim

This novelty claim remains conditional until you complete a proper literature search confirming no published system jointly addresses all four contributions above in a single face deepfake video detection pipeline. Check arXiv, CVPR 2025–2026, ICCV 2025, WACV 2026, IEEE TIFS before locking the claim.

### 9.4 The Defensible Paper Statement

> "FORGE-Lite is a quality-aware, codec-consistent, foundation-adapted spatial-temporal detector for facial deepfake videos. It combines semantic features, low-level forensic residuals, temporal evidence, optional physiological measurements, calibrated reliability-aware fusion, and selective prediction. Its primary contribution is not the number of streams, but demonstrable cross-domain generalization under realistic post-processing and controlled failure handling."

---

---

# SECTION 10 — Release Gates

---

## 10. Release Gates

All gates must pass before the system is submitted or released.

### Scientific Validity
- No source leakage, no identity leakage, no frame leakage
- Locked test set — never touched before final evaluation
- Three random seeds with confidence intervals on video-level metrics

### Generalization
- Cross-domain macro-AUROC improvement ≥ 3pp over strongest baseline
- In-domain degradation ≤ 1pp

### Robustness
- Compression degradation ≤ 10% relative at operating point
- Adversarial claims use adaptive end-to-end attacks
- Clean and attacked results both reported

### Calibration
- ECE ≤ 0.05
- Brier score reported
- Abstention curve (coverage vs. risk) reported

### Localization
- Pixel metrics only on valid documented ground-truth masks
- Learned mask head evaluated separately from attribution heatmaps

### Deployment
- Peak VRAM measured and reported
- P50 and P95 latency measured and reported
- Model version and preprocessing version recorded in every output
- Temporary files deleted on cleanup
- Low-quality inputs rejected safely with INSUFFICIENT_EVIDENCE

### Research Quality
- Every retained component has ablation evidence
- Failure cases documented and included in the paper
- No unsupported universal claims
- All datasets and pretrained models have documented licenses reviewed before use

---

---

# SECTION 11 — Repository Structure

---

## 11. Repository Structure

```
forge-lite/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── eval.yaml
├── data/
│   ├── manifests/
│   ├── splits/
│   ├── tracks/
│   ├── quality/
│   ├── masks/
│   └── features/
├── preprocessing/
│   ├── decode.py
│   ├── detect_faces.py
│   ├── track_faces.py
│   ├── align_faces.py
│   ├── quality.py
│   └── build_manifest.py
├── models/
│   ├── dino_branch.py
│   ├── forensic_branch.py
│   ├── temporal_branch.py
│   ├── rppg_branch.py
│   ├── fusion.py
│   ├── localization.py
│   └── calibration.py
├── training/
│   ├── train_baseline.py
│   ├── train_spatial.py
│   ├── train_temporal.py
│   ├── train_fusion.py
│   └── adversarial_train.py
├── evaluation/
│   ├── evaluate_video.py
│   ├── evaluate_cross_domain.py
│   ├── evaluate_robustness.py
│   ├── evaluate_localization.py
│   ├── evaluate_calibration.py
│   └── bootstrap_ci.py
├── service/
│   ├── api.py
│   ├── inference.py
│   └── schemas.py
├── app/
│   └── gradio_app.py
├── reports/
│   ├── tables/
│   ├── figures/
│   └── failure_cases/
└── README.md
```

---

---

# SECTION 12 — Final Positioning

---

## 12. Final Positioning

The strongest version of this project is NOT:

> "A multi-stream system that reverse-engineers every deepfake generator and achieves 100% accuracy."

That is scientifically vulnerable and cannot be defended.

The strongest version IS:

> "FORGE-Lite is a quality-aware, codec-consistent, foundation-adapted spatial-temporal detector for facial deepfake videos. It combines semantic features, low-level forensic residuals, temporal evidence, optional physiological measurements, calibrated reliability-aware fusion, and selective prediction. Its primary contribution is not the number of streams, but demonstrable cross-domain generalization under realistic post-processing and controlled failure handling."

This version can:
- Run fully on RTX 4060 8GB
- Be completed without scope creep
- Produce scientifically meaningful ablations
- Support a credible CVPR 2027 workshop paper submission
- Generate a compelling live demonstration
- Be extended to audio, multi-face, and full-scene AI video in v2.0

---

## Implementation Order — Final

```
1.  Freeze the claim
2.  Build the manifest
3.  Prevent leakage
4.  Build preprocessing
5.  Establish baselines
6.  Train DINO semantic branch
7.  Train forensic residual branch
8.  Add temporal modeling
9.  Add quality-aware fusion
10. Add codec consistency training
11. Add localization head
12. Add rPPG experimentally
13. Add adversarial robustness
14. Calibrate and implement abstention
15. Run locked cross-domain evaluation
16. Build demo and write paper from measured results
```

---

*Document ends — FORGE-Lite PRD v1.0 — APPROVED FOR DEVELOPMENT*