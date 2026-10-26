# What Are We Actually Measuring? Pairing Choices as a Hidden Driver of Deepfake Detection Accuracy

**Abhi**  
Department of Artificial Intelligence & Machine Learning  
Madhav Institute of Technology and Science (MITS), Gwalior, India

*Note on AI assistance: Python pipelines, evaluation scripts, bootstrap computations, and portions of this draft were developed with AI pair-programming support (Antigravity / Claude). Hypotheses and decision rules were pre-registered before running experiments; pre-registration and any post-hoc deviations are documented in PREREG.md and ANALYSIS_LOG.md respectively. Every number in this paper comes from a machine-written JSON ledger — nothing was typed or rounded by hand.*

---

## Abstract

Deepfake detection methods routinely report AUROC values above 0.97 on standard benchmarks, yet the same methods often fall apart the moment they face video footage from a different source. The standard explanation is generalization failure — the model learned something too specific to its training distribution. That may be true. But there is a more basic problem that gets less attention: the benchmarks themselves are contaminated, not by bad data, but by how real and fake samples are paired together.

This paper is an audit. We take the same frozen encoder (DINOv2-S/14, no fine-tuning, no task-specific head beyond a logistic regression) and evaluate it under different pairing conditions on FaceForensics++ and Celeb-DF-v2. The results are hard to dismiss. Under an uncurated cross-dataset pairing — DF40 PNG fakes against FF++ JPEG reals — the probe reaches AUROC 0.998 ± 0.001. When we test that same probe on geometry-matched real frames (C→M cross-evaluation), performance drops to 0.596 ± 0.013, essentially chance. The probe is not detecting manipulation; it is detecting image container properties.

The background region alone — center-masked so the face is replaced with the image mean — reaches 0.998 AUROC on naive pairings and drops to 0.591 ± 0.010 once we enforce a shared spatial transform on both the real and fake crops. We call this the pairing effect, and we document it across DINOv2, a supervised ResNet-50, and a supervised ViT-B/16. All three show the same collapse when pairings are cleaned up. For a separate sanity check on our combined multi-dataset training manifest (949,254 samples), a classifier that never looks at an image pixel — only the dataset label — achieves AUROC 0.9208 [0.9201, 0.9216]. That number alone should give pause to anyone reporting results on such pools without controlling for this.

We also audit Celeb-DF-v2, which is generally considered a cleaner benchmark because fakes are generated from real source videos. Low-level corner probes pass at chance (AUROC 0.458 [0.413, 0.503]), but a DINOv2 background probe still reaches 0.751 ± 0.005. Enforcing a shared spatial transform brings this down to 0.604 ± 0.011, near our pre-registered ceiling. The shortcut is real and it is not caught by the standard sanity checks in the literature.

Our conclusion is narrow but concrete: in our setting, pairing choices set the ceiling on reported accuracy as much as any model property. Future benchmarks need shared spatial transforms and routine background-probe audits before publishing detection scores.

---

## 1. Introduction

The gap between benchmark performance and real-world deployment in deepfake detection has been noted since at least 2020. Methods achieving 0.98+ AUROC on FaceForensics++ [Rössler et al., 2019] regularly drop below 0.65 when evaluated on footage from different cameras, compression chains, or generative models [Yan et al., 2023]. The accepted explanation is distribution shift: training on one set of generators creates brittle, generator-specific features that do not transfer.

That explanation is probably partially right. But the frame it provides — better architectures, more diverse training, stronger augmentation — keeps attention on the model side of the problem. What if part of the issue is on the evaluation side?

Consider how most deepfake benchmarks are constructed. A real video is recorded with one camera, cropped with one face-tracker, and saved in one codec. A fake video is generated from that same source, but then processed independently: different face-detection threshold, different crop boundary, often a different image format (PNG for synthetic renders, JPEG for real captures). Any modern deep encoder trained to distinguish these two populations will find the easier signal first. It does not need to detect manipulation artifacts; the image containers are already separable.

This is not a new observation in the broader vision literature. Torralba and Efros [2011] showed dataset bias in object recognition nearly fifteen years ago; Geirhos et al. [2020] documented shortcut learning in texture vs. shape perception. In generated-image detection specifically, Grommelt et al. [2024] showed that JPEG quantization tables and downsampling artifacts predict whether an image is AI-generated, independent of semantic content. What is less documented — and what this paper tries to measure directly — is how much of the reported improvement in deepfake detection accuracy disappears once the pairing confound is controlled.

We run a deliberately simple experiment. We take a frozen DINOv2-S/14 encoder — no fine-tuning, logistic regression probe on top — and evaluate it under multiple pairing conditions on FaceForensics++ [Rössler et al., 2019] and Celeb-DF-v2 [Li et al., 2020c]. We also run the same evaluation with frozen ResNet-50 and ViT-B/16 to check that results do not depend on the particular encoder. The key manipulation is whether real frames and fake frames share the same spatial crop transform (identical affine parameters extracted from the real frame applied to both) or whether they are cropped independently.

We pre-registered all primary hypotheses and decision thresholds before running any experiment (tracked with git timestamps in PREREG.md). Where results contradict our expectations, we say so directly in Section 5.

### 1.1 What We Are and Are Not Claiming

To be precise about scope: everything in this paper is bounded to our experimental setting. We are not claiming that every deepfake benchmark is fatally flawed, or that no method in the literature has found real forensic signal. We are claiming that in the specific setting we tested — frozen linear probes on FaceForensics++ and Celeb-DF-v2 under different pairing regimes — reported accuracy tracks pairing choices very closely, and that the background region outside the face carries most of the discriminative signal under naive pairings.

Five concrete claims motivate the experiments:

1. **Naive pairing inflation.** A probe trained on cross-dataset naive pairs achieves near-perfect AUROC but collapses when tested on geometry-matched pairs from the same videos.
2. **Background separability.** Under naive pairings, the image background alone (face region blanked out) is sufficient for near-perfect classification.
3. **Shared transforms as a clean reference.** Applying the same affine crop to real and fake reduces the background probe to chance (AUROC ≤ 0.60).
4. **Architecture independence.** This pattern holds for DINOv2, ResNet-50, and ViT-B/16.
5. **Native pairing fallacy.** Celeb-DF-v2-style target-video matching does not fully remove the background shortcut.

---

## 2. Background

### 2.1 Deepfake Benchmarks

FaceForensics++ [Rössler et al., 2019] remains the most widely used deepfake detection testbed, providing 1,000 real YouTube videos and their manipulations via Deepfakes, Face2Face, FaceSwap, NeuralTextures, and FaceShifter [Li et al., 2020a]. Celeb-DF-v2 [Li et al., 2020c] paired 590 real celebrity videos with 5,639 high-quality synthesis examples and was specifically designed to address quality imbalances in earlier datasets. More recently, DF40 [Yan et al., 2024a] and the DeepfakeBench evaluation suite [Yan et al., 2023] added cross-domain and leave-one-generator-out protocols across diffusion-based and GAN-based generators. The Deepfake Detection Challenge [Dolhansky et al., 2020] and DeeperForensics-1.0 [Jiang et al., 2020] scaled data diversity further.

What unites nearly all of these is that they apply independent face-tracking pipelines to real and synthetic videos. This is operationally convenient but creates the spatial alignment confound we study here.

### 2.2 Shortcuts and Dataset Biases

The vision community has documented dataset bias extensively. Torralba and Efros [2011] showed that dataset membership alone predicts object category better than it should. Geirhos et al. [2020] showed that ImageNet-trained models attend to texture rather than shape, which is a shortcut arising from training data statistics. In forensics, Frank et al. [2020] identified frequency-domain artifacts from GAN upsampling that allow classification without high-level semantic processing. Grommelt et al. [2024] audited the GenImage benchmark and found that JPEG quality factors, downsampling grids, and sensor noise differences account for most separability between real and AI-generated images. Yermakov [2026] subsequently showed across 14 benchmarks that source-matched real-fake pairing mitigates these shortcuts during foundation model adaptation.

Our contribution sits closest to Grommelt et al. [2024] but focuses specifically on spatial pairing — crop alignment and container format — rather than compression statistics. We use a shared spatial transform as a controlled clean-reference condition and verify its effectiveness with both geometry measurements and background-probe audits.

---

## 3. Methods

### 3.1 Pre-Registration

Every primary hypothesis and quantitative decision threshold was written into PREREG.md and committed to git before the corresponding experiment ran. Table 1 lists the seven decision rules.

**Table 1. Pre-Registered Decision Rules**

| Symbol | Condition | Threshold | What it tests |
| :--- | :--- | :---: | :--- |
| P-S1 | Paired face crop center MAD | ≤ 4.0 px | Geometric alignment is equalized |
| P-S2 | Center-masked probe AUROC, M→M | ≤ 0.60 | Background shortcut is gone |
| P-S3 | Corner-masked probe AUROC, M→M | ≥ 0.85 | Face region retains synthesis signal |
| P-S4 | Full-image probe AUROC, M→M | ≤ 0.70 | Pairing cleanup reduces reported accuracy |
| P-C1 | Dataset-ID classifier 5-fold CV AUROC | ≥ 0.90 | Dataset label predicts real/fake |
| P-C2 | Celeb-DF native background probe AUROC | ≤ 0.60 | Native pairing is background-clean |
| P-C3 | Multi-backbone background probe AUROC, M→M | ≤ 0.60 | Background removal is architecture-agnostic |

A pairing protocol counts as a clean reference only if P-S1 *and* P-S2 are both met. If either fails, the pairing remains confounded.

### 3.2 Pre-Registration Chronology

Rules P-S1 through P-S4 and P-C1 were committed before any experiment ran (commit `cb5b1bf99`, 2026-10-10 21:39 IST; experiments ran 55 minutes to several hours later). P-C2 was committed before the Celeb-DF runs. P-C3 (multi-backbone replication) is **exploratory**: the git commit (`eeaa3cc03`, 00:46 IST) postdates the first ResNet-50 run (00:23–00:26 IST) and first ViT-B/16 run (00:40–00:44 IST). We note this because it affects how strongly the multi-backbone result should be weighted; subsequent re-runs postdate the commit, but we cannot establish strict pre-registration for the initial runs.

### 3.3 Encoders and Probe

We evaluate three frozen encoders:

- **DINOv2-S/14** (self-supervised ViT-S, patch size 14): 768-d features — concatenation of the 384-d CLS token and the 384-d mean of all 256 patch tokens. No fine-tuning.
- **ResNet-50** (supervised ImageNet-1k): 2048-d embedding from the penultimate pooling layer.
- **ViT-B/16** (supervised ImageNet-1k): 768-d CLS token.
- **Handcrafted low-level baseline**: 8-d vector — log JPEG filesize per pixel, log Laplacian variance, high-frequency FFT energy ratio, grayscale standard deviation, grayscale mean, R/G/B channel means.

The probe is logistic regression with L2 regularization (C=0.1) and balanced class weights in every case. No architecture search, no hyperparameter tuning on test data.

### 3.4 Pairing Conditions

- **C→C (naive):** DF40 PNG fakes paired with FF++ JPEG reals. Independent face trackers. Establishes a confounded ceiling.
- **M→M (shared-transform):** Real frame and fake frame extracted from the same video at the same timestamp, cropped with the same affine transform derived from the real. 600 pairs, 4 frames per pair.
- **C→M (cross-evaluation):** Probe trained on C pairing, evaluated on M-matched reals.
- **FF++ native:** Standard FaceForensics++ crops from the dataset's own face-tracker output.

Shared transform pairs were generated by `scripts/extract_shared_transform_pairs.py`, which extracts face-alignment affine parameters from each real frame and applies them identically to the corresponding deepfake at the same timestamp. Paired crop center MAD was 0.721 px [Measured], well under the 4.0 px threshold.

### 3.5 Spatial Masking

To locate where the shortcut signal comes from, we evaluate probes on two masked variants of each image:

- **face_out (center-masked):** The central 112×112 square (25% of a 224×224 image, 12,544 px) is replaced with the per-image mean RGB colour. The remaining 75% includes peripheral face boundary, hairline, ears, and background.
- **face_only (corner-masked):** The four 40×40 corners (12.8% total, 6,400 px) are replaced with mean colour. Note: 87.2% of the image remains visible, including most of both the face *and* the background. This mask does not cleanly isolate face from background.

Masking is applied in pixel space before encoding; the encoder never sees the unmasked region when a mask is active. Implementation is in `evaluation/linear_probe_beforeafter.py`.

Evaluation uses official scene-disjoint FF++ splits (140 test scenes, roughly 249 real and 296 fake frames per seed). We run five seeds [42, 7, 13, 99, 2024] and report mean ± standard deviation across seeds alongside 95% bootstrap CIs over test scenes (300 resamples, scene as the resampling unit).

---

## 4. Results

All numbers below come from JSON ledgers in `reports/`, generated by evaluation scripts and read back by `evaluation/make_latex_tables.py`. Nothing was entered by hand.

### 4.1 The Pairing Effect

The clearest result is the gap between C→C and C→M performance. A DINOv2 probe trained on naive pairings reaches AUROC 0.998 ± 0.001 [Measured]. When tested on geometry-matched real frames from the same videos (C→M), it drops to 0.596 ± 0.013 [Measured] — essentially chance. ResNet-50 goes from 0.972 to 0.524; ViT-B/16 from 0.966 to 0.553. The probe is not detecting manipulation; it is detecting container.

The same-frame ablation clarifies what drives the gap. When we independently align crops on the exact same 600 video pairs and frames (rather than sharing a transform), DINOv2 achieves 0.774 ± 0.004 under C→C and 0.767 ± 0.005 under M→M — essentially identical across conditions. Independent face-tracker alignment on identical source material is not what creates the shortcut. The large gap between FF++ native (0.874) and shared-transform clean (0.763) comes from container format, compression, and frame-selection differences, not from alignment differences alone.

**Table 2. Pairing Effect Across Representations (FaceForensics++, 5 seeds)**

| Encoder | C→C | C→M | M→M | M→C | M→M Bal.Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DINOv2-S/14 (naive vs clean) | 0.998 ± 0.001 | 0.596 ± 0.013 | 0.763 ± 0.012 | 0.793 ± 0.035 | 0.691 |
| DINOv2-S/14 (FF++ native) | 0.874 ± 0.006 | — | — | — | 0.784 |
| DINOv2-S/14 (same-frame, indep. align) | 0.774 ± 0.004 | 0.772 ± 0.007 | 0.767 ± 0.005 | 0.770 ± 0.004 | 0.690 |
| ResNet-50 | 0.972 ± 0.006 | 0.524 ± 0.009 | 0.613 ± 0.013 | 0.659 ± 0.031 | 0.592 |
| ViT-B/16 | 0.966 ± 0.005 | 0.553 ± 0.010 | 0.690 ± 0.010 | 0.718 ± 0.024 | 0.638 |
| Handcrafted baseline | 0.812 ± 0.007 | 0.467 ± 0.009 | 0.546 ± 0.009 | 0.299 ± 0.016 | 0.540 |

*Source: `reports/lp_shared_dino_5seeds.json`, `reports/lp_ff_native_dino_5seeds.json`, `reports/lp_ff_indep_dino_5seeds.json`, `reports/lp_shared_resnet50_5seeds.json`, `reports/lp_shared_vitb16_5seeds.json`*

### 4.2 Where the Signal Lives

The background probe (face_out, center-masked) is the sharpest diagnostic. Under naive C→C pairings, removing the face does not hurt at all: DINOv2 background-only AUROC is 0.998 ± 0.001 [Measured]; ResNet-50 and ViT-B/16 both reach 0.986 ± 0.004 [Measured]. You do not need to see the deepfake to detect it — the background already carries the answer.

Under shared-transform M→M pairing, the background probe drops to 0.591 ± 0.010 [0.580, 0.603] on DINOv2, 0.537 ± 0.009 on ResNet-50, and 0.567 ± 0.009 on ViT-B/16 [Measured]. The DINOv2 point estimate (0.591) clears our pre-registered ceiling of ≤ 0.60 (P-S2 met), though the 95% CI upper bound reaches 0.603. ResNet-50 and ViT-B/16 clear the ceiling with their full CIs below 0.60 (exploratory P-C3).

The same-frame independent alignment ablation gives 0.600 ± 0.010 on the background probe [Measured], a delta of only 0.009 relative to shared-transform. This directly tests the alignment hypothesis: independent face-tracker choices on identical PNG frames do not recreate the ~0.758 background AUROC seen in native FF++ crops.

**Table 3. Background Probe Across Pairing Conditions (AUROC, 5 seeds)**

| Encoder | Center-masked C→C | Center-masked M→M | Corner-masked M→M |
| :--- | :---: | :---: | :---: |
| DINOv2 (shared-transform) | 0.998 ± 0.001 | 0.591 ± 0.010 [P-S2 ✓] | 0.774 ± 0.007 |
| DINOv2 (same-frame, indep. align) | 0.596 ± 0.013 | 0.600 ± 0.010 | 0.785 ± 0.011 |
| DINOv2 (FF++ native) | 0.758 ± 0.006 | 0.755 ± 0.011 | 0.872 ± 0.009 |
| ResNet-50 (shared-transform) | 0.986 ± 0.004 | 0.537 ± 0.009 [exploratory] | 0.619 ± 0.011 |
| ViT-B/16 (shared-transform) | 0.986 ± 0.004 | 0.567 ± 0.009 [exploratory] | 0.706 ± 0.012 |

*Source: `reports/lp_shared_dino_face_out_5seeds.json`, `reports/lp_ff_indep_dino_face_out_5seeds.json`, `reports/lp_ff_native_dino_face_out_5seeds.json`, `reports/lp_shared_resnet50_face_out_5seeds.json`, `reports/lp_shared_vitb16_face_out_5seeds.json`*

On corner-masked images (face_only), DINOv2 achieves 0.774 ± 0.007 under shared-transform M→M [Measured], short of our pre-registered threshold of ≥ 0.85 (P-S3 not met). Because this mask leaves 87.2% visible — including most of the background — it cannot be read as face signal only. We discuss this in Section 5.

### 4.3 Celeb-DF-v2: When Low-Level Checks Pass but the Shortcut Remains

Celeb-DF-v2 is often treated as a cleaner benchmark because synthesis videos were generated from real source footage rather than independently sourced. The assumption is that this source-video pairing removes the container shortcut.

Low-level handcrafted features on the outer corners of Celeb-DF-v2 crops give AUROC 0.458 [0.413, 0.503] [Measured] — solidly at chance, consistent with a well-paired benchmark. But the DINOv2 background probe (face_out) on the same images reaches 0.751 ± 0.005 [Measured]. That is a substantial gap: the standard sanity check says clean, but a deep encoder finds a reliable background signal anyway.

This contradicts P-C2. We [Hypothesis] attribute it to bounding-box jitter in independent face-tracking: when real and fake crops are not derived from identical affine parameters, the boundary regions encode tracker-specific differences that a deep encoder can distinguish.

Applying a shared spatial transform reduces the Celeb-DF background AUROC from 0.751 to 0.604 ± 0.011 [Measured], reaching the P-C2b decision boundary. Low-level features under shared transform drop to chance (0.507 ± 0.003, balanced accuracy 0.504 [Measured]). A probe trained on native pairs collapses to 0.532 ± 0.007 when evaluated on shared-transform pairs [Measured].

**Table 4. Celeb-DF-v2 Pairing Audit**

| Condition | Region | AUROC | Status |
| :--- | :--- | :---: | :---: |
| **Native (independent tracking)** | | | |
| Handcrafted low-level | Outer corners | 0.458 [0.413, 0.503] | At chance |
| DINOv2 | Background (face_out) | 0.751 ± 0.005 | **P-C2 failed** |
| DINOv2 | Face region (face_only) | 0.911 ± 0.010 | Synthesis signal |
| DINOv2 | Full image | 0.889 ± 0.007 | Confounded |
| **Shared-transform** | | | |
| Low-level | Background (face_out) | 0.507 ± 0.003 | At chance |
| DINOv2 | Background (face_out, M→M) | 0.604 ± 0.011 | P-C2b boundary |
| DINOv2 | Cross-eval (C→M) | 0.532 ± 0.007 | Collapses |
| DINOv2 | Face region (face_only, M→M) | 0.835 ± 0.014 | Residual signal |
| DINOv2 | Full image (M→M) | 0.806 ± 0.015 | Clean reference |

*Source: `reports/region_probe_celeb.json`, `reports/lp_celeb_5seeds.json`, `reports/lp_celeb_face_out_5seeds.json`, `reports/lp_celeb_shared_face_out_5seeds.json`, `reports/lp_celeb_shared_5seeds.json`, `reports/lp_celeb_shared_face_only_5seeds.json`*

### 4.4 Dataset Label as a Classifier

The starkest demonstration of benchmark contamination does not need image pixels at all. On our combined training manifest (949,254 samples across 9 datasets), a Bayesian estimator predicting P(fake | dataset_name) via 5-fold cross-validation achieves AUROC 0.9208 [0.9201, 0.9216] [Measured]. P-C1 is met.

This is not a model learning forensics; it is a lookup table. DF40 is 100% fake in the pool (686,358 samples), FFHQ and HDTF are 0% fake, FakeAVCeleb is 97.7% fake. A model without pixels can assign near-correct labels to most of the dataset by remembering which source is predominantly real and which is predominantly fake. Any method trained on this pool without explicit balancing will partly learn "this looks like it came from DF40" rather than "this has GAN upsampling artifacts."

**Table 5. Dataset Provenance Shortcut (949,254 samples)**

| Dataset | N | Fake % | P(Fake ∣ Dataset) |
| :--- | :---: | :---: | :---: |
| DF40 | 686,358 | 100.0 | 1.000 |
| FaceForensics++ | 127,068 | 85.8 | 0.858 |
| Celeb-DF-v2 | 84,900 | 85.7 | 0.857 |
| WildDeepfake | 23,100 | 51.7 | 0.517 |
| FakeAVCeleb | 21,566 | 97.7 | 0.977 |
| FFHQ | 4,062 | 0.0 | 0.000 |
| LAV-DF | 2,000 | 100.0 | 1.000 |
| HDTF | 100 | 0.0 | 0.000 |
| TalkingHeadBench | 100 | 100.0 | 1.000 |
| **5-fold CV AUROC** | | | **0.9208 [0.9201, 0.9216]** |

*Source: `reports/dataset_id_baseline.json`*

### 4.5 Geometry and Pixel-Level Verification

Shared-transform pairing achieves a face crop center MAD of 0.721 px [Measured] — well under 4.0 px (P-S1 met). Independent alignment on the same frames gives 0.823 px.

Background pixel differences under shared transform: real-vs-fake MAD median is 3.244 gray levels [2.415, 4.994]; fake-vs-fake MAD is 3.372 [2.517, 5.207], ratio 0.962 [Measured]. The ratio equivalence test at ±0.05 margin passes. A paired Wilcoxon test is significant (p = 1.96×10⁻¹²), reflecting a ~3.8% lower real-vs-fake background variance — small and in the right direction but not a practical shortcut. Under independent alignment on identical frames, real-vs-fake background MAD is 5.646 gray levels and the ratio equivalence test fails.

**Table 6. Geometry and Low-Level Verification (FaceForensics++ shared pairs)**

| Measurement | Value | Criterion |
| :--- | :---: | :---: |
| Crop center MAD | 0.721 px | ≤ 4.0 px ✓ |
| Background MAD: Real vs Fake | 3.244 [2.415, 4.994] gl | — |
| Background MAD: Fake vs Fake | 3.372 [2.517, 5.207] gl | noise floor |
| Ratio of medians (RF/FF) | 0.962 | ±0.05 margin ✓ |
| Wilcoxon p | 1.96×10⁻¹² | ~3.8% difference |
| Composited low-level background probe | 0.437 [0.383, 0.486] | Below chance |
| Reenactment low-level background probe | 0.521 [0.482, 0.570] | [0.45, 0.55] ✓ |

*Source: `reports/paired_stats_shared.json`, `reports/region_probe_shared.json`*

---

## 5. What Went Wrong

Three pre-registered thresholds were not met. We report each with the exact measured number.

**P-S3 not met — Corner-masked signal retention.** We expected the face_only probe on clean M→M pairs to achieve AUROC ≥ 0.85. Measured: 0.774 ± 0.007 [Measured]. The corner mask leaves 87.2% of the image visible — a substantial fraction of background — so this cannot be read as pure face signal. We cannot determine from this alone whether frozen DINOv2-S/14 lacks the necessary resolution to detect c23-level artifacts in a 112×112 face region, or whether the shortcut it previously exploited is now absent and the face-region signal is genuinely weaker. [Hypothesis]

**P-S4 not met — Full-image clean AUROC bound.** We expected the full-image probe on clean M→M pairs to be ≤ 0.70 if pairing dominates reported accuracy. Measured: 0.763 ± 0.012 [Measured]. This means even after removing the pairing confound, there is detectable synthesis signal in the face region. The pairing effect is real and large (full-image drops from 0.998 to 0.763; background drops from 0.998 to 0.591), but face-region forensic information does not disappear entirely under clean pairings. [Hypothesis]

**P-C2 not met — Celeb-DF native background cleanliness.** We expected native Celeb-DF-v2 target-video pairing to pass the background probe at ≤ 0.60. Measured: 0.751 ± 0.005 [Measured]. The shared-transform ablation subsequently brings this to 0.604 (P-C2b boundary), suggesting the failure mode is face-tracker bounding-box jitter rather than something structural about the benchmark design. [Hypothesis]

**Disclosed separately — Legacy training confound.** A post-hoc audit of `configs/train.yaml` found that `pos_weight=18.4` was combined with `balanced_sampling=true`. In `data/dataset.py`, label 1 is assigned to fakes, so this configuration applies excess loss weight to fakes, not reals as might be intended. Additionally, that training pool is the confounded combined manifest where dataset ID alone achieves 0.9208 AUROC. None of the numbers in this paper come from that checkpoint; all results are generated by frozen linear probes evaluated from scratch.

---

## 6. Limitations

The experiment is deliberately narrow. FaceForensics++ c23 and Celeb-DF-v2 cover a specific slice of the detection landscape: studio-like videos, c23 compression, face-swap and reenactment methods from 2018–2020. More recent diffusion-based generators (many in DF40 [Yan et al., 2024a]) may produce artifacts distributed differently across spatial regions.

Linear probes are a specific architectural choice. They cannot capture spatial attention patterns or multi-scale feature interactions. We test three encoders to check that results are not encoder-specific, but the probe architecture itself is fixed.

Shared spatial transforms eliminate geometric misalignment but do not fully eliminate second-generation compression effects. When a fake frame is re-encoded into the same container as the real, there may be subtle re-encoding differences that a sufficiently sensitive feature extractor could pick up. We cannot rule this out from current measurements.

---

## 7. Recommendations

Three things would most improve deepfake benchmark validity:

**Enforce shared spatial transforms.** Real and fake frames should be extracted from the same source video at the same timestamp, using affine parameters derived once from the real frame and applied to both. This is not expensive — `extract_shared_transform_pairs.py` processes 600 pairs in minutes on a single GPU.

**Include a background-probe baseline in every benchmark release.** Any detector achieving AUROC > 0.60 on center-masked images should be treated as evidence of a pairing confound, not a model capability. This takes one extra evaluation run.

**Report a zero-pixel dataset-ID baseline for multi-dataset benchmarks.** If categorical dataset membership alone achieves near-perfect separation, the evaluation is measuring dataset provenance, not manipulation artifacts. This baseline is trivial to compute and should be standard.

---

## Reproducibility

All evaluation scripts (`evaluation/linear_probe_beforeafter.py`, `evaluation/make_latex_tables.py`, `evaluation/dataset_id_baseline.py`, `scripts/extract_shared_transform_pairs.py`) are deterministic under fixed seeds. Results are written to JSON ledgers in `reports/` with SHA-256 checksums. The full experiment log, pre-registration, and analysis notes are tracked in git at `https://github.com/abhishek036/deepfake`.

*AI assistance declaration: Python pipeline development, bootstrap computation, and portions of this text were written with AI assistance under researcher supervision. Hypotheses and thresholds are documented in git-timestamped commits in PREREG.md.*

---

## References

Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., & Canton-Ferrer, C. (2020). The DeepFake Detection Challenge (DFDC) Dataset. *arXiv:2006.07397*.

Durall, R., Keuper, M., & Keuper, J. (2020). Watch your Up-Convolution: CNN Based Generative Deepfake Detection. *CVPR 2020*.

Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., & Holz, T. (2020). Leveraging Frequency Analysis for Deep Fake Image Recognition. *ICML 2020*.

Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.

Grommelt, P., Weiss, L., Pfreundt, F. J., & Keuper, J. (2024). Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. *ECCV Workshops 2024*.

Jiang, L., Li, R., Wu, W., Qian, C., & Loy, C. C. (2020). DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection. *CVPR 2020*.

Li, L., Bao, J., Yang, H., Chen, D., & Wen, F. (2020a). Advancing High Fidelity Identity Swapping for Forgery Detection. *CVPR 2020*.

Li, L., Bao, J., Zhang, T., Yang, H., Chen, D., & Wen, F. (2020b). Face X-Ray for More General Face Forgery Detection. *CVPR 2020*.

Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020c). Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. *CVPR 2020*.

Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV 2019*.

Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. *CVPR 2011*.

Yan, Z., Yao, T., Chen, S., Zhao, Y., Fu, X., Zhu, J., Luo, D., Yuan, L., Wang, C., Ding, S., & Wu, Y. (2024a). DF40: Toward Next-Generation Deepfake Detection. *NeurIPS 2024 Datasets and Benchmarks Track*.

Yan, Z., Zhang, Y., Yuan, X., Lyu, S., & Wu, B. (2023). DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection. *NeurIPS 2023 Datasets and Benchmarks Track*.

Yermakov, A. (2026). Deepfake Detection that Generalizes Across Benchmarks. *WACV 2026*.
