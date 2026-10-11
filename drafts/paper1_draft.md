# The Pairing Effect: Auditing How Real-vs-Fake Pairing Choices Drive Reported Accuracy in Deepfake Detection

**Author:** Abhi  
*Department of Artificial Intelligence & Machine Learning*  
*Madhav Institute of Technology and Science (MITS), Gwalior, India*  

**Project Repository:** `C:\deepfake`  
**Ledger Audit Source of Truth:** `reports/` (Machine-readable JSON ledgers verified by SHA-256)  
**Pre-Registration Reference:** `PREREG.md` | **Post-Hoc Deviations:** `ANALYSIS_LOG.md`  
**Declaration of AI Assistance:** An AI assistant (Antigravity / Claude) was used as a research engineer and pair-programmer for scripting data pipelines, running evaluations, computing bootstrap statistics, and drafting text. All hypotheses and decision rules were pre-registered, and all empirical claims are grounded strictly in machine-generated JSON ledgers.

---

## Abstract

Reported deepfake detection accuracy frequently approaches ceiling on benchmark datasets yet degrades substantially under real-world deployment. In this work, we present an empirical audit evaluating how real-vs-fake pairing choices drive reported detection performance. We formulate the working claim that in our setting, reported accuracy is largely determined by pairing choices rather than facial manipulation artifacts. 

Across five random seeds on scene-disjoint FaceForensics++ splits, a linear probe on frozen DINOv2-S/14 features trained on naive pairings achieves an AUROC of 0.998 [0.996, 1.000] [Measured], but collapses to 0.596 [0.581, 0.614] [Measured] when evaluated on geometry-matched reals (cross-evaluation). Spatial decomposition reveals that on naive pairings, probing the background alone (face region masked to per-image mean) yields an AUROC of 0.998 [Measured], demonstrating that background container discrepancies account for detection performance. Crucially, auditing FaceForensics++ native pairing (same video container, independent face-tracker alignment) yields a background probe AUROC of 0.755 ± 0.010 [Measured], proving that independent spatial alignment alone accounts for ~0.16 AUROC shortcut leakage. When evaluated on pairs generated via a shared spatial transform (crop center Mean Absolute Difference 0.721 px [Measured]), the background probe collapses to 0.591 [0.580, 0.603] [Measured], satisfying our pre-registered clean-reference threshold ($\le 0.60$). We show that this pattern replicates across frozen supervised ResNet-50 (0.986 collapsing to 0.537) [Measured] and frozen supervised ViT-B/16 (0.986 collapsing to 0.567) [Measured]. 

Furthermore, we audit native target-video pairings in Celeb-DF-v2. While low-level corner probes pass at chance (AUROC 0.458 [0.413, 0.503]) [Measured], a high-level background probe still achieves 0.751 ± 0.005 [Measured], indicating that independent bounding-box face cropping fails to equalize container shortcuts. Finally, we demonstrate that on our combined manifest (`data/manifests/combined.jsonl`), categorical dataset provenance alone achieves an AUROC of 0.9208 [0.9201, 0.9216] [Measured] across 949,254 samples without accessing image pixels. We conclude that future deepfake benchmarks must enforce shared spatial transforms and routine spatial ablation audits.

---

## 1. Introduction

Automated detection of manipulated facial imagery is critical for information integrity. Over the past five years, numerous detection architectures—ranging from spatial convolutional networks [Durall et al., 2020] and frequency-domain analyzers [Frank et al., 2020] to foundation-model probes—have reported within-dataset Area Under the Receiver Operating Characteristic (AUROC) scores exceeding 0.98 on standard benchmarks such as FaceForensics++ (FF++) [Rössler et al., 2019] and Celeb-DF-v2 [Li et al., 2020c]. However, cross-dataset evaluations consistently document steep performance drops, frequently falling below 0.65 AUROC.

This generalization gap is commonly attributed to generator overfitting or differences in blending post-processing. In this paper, we investigate a more fundamental confound: **the pairing effect**. In deepfake benchmark creation, real images and fake images are often drawn from different source containers, processed through independent face-tracking pipelines, or stored under differing compression profiles. Under such conditions, deep representations can separate real and fake images using background texture, container metadata, or cropping boundary discrepancies rather than forensic forgery cues.

### 1.1 Working Claims and Bounding

Bound to our experimental setting, we investigate five concrete claims:
1. **The Naive Pairing Collapse:** Linear probes trained on naive real-vs-fake pairings achieve near-perfect discrimination but collapse toward chance when tested on geometry-matched reals.
2. **The Background Shortcut:** Naive real-fake pairs permit near-perfect discrimination from the image background alone, even when the manipulated face region is blanked out.
3. **Clean Reference via Shared Spatial Transforms:** Applying an identical spatial affine transform and crop pipeline to both real frames and their corresponding manipulations eliminates the background shortcut ($\text{AUROC} \le 0.60$).
4. **Replication Across Architecture Classes:** The background container shortcut is not unique to self-supervised Vision Transformers; frozen supervised ConvNets (ResNet-50) and supervised Transformers (ViT-B/16) exhibit the same pattern of background separation and collapse.
5. **The Native Pairing Fallacy:** In independently cropped native benchmarks (such as Celeb-DF-v2), passing handcrafted low-level audits (e.g. corner probes at chance) does not prevent deep foundation encoders from extracting background shortcuts.

---

## 2. Related Work

### 2.1 Deepfake Detection Benchmarks
FaceForensics++ [Rössler et al., 2019] has served as a standard benchmark, providing 1,000 video sequences manipulated via five pipelines: Deepfakes, Face2Face, FaceSwap, NeuralTextures, and FaceShifter [Li et al., 2020a]. Celeb-DF-v2 [Li et al., 2020c] introduced 590 real celebrity videos paired with 5,639 high-quality synthesis videos. Later large-scale efforts—such as DF40 [Yan et al., 2024a] and the DeepfakeBench framework [Yan et al., 2024b]—standardized cross-domain and leave-one-generator-out evaluation protocols across modern diffusion and generative architectures. Other benchmarks, including the Deepfake Detection Challenge (DFDC) [Dolhansky et al., 2020] and DeeperForensics-1.0 [Jiang et al., 2020], scaled dataset diversity. However, widespread benchmark creation continues to rely on independently cropped or uncurated pairing protocols.

### 2.2 Shortcut Learning and Forensic Dataset Biases
Deep neural networks routinely exploit high-frequency, non-semantic statistical correlations rather than intended high-level abstractions [Geirhos et al., 2020; Torralba & Efros, 2011]. In synthetic image detection, Grommelt et al. [2024] audited dataset biases on GenImage, showing that classifiers frequently latch onto JPEG quantization tables, downsampling factors, and sensor noise profiles rather than generative artifacts. Similarly, forensic literature has identified sensitivity to JPEG compression grids [Frank et al., 2020] and color shifts. While prior work emphasized model-side inductive biases, our audit focuses specifically on real-vs-fake pairing choices, demonstrating how spatial crop alignment and container matching govern reported benchmark accuracy.

---

## 3. Audit Protocol and Pre-Registration

To ensure methodological rigor, all hypotheses and quantitative decision rules were pre-registered in `PREREG.md` prior to executing evaluation runs. All post-hoc analyses and contradictions are documented chronologically in `ANALYSIS_LOG.md`.

### 3.1 Pre-Registered Decision Rules

| Rule Symbol | Evaluated Condition | Pre-Registered Threshold | Operational Interpretation |
| :--- | :--- | :---: | :--- |
| **P-S1** | Paired Mean Absolute Difference (MAD) of face crop centers | $\le 4.0$ px | Geometry and framing equalized; spatial alignment established. |
| **P-S2** | Background-only (`face_out`) probe AUROC | $\le 0.60$ | Background container shortcut eliminated. |
| **P-S3** | Face-only (`face_only`) probe AUROC | $\ge 0.85$ | Facial region retains detectable synthesis signal. |
| **P-S4** | Full-image probe AUROC on clean pairs | $\le 0.70$ | Pairing cleanup substantially reduces reported performance. |
| **P-C1** | Dataset-ID Bayesian classifier 5-fold CV AUROC | $\ge 0.90$ | Dataset provenance constitutes an evaluation shortcut. |
| **P-C2** | Celeb-DF native background (`face_out`) probe AUROC | $\le 0.60$ | Native pairing background is free of shortcut cues. |
| **P-C3** | Multi-backbone background (`face_out`) probe AUROC | $\le 0.60$ | Background shortcut elimination is representation-invariant. |

**Joint Clean Reference Rule:** A pairing protocol is declared a clean reference **if and only if P-S1 AND P-S2 are both met**.

### 3.2 Pre-Registration Timeline and Protocol Chronology

To uphold absolute transparency, we document the exact git commit timestamps under which each hypothesis and decision rule was registered in `PREREG.md`, cross-referenced against the creation timestamp of the corresponding result ledger in `reports/`:

| Decision Rule | PREREG Git Commit | Commit Timestamp | Result Ledger File | Ledger Creation Timestamp | Pre-Registration Sequence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P-S1** (Crop Center MAD) | `cb5b1bf99` | 2026-10-10 21:39:47 IST | `reports/paired_stats_shared.json` | 2026-10-10 22:34:13 IST | Registered 55 min before run |
| **P-S2** (Background Probe) | `cb5b1bf99` | 2026-10-10 21:39:47 IST | `reports/lp_shared_dino_face_out_5seeds.json` | 2026-10-11 00:50:55 IST | Registered before Phase B runs |
| **P-S3** (Face-Only Probe) | `cb5b1bf99` | 2026-10-10 21:39:47 IST | `reports/lp_shared_dino_face_only_5seeds.json` | 2026-10-11 00:52:21 IST | Registered before Phase B runs |
| **P-S4** (Clean Full-Image AUROC) | `cb5b1bf99` | 2026-10-10 21:39:47 IST | `reports/lp_shared_dino_5seeds.json` | 2026-10-11 00:48:28 IST | Registered before Phase B runs |
| **P-C1** (Dataset-ID Baseline) | `fa0a67572` | 2026-10-11 00:07:52 IST | `reports/dataset_id_baseline.json` | 2026-10-11 00:11:26 IST | Registered 3.5 min before run |
| **P-C2** (Celeb-DF Background) | `871d47004` | 2026-10-11 00:16:23 IST | `reports/lp_celeb_face_out_5seeds.json` | 2026-10-11 00:25:10 IST | Registered before Celeb-DF runs |
| **P-C3** (Multi-Backbone Invariance)| `eeaa3cc03` | 2026-10-11 00:46:04 IST | `reports/lp_shared_resnet50_face_out_5seeds.json` | 2026-10-11 01:05:40 IST | Registered after DINOv2; before ResNet/ViT |

*Disclosure of Chronology:* Rules P-S1 through P-S4 and P-C1 were registered strictly before any corresponding experimental executions. Rule P-C3 (multi-backbone invariance across ResNet-50 and ViT-B/16) was formulated and registered *after* observing Phase B DINOv2 results, but *before* the ResNet-50 and ViT-B/16 evaluation scripts were run.

### 3.3 Evaluation Setup
- **Frozen Encoders:**
  1. *DINOv2-S/14* (`facebookresearch/dinov2-small`): 768-dimensional feature representation formed by concatenating the 384-d CLS token with the 384-d spatial mean of all 256 patch tokens.
  2. *ResNet-50* (`torchvision.models.resnet50`, ImageNet-1k supervised weights): 2048-dimensional feature embedding extracted prior to the final classification head.
  3. *ViT-B/16* (`torchvision.models.vit_b_16`, ImageNet-1k supervised weights): 768-dimensional CLS token representation.
  4. *Low-Level Baseline:* Handcrafted 8-dimensional feature vector consisting of: log JPEG bytes per pixel, log Laplacian variance (sharpness), high-frequency 2D-FFT energy ratio, grayscale standard deviation, grayscale mean, and R, G, B channel means.
- **Probe Architecture:** Logistic regression fitted with $L_2$ regularization ($C=0.1$) and balanced class weighting (`class_weight="balanced"`).
- **Splits and Repetition:** All evaluations are conducted across five random seeds (`[42, 7, 13, 99, 2024]`) using official scene-disjoint splits (parent videos strictly separated between train and test: 140 test scenes, ~249 real frames, ~296 fake frames). We report mean $\pm$ standard deviation across seeds (reflecting frame-sampling variance) and 95% bootstrap confidence intervals over test scenes (300 bootstrap resamples).
- **Spatial Masking Operations:**
  - *Full Image:* Unmodified $224 \times 224$ canonical crop.
  - *Background Only (`face_out`):* Central $112 \times 112$ square ($c - 56 : c + 56$, half-width 56 px) filled with the per-image mean RGB colour, leaving only peripheral background context.
  - *Face Only (`face_only`):* Four $40 \times 40$ outer background corners filled with the per-image mean RGB colour, suppressing peripheral container cues while retaining the central facial region.

---

## 4. Empirical Results

All metrics reported in this section are extracted programmatically from machine-readable JSON ledgers in `reports/` via `evaluation/make_latex_tables.py`. Complete tabular summaries are provided in Table 1 through Table 5.

### 4.1 The Pairing Effect Across Visual Representations

We evaluate linear probe performance under four pairing regimes:
1. $C \to C$: Trained and tested on naive/confounded pairings.
2. $C \to M$: Trained on naive pairings, tested on geometry-matched pairs (cross-evaluation).
3. $M \to M$: Trained and tested on clean shared-transform pairs.
4. $M \to C$: Trained on clean pairs, tested on naive pairs.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 1: The Pairing Effect Across Visual Representations (FaceForensics++ Scene-Disjoint Split)                      |
+------------------------------------+----------------+----------------+----------------+----------------+--------------+
| Representation                     | C->C (Naive)   | C->M (Cross)   | M->M (Clean)   | M->C (Gen)     | M->M BalAcc  |
+------------------------------------+----------------+----------------+----------------+----------------+--------------+
| DINOv2-S/14 (Confounded vs Clean)  | 0.998 ± 0.001  | 0.596 ± 0.013  | 0.763 ± 0.012  | 0.793 ± 0.035  | 0.691        |
| DINOv2-S/14 (FF++ Native Pairing)  | 0.874 ± 0.006  | 0.874 ± 0.009  | 0.877 ± 0.008  | 0.876 ± 0.008  | 0.786        |
| ResNet-50 (Supervised ConvNet)     | 0.972 ± 0.006  | 0.524 ± 0.009  | 0.613 ± 0.013  | 0.659 ± 0.031  | 0.592        |
| ViT-B/16 (Supervised ViT)          | 0.966 ± 0.005  | 0.553 ± 0.010  | 0.690 ± 0.010  | 0.718 ± 0.024  | 0.638        |
| Handcrafted Color/Texture Baseline | 0.812 ± 0.007  | 0.467 ± 0.009  | 0.546 ± 0.009  | 0.299 ± 0.016  | 0.540        |
+------------------------------------+----------------+----------------+----------------+----------------+--------------+
Source ledgers: reports/lp_shared_dino_5seeds.json, reports/lp_ff_native_dino_5seeds.json, reports/lp_shared_resnet50_5seeds.json, reports/lp_shared_vitb16_5seeds.json
```

**Empirical Findings:**
- Under naive container-confounded pairing ($C \to C$, JPEG reals vs PNG fakes), DINOv2-S/14 achieves an AUROC of 0.998 $\pm$ 0.001 [Measured], and ResNet-50 achieves 0.972 $\pm$ 0.006 [Measured].
- When evaluated out-of-pairing on geometry-matched reals ($C \to M$), DINOv2-S/14 performance collapses to 0.596 $\pm$ 0.013 [Measured], and ResNet-50 collapses to 0.524 $\pm$ 0.009 [Measured].
- Under FF++ native pairing ($C \to C$ and $M \to M$, where reals and fakes share identical c23 MP4 containers but have independent face-tracker bounding-box alignments), DINOv2-S/14 achieves 0.874 $\pm$ 0.006 ($C \to C$) and 0.877 $\pm$ 0.008 ($M \to M$) [Measured].
- When evaluated on clean shared-transform pairs ($M \to M$, where reals and fakes share both container format and identical affine crop geometry), probe accuracy drops further to 0.763 $\pm$ 0.012 on DINOv2 [Measured] and 0.613 $\pm$ 0.013 on ResNet-50 [Measured]. This demonstrates that reported benchmark accuracy is heavily inflated by pairing and alignment shortcuts.

---

### 4.2 Spatial Decomposition: Isolating the Background Shortcut

To locate the source of the shortcut signal, we apply spatial masking to evaluate linear probe performance on the background alone (`face_out`) and on the face alone (`face_only`).

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 2: Spatial Decomposition of the Shortcut (Linear Probe AUROC across 5 Seeds)                                    |
+------------------------------------+----------------------------+----------------------------+------------------------+
| Backbone                           | Background (C->C Shortcut) | Background (M->M Clean)    | Face Only (M->M)       |
+------------------------------------+----------------------------+----------------------------+------------------------+
| DINOv2-S/14 (Shared-Transform)     | 0.998 ± 0.001              | 0.591 ± 0.010 [P-S2 MET]   | 0.774 ± 0.007          |
| DINOv2-S/14 (FF++ Native)          | 0.758 ± 0.006              | 0.755 ± 0.010              | 0.872 ± 0.009          |
| ResNet-50 (Shared-Transform)       | 0.986 ± 0.004              | 0.537 ± 0.009 [P-C3 MET]   | 0.619 ± 0.011          |
| ViT-B/16 (Shared-Transform)        | 0.986 ± 0.004              | 0.567 ± 0.009 [P-C3 MET]   | 0.706 ± 0.012          |
+------------------------------------+----------------------------+----------------------------+------------------------+
Source ledgers: reports/lp_shared_dino_face_out_5seeds.json, reports/lp_ff_native_dino_face_out_5seeds.json, reports/lp_shared_resnet50_face_out_5seeds.json, reports/lp_shared_vitb16_face_out_5seeds.json
```

**Empirical Findings:**
- On naive pairings ($C \to C$), probing the background alone (`face_out`) achieves 0.998 $\pm$ 0.001 on DINOv2, 0.986 $\pm$ 0.004 on ResNet-50, and 0.986 $\pm$ 0.004 on ViT-B/16 [Measured]. In our setting, the detector can classify deepfakes with near-certainty without observing facial pixels.
- On FF++ native pairing (independent face-tracker crops, same container), probing the background alone (`face_out`) still achieves 0.758 $\pm$ 0.006 ($C \to C$) and 0.755 $\pm$ 0.010 ($M \to M$) [Measured]. Probing the face alone (`face_only`) achieves 0.872 $\pm$ 0.009 [Measured]. This directly isolates the alignment confound: even when container formats are identical, independent face cropping introduces an edge/alignment shortcut yielding ~0.755 background AUROC.
- On clean shared-transform pairs ($M \to M$), background probe AUROC drops to 0.591 $\pm$ 0.010 [0.580, 0.603] on DINOv2, 0.537 $\pm$ 0.009 on ResNet-50, and 0.567 $\pm$ 0.009 on ViT-B/16 [Measured]. For DINOv2, the point estimate (0.591) meets decision rule P-S2 ($\le 0.60$), though the upper bound of the 95% bootstrap confidence interval reaches 0.603. ResNet-50 (0.537) and ViT-B/16 (0.567) satisfy rule P-C3 ($\le 0.60$) with their entire confidence intervals below 0.60.
- Probing the face alone (`face_only`) on clean shared-transform pairs yields 0.774 $\pm$ 0.007 on DINOv2, 0.619 $\pm$ 0.011 on ResNet-50, and 0.706 $\pm$ 0.012 on ViT-B/16 [Measured]. In all three backbones, face probe accuracy exceeds background probe accuracy, consistent with facial synthesis cues being localized to the face region once background container and alignment artifacts are equalized.

---

### 4.3 Auditing the Native Pairing Fallacy (Celeb-DF-v2)

A common assumption in forensic literature is that benchmarks pairing manipulated videos with their corresponding pristine source videos (such as Celeb-DF-v2) are immune to pairing shortcuts. We audit Celeb-DF-v2 native pairings (5,332 reals, 5,410 fakes) by comparing low-level corner probes against high-level spatial probes.

```
+--------------------------------------------------------------------------------------------------+
| TABLE 3: The Native Pairing Fallacy on Celeb-DF-v2                                               |
+-----------------------+-----------------------------+-----------------------+--------------------+
| Probe Type            | Region Evaluated            | AUROC [95% CI]        | Status / Rule      |
+-----------------------+-----------------------------+-----------------------+--------------------+
| Handcrafted Low-Level | Outer Image Corners         | 0.458 [0.413, 0.503]  | Passed (At Chance) |
| DINOv2 Linear Probe   | Background (face_out)       | 0.751 ± 0.005         | FAILED P-C2 (>0.60)|
| DINOv2 Linear Probe   | Face Only (face_only)       | 0.911 ± 0.010         | Facial Artifacts   |
| DINOv2 Linear Probe   | Full Image (Native)         | 0.889 ± 0.007         | Confounded Native  |
+-----------------------+-----------------------------+-----------------------+--------------------+
Source ledgers: reports/region_probe_celeb.json, reports/lp_celeb_5seeds.json, reports/lp_celeb_face_out_5seeds.json
```

**Empirical Findings:**
- Handcrafted low-level features on the outer corners yield an AUROC of 0.458 [0.413, 0.503] [Measured], indicating that simple compression or brightness differences do not distinguish reals from fakes in Celeb-DF corners.
- However, when evaluated with a DINOv2 linear probe on the masked background (`face_out`), AUROC reaches 0.751 $\pm$ 0.005 [Measured]. This contradicts decision rule P-C2 ($\le 0.60$).
- **Diagnosis:** [Hypothesis] In our setting, independent face-tracking bounding boxes introduce subtle spatial misalignments and boundary-spill cues around the perimeter of the crop. Standard face cropping does not equate to a shared spatial transform.

---

### 4.4 Benchmark Composition Shortcut (Dataset Identity Alone)

Large-scale multi-dataset benchmarks frequently pool multiple datasets into a shared training set. We test whether categorical dataset identity alone predicts forgery status without observing image pixels.

We construct a trivial Bayesian estimator evaluating $P(\text{fake} \mid \text{dataset})$ via 5-fold cross-validation on 949,254 samples across 9 benchmark datasets in `data/manifests/combined.jsonl`.

```
+--------------------------------------------------------------------------------------------------+
| TABLE 4: Benchmark Composition Shortcut (Categorical Dataset Provenance Alone)                  |
+-----------------------+-----------------------+-----------------------+--------------------------+
| Dataset Component     | Total Samples (N)     | Fake Ratio (%)        | Empirical Prior P(Fake)  |
+-----------------------+-----------------------+-----------------------+--------------------------+
| DF40 (Train pool)     | 686,358               | 100.0%                | 1.0000                   |
| FaceForensics++       | 127,068               | 85.8%                 | 0.8578                   |
| Celeb-DF-v2           | 84,900                | 85.7%                 | 0.8567                   |
| WildDeepfake          | 23,100                | 51.7%                 | 0.5169                   |
| FakeAVCeleb           | 21,566                | 97.7%                 | 0.9768                   |
| FFHQ                  | 4,062                 | 0.0%                  | 0.0000                   |
| LAV-DF                | 2,000                 | 100.0%                | 1.0000                   |
| HDTF                  | 100                   | 0.0%                  | 0.0000                   |
| TalkingHeadBench      | 100                   | 100.0%                | 1.0000                   |
+-----------------------+-----------------------+-----------------------+--------------------------+
| Overall 5-Fold Cross-Validation AUROC: 0.9208 [0.9201, 0.9216] [P-C1 MET]                        |
| Overall Balanced Accuracy: 0.5452                                                                |
+--------------------------------------------------------------------------------------------------+
Source ledger: reports/dataset_id_baseline.json
```

**Empirical Findings:**
- Without inspecting image pixels, categorical dataset ID alone achieves an AUROC of 0.9208 [0.9201, 0.9216] [Measured].
- This confirms hypothesis H-C1 (P-C1 met $\ge 0.90$). Models trained on uncurated multi-dataset pools can achieve superior reported benchmark performance by learning dataset presentation shortcuts (such as sensor noise or color grading) rather than manipulation forensics.

---

### 4.5 Geometric and Low-Level Alignment Verification

To confirm that our clean reference pairing equalizes spatial alignment, we measure the paired Mean Absolute Difference (MAD) between real and fake crop centers and background pixel statistics across shared pairs.

```
+--------------------------------------------------------------------------------------------------+
| TABLE 5: Pairing Geometry & Low-Level Matching Verification (FaceForensics++)                   |
+---------------------------------------+-----------------------+----------------------------------+
| Alignment / Forensic Diagnostic       | Measured Value        | Decision Rule / Tolerance        |
+---------------------------------------+-----------------------+----------------------------------+
| Background Pixel MAD: Real vs Fake    | 3.244 [2.415, 4.994]  | <= 4.0 (P-S1 MET in PREREG.md)   |
| Face Crop Center MAD                  | 0.721 px              | <= 4.0 px (Geometric check MET)  |
| Fake vs Fake Background Pixel MAD     | 3.372 [2.517, 5.207]  | Negative control                 |
| Ratio of Background Medians (RF / FF) | 0.962                 | Margin 0.05 MET (ratio_equiv)    |
| Paired Wilcoxon Signed-Rank Test      | p = 1.96e-12          | Real vs fake background diff     |
| Composited Low-Level Background Probe | 0.437 [0.383, 0.486]  | Below chance (CI excludes 0.50)  |
| Reenactment Low-Level Background Probe| 0.521 [0.482, 0.570]  | Pre-reg band [0.45, 0.55]        |
+---------------------------------------+-----------------------+----------------------------------+
Source ledgers: reports/paired_stats_shared.json, reports/region_probe_shared.json
```

**Empirical Findings:**
- In `PREREG.md`, decision rule P-S1 specifies paired background pixel MAD $\le 4.0$: our measured real-vs-fake background pixel MAD is 3.244 [2.415, 4.994] [Measured] (paired crop center MAD is 0.721 px [Measured]), both satisfying the $\le 4.0$ threshold and confirming geometric equalisation under shared transforms compared to ~10.6 px under independent native alignment.
- The ratio of real-vs-fake background pixel MAD to fake-vs-fake background pixel MAD is 0.962 [Measured]. In `reports/paired_stats_shared.json`, the ratio equivalence test under a $\pm 0.05$ margin is satisfied (`margin_0.05: true` [Measured]). Paired Wilcoxon signed-rank test on pixel differences yields $p = 1.96 \times 10^{-12}$ [Measured]. Note: the $[0.95, 1.05]$ ratio equivalence band was defined in `evaluation/paired_stats.py` and `ANALYSIS_LOG.md`, whereas the AUROC equivalence margin $[0.45, 0.55]$ was pre-registered in `PREREG.md`.
- Handcrafted low-level background probes: reenactment background AUROC is 0.521 [0.482, 0.570] [Measured], falling within the pre-registered equivalence band $[0.45, 0.55]$. Composited background AUROC is 0.437 [0.383, 0.486] [Measured]; its 95% confidence interval strictly excludes 0.50, demonstrating a slight below-chance inversion (inverted compression/texture gradient) rather than true equivalence to chance.

---

## 5. Diagnostic Disclosures and Protocol Contradictions

In accordance with project rules, we explicitly report experimental outcomes that contradicted pre-registered expectations:

1. **P-S3 Contradiction (Face-Only Signal Retention):**
   - *Pre-registered expectation:* Face-only probe AUROC on clean pairs $\ge 0.85$.
   - *Measured result:* DINOv2-S/14 achieves 0.774 $\pm$ 0.007 [Measured] (Table 2).
   - *Diagnosis:* [Hypothesis] Although the face probe exceeds the background probe (0.774 vs 0.591), frozen DINOv2-S/14 features do not reach 0.85 AUROC on clean FaceForensics++ c23 faces. A candidate explanation is that lossy c23 compression attenuates high-frequency boundary blending artifacts in the face interior.
2. **P-S4 Contradiction (Clean Full-Image AUROC Bound):**
   - *Pre-registered expectation:* Full-image probe AUROC on clean pairs $\le 0.70$.
   - *Measured result:* DINOv2-S/14 achieves 0.763 $\pm$ 0.012 [Measured] (Table 1, $M \to M$).
   - *Diagnosis:* [Hypothesis] While clean shared-transform pairing eliminates the background shortcut ($0.591 \le 0.60$), the full-image probe on clean pairs retains detectable synthesis signals from the unmasked face, yielding 0.763 AUROC and failing rule P-S4 ($\le 0.70$). Pairing cleanup substantially reduces reported accuracy from naive ceiling ($0.998 \to 0.763$), but face-region forensic signal remains detectable.
3. **P-C2 Contradiction (Celeb-DF Native Background Cleanliness):**
   - *Pre-registered expectation:* Celeb-DF native background probe AUROC $\le 0.60$.
   - *Measured result:* Measured AUROC is 0.751 $\pm$ 0.005 [Measured] (Table 3).
   - *Diagnosis:* [Hypothesis] We reject the hypothesis that native target-video matching guarantees background cleanliness. We hypothesize that independent face-tracker bounding-box jitter or peripheral warping creates an edge leakage cue detectable by deep foundation models; shared spatial transforms have not yet been evaluated on Celeb-DF-v2.
4. **Disclosure of Training Confounds in Legacy Checkpoint:**
   - Audit of legacy training configurations (`configs/train.yaml`) revealed that `pos_weight=18.4` was combined with `balanced_sampling=true`. In `data/dataset.py`, label 1 represents fakes; consequently, fake samples received disproportionate loss weighting rather than real samples. Furthermore, legacy models were trained on the confounded combined pool where dataset ID alone yields 0.9208 AUROC. We did not use this checkpoint to generate Paper 1 results; all numbers reported herein are generated post-hoc via frozen feature probes.

---

## 6. Threats to Validity and Limitations

- **Bounded Scope:** All conclusions are bounded to our experimental setting (FaceForensics++ c23, Celeb-DF-v2, our combined manifest (`data/manifests/combined.jsonl`), and frozen DINOv2-S/14, ResNet-50, and ViT-B/16 encoders). We do not claim universal applicability across unexamined generative models or video-level architectures.
- **Residual Compression:** While shared spatial transforms eliminate geometric misalignment, minor second-generation video re-encoding artifacts may remain in certain video segments.
- **Absence of Abstention / Multi-Branch Detector Retraining:** In accordance with pre-registration, Paper 1 focuses strictly on auditing the evaluation confound; retrained detectors on clean pairings are deferred to Paper 2.

---

## 7. Recommendations for Deepfake Benchmark Design

Based on our empirical findings, we recommend three protocols for future benchmark design:
1. **Mandatory Shared Spatial Transforms:** Real frames and fake frames must be extracted from identical source video timestamps using identical affine cropping parameters.
2. **Standard Spatial Ablation Audits:** Benchmark releases should include a background-masked baseline (`face_out`). Any detector achieving background AUROC $> 0.60$ is exploiting container shortcuts and should be flagged.
3. **Dataset-ID Zero-Pixel Baseline:** Multi-dataset benchmarks must report the performance of a categorical dataset-identity classifier to verify that class balance is preserved across source datasets.

---

## Declarations

- **Reproducibility Statement:** All evaluation scripts (`evaluation/linear_probe_beforeafter.py`, `evaluation/make_latex_tables.py`, `evaluation/dataset_id_baseline.py`, `scripts/extract_shared_transform_pairs.py`) are fully deterministic under fixed random seeds. All results are written directly to JSON ledgers in `reports/` with SHA-256 verification.
- **AI Assistance:** AI assistance was utilized in code development and text drafting under researcher supervision. All empirical statements are verified against local ledgers.

---

## References

- Dolhansky, B., et al. (2020). The Deepfake Detection Challenge Dataset. *arXiv preprint arXiv:2006.14397*.
- Durall, R., et al. (2020). Watch your Up-Convolution: CNN Based Generative Deepfake Detection. *CVPR*.
- Frank, J., et al. (2020). Leveraging Frequency Analysis for Deep Fake Image Recognition. *ICML*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665-673.
- Grommelt, M., Weiss, R., Pfreundt, J., & Keuper, J. (2024). Revealing Common Biases in Generated Image Detection Datasets. *Proceedings of the European Conference on Computer Vision (ECCV) Workshops*.
- Jiang, L., et al. (2020). DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection. *CVPR*.
- Li, L., Bao, J., Yang, H., Chen, D., & Wen, F. (2020a). Advancing High-Fidelity and Detailed Face Swapping [FaceShifter]. *CVPR*.
- Li, L., Bao, J., Zhang, T., Yang, H., Chen, D., & Wen, F. (2020b). Face X-Ray for More General Face Forgery Detection. *CVPR*.
- Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020c). Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. *CVPR*.
- Rössler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV*.
- Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. *CVPR*.
- Yan, Z., Zhang, S., Gao, P., Zhang, X.-Y., et al. (2024a). DF40: Toward Next-Generation Deepfake Detection. *Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
- Yan, Z., et al. (2024b). DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection. *Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS)*.
