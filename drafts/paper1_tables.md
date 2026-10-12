# Paper 1 Tables (Generated from JSON Ledgers)

## Table 1: Pairing Effect across Visual Representations

| Representation | C->C (Confounded) | C->M (Cross) | M->M (Clean Ref) | M->C (Gen) | M->M BalAcc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DINOv2-S/14 (Confounded vs Clean) | 0.998 ± 0.001 | 0.596 ± 0.013 | 0.763 ± 0.012 | 0.793 ± 0.012 | 0.691 |
| DINOv2-S/14 (FF++ Native Pairing) | 0.874 ± 0.006 (Native) | -- | -- | -- | 0.784 |
| ResNet-50 (Supervised ImageNet) | 0.972 ± 0.006 | 0.524 ± 0.009 | 0.613 ± 0.013 | 0.659 ± 0.013 | 0.592 |
| ViT-B/16 (Supervised ImageNet) | 0.966 ± 0.005 | 0.553 ± 0.010 | 0.690 ± 0.010 | 0.718 ± 0.010 | 0.638 |
| Color \& Texture Baseline | 0.812 ± 0.007 | 0.467 ± 0.009 | 0.546 ± 0.009 | 0.299 ± 0.009 | 0.540 |

## Table 2: Spatial Decomposition of the Shortcut

| Backbone | Center-Masked (C->C Shortcut) | Center-Masked (M->M Clean) | Corner-Masked (M->M) | Full Image (M->M) |
| :--- | :---: | :---: | :---: | :---: |
| DINOv2-S/14 (Shared-Transform) | 0.998 ± 0.001 | 0.591 ± 0.010 | 0.774 ± 0.007 | 0.763 ± 0.012 |
| DINOv2-S/14 (FF++ Native) | 0.758 ± 0.006 | 0.755 ± 0.010 | 0.872 ± 0.009 | 0.877 ± 0.008 |
| ResNet-50 (Shared-Transform) | 0.986 ± 0.004 | 0.537 ± 0.009 | 0.619 ± 0.011 | 0.613 ± 0.013 |
| ViT-B/16 (Shared-Transform) | 0.986 ± 0.004 | 0.567 ± 0.009 | 0.706 ± 0.012 | 0.690 ± 0.010 |

## Table 3: The Native Pairing Fallacy (Celeb-DF-v2)

| Regime / Probe | Region Evaluated | AUROC | Status / Rule |
| :--- | :--- | :---: | :---: |
| **Native Pairing (Independent Tracking)** | | | |
| Handcrafted Low-Level | Outer Corners | 0.458 [0.413, 0.503] | Passed (Chance) |
| DINOv2 Linear Probe | Background Only (face_out) | 0.751 ± 0.005 | FAILED P-C2 (>0.60) |
| DINOv2 Linear Probe | Face Region (face_only) | 0.911 ± 0.010 | Synthesis Signal |
| DINOv2 Linear Probe | Full Image (Native) | 0.889 ± 0.007 | Confounded Benchmark |
| **Clean Reference (Shared Transform)** | | | |
| DINOv2 Linear Probe | Background (face_out, M->M) | 0.604 ± 0.011 | P-C2b Boundary (0.604) |
| Low-Level Baseline | Background (face_out, M->M) | 0.507 ± 0.003 | At Chance (0.507) |
| DINOv2 Linear Probe | Cross-Transfer (C->M) | 0.532 ± 0.007 | Collapses to Chance |
| DINOv2 Linear Probe | Face Region (face_only, M->M) | 0.835 ± 0.014 | Residual Signal |
| DINOv2 Linear Probe | Full Image (M->M) | 0.806 ± 0.015 | Clean Reference |

## Table 4: Benchmark Composition Shortcut (Dataset Identity Alone)

| Dataset | Total Samples (N) | Fake Ratio (%) | Empirical Prior |
| :--- | :---: | :---: | :---: |
| df40 | 686,358 | 100.0% | 1.0000 |
| faceforensics | 127,068 | 85.8% | 0.8578 |
| celeb_df | 84,900 | 85.7% | 0.8567 |
| wild_deepfake | 23,100 | 51.7% | 0.5169 |
| fakeavceleb | 21,566 | 97.7% | 0.9768 |
| ffhq | 4,062 | 0.0% | 0.0000 |
| lav-df | 2,000 | 100.0% | 1.0000 |
| hdtf | 100 | 0.0% | 0.0000 |
| talkingheadbench | 100 | 100.0% | 1.0000 |
| **Overall 5-Fold CV AUROC** | **0.9208** [0.9201, 0.9216] | | |
| **Overall Balanced Accuracy** | **0.5452** | | |

## Table 5: Pairing Geometry & Low-Level Matching Verification

| Alignment / Forensic Diagnostic | Measured Value | Pre-Registered Threshold / Rule |
| :--- | :---: | :---: |
| Face Crop Center MAD | 0.721 px | <= 4.0 px (P-S1 MET, p_s1_center_mad_px) |
| Background Pixel MAD: Real vs Fake | 3.244 gray levels | Gray-level intensity |
| Background Pixel MAD: Fake vs Fake | 3.372 gray levels | Noise floor reference |
| Ratio of Background Medians (RF / FF) | 0.962 | Clean ([0.95, 1.05], margin 0.05 MET) |
| Composited Low-Level Background Probe | 0.437 [0.383, 0.486] | Excludes 0.50 ([0.383, 0.486]) |
| Reenactment Low-Level Background Probe | 0.521 [0.482, 0.570] | At Chance ([0.45, 0.55]) |
