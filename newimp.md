# FORGE-Lite: From 60–65% Unseen Accuracy to a Defensible Verification System

**Engineering review and improvement plan · 15 September 2026**  
**Inputs reviewed:** your complete `readme.md`; the uploaded `improve.md` was empty. No source code, model weights, split manifests, predictions, or training logs were supplied. The reported 60–65% is therefore a user-reported result, not a reproduced measurement.

> **Seedha answer:** Har arbitrary unseen video ko pixels se 100% real/fake classify karne ki guarantee nahi mil sakti. Lekin iska matlab project fail nahi hai. Strong solution hai: **better-generalizing detection + source verification where available + honest uncertainty + stronger checks before high-stakes actions.**

## 1. The decision I would make

**Do not expand the six-branch architecture yet.** First determine whether the current failure is ranking, thresholding, preprocessing, or training-data bias. Then build the smallest model that survives genuinely unseen domains.

My recommended order:

1. **Audit the 60–65% measurement and data pipeline.**
2. **Fix source/content/codec shortcuts; establish a small foundation-model baseline.**
3. **Test source-paired training and conservative adaptation.**
4. **Add modern manipulation diversity and one small temporal module.**
5. **Calibrate decisions, including an uncertain state; measure error versus coverage.**
6. **Build provenance/capture verification as a separate assurance layer—not another fake-score feature.**

Your README already contains several sound ideas. Its biggest weakness is not a missing exotic model; it is that a very large proposed architecture sits on top of unresolved data, evaluation, and output-contract assumptions.

**Research scope:** I reviewed primary conference papers, current preprints, benchmark documentation, and official C2PA/NIST guidance available by the date above. This is a targeted international literature review, not a claim to have evaluated every detector worldwide. None of the recommendations below is a promised accuracy gain.

---

## 2. Why 100% correctness is not the right engineering requirement

### Human realism and machine detectability are different

A video that looks perfect to a person may still contain machine-detectable traces. Conversely, a suspicious-looking genuine video can be degraded by lighting, compression, camera processing, or bad tracking. **“Humans cannot tell” does not establish that detectors cannot tell.**

The universal guarantee fails for a deeper reason: the observation does not necessarily identify its history. If authentic and synthetic processes can produce indistinguishable detector inputs, a classifier cannot always recover which process occurred. Compression, resizing, and recapture can remove evidence rather than merely make a model work harder.

For a binary problem with equal class priors, the optimal achievable accuracy is:

`best accuracy = (1 + total-variation distance between the two input distributions) / 2`

Disjoint distributions permit perfect classification in principle; identical distributions permit only 50%. Overlap creates irreducible error. This is a general statistical fact, **not a claim that today's real and fake video distributions are identical**.

Also separate four questions:

- Was this media manipulated or generated?
- Did a trusted device capture these bytes through an approved path?
- Is the visible person live and the claimed identity?
- Is this person authorized to approve the requested action?

One face classifier cannot answer all four.

### What “100%” commonly hides

- **100% on a finite test set:** not proof about all future inputs.
- **A score of 1.00:** model output, not guaranteed truth.
- **100% after rejecting difficult cases:** incomplete unless acceptance coverage and uncertainty are reported.
- **Valid digital signature:** evidence about a signed record, not proof that the depicted event happened as claimed.

The useful requirement is: **under a defined threat model, minimize harmful errors at a measured acceptance rate, and require more evidence when necessary.**

---

## 3. Diagnose the 60–65% before retraining

These are hypotheses to test—not findings about code I have not seen.

| Observation | Likely explanations to investigate | First check |
|---|---|---|
| High AUROC, only 60–65% accuracy | Wrong threshold, class-prior shift, calibration mismatch | Score distributions and validation-selected operating point |
| AUROC also around 0.60–0.65 | Weak transfer, dataset shortcuts, insufficient manipulation coverage | Per-domain/per-generator ranking and balanced controls |
| Strong frame results, weak video results | Aggregation, sparse sampling, short manipulated segments | Same predictions under fixed video aggregation |
| Failure mainly after recompression | Codec shortcuts or erased evidence | Matched real/fake re-encoding sweeps |
| Failure after changing detector/crop code | RGB/BGR, normalization, alignment, interpolation or crop mismatch | Golden-input preprocessing tests |
| Near-constant predictions | Label mapping, logits/sigmoid handling, checkpoint or evaluation-mode bugs | Tiny inspected batch and saved logits |
| Excellent familiar data, weak genuinely new sources | Identity/background/source or generator overfitting | Source-disjoint and generator-disjoint evaluation |

### Required diagnostic export

Save one row per evaluated video, including:

`video_id, source_group_id, identity_group_if_known, dataset, generator_family, label, fake_score, predicted_label, threshold, face_count, usable_frames, duration, face_size, quality_flags, preprocessing_version, model_version`

Add split membership and sampling/aggregation configuration. Preserve clip-level predictions for debugging, but **do not treat correlated frames as independent test samples**.

Recompute:

- video-level AUROC, AP, balanced accuracy, confusion matrix;
- fake recall/TPR and genuine-video false-positive rate;
- accuracy at a threshold fixed outside the final test set;
- per-domain, per-generator, duration, compression, face-size and lighting slices;
- majority-class baseline and confidence intervals grouped by independent source.

If a test set is 65% fake, predicting “fake” for everything already gives 65% accuracy. That is an illustration, **not an assumption about your dataset**.

**Decision gate:** do not buy model complexity until this audit identifies what is actually failing.

---

## 4. What the research supports—and what it does not

| Evidence | Useful lesson for FORGE-Lite | Boundary |
|---|---|---|
| **GenD, WACV 2026; revised preprint May 2026** [1] | Test LayerNorm-only adaptation, normalized representations, metric learning and source-paired real/fake training | Not simply “unfreeze LayerNorm and generalization is solved.” Its large-backbone results do not transfer automatically to DINOv3-S |
| **Forensics Adapter, CVPR 2025** [2] | Small task adapters can preserve useful foundation features while learning facial-forgery cues | Its local face-forgery scope is not arbitrary whole-face/full-scene generation |
| **Facial Component Guided Adaptation, CVPR 2025** [3] | Facial-region guidance and spatial/temporal side modules are credible experiments | Video AUROC under that paper's protocol is not your unseen-data accuracy |
| **Self-Blended Images, CVPR 2022** [4] | Cheap synthetic training diversity can reduce dependence on a few known generators | Blending artifacts are not universal signatures of modern synthesis |
| **FreqDebias, CVPR 2025** [5] | Frequency shortcuts deserve explicit controls and consistency tests | A custom codec bank is an engineering extrapolation, not a proven universal fix |
| **DF40, NeurIPS 2024** [6] | Evaluate more than FF++'s four methods; DF40 covers 40 techniques across broader forgery types | Separate image and video tasks; don't pretend every technique supplies temporal examples |
| **Deepfake-Eval-2024, revised May 2026** [7] | In-the-wild evaluation shows substantial deterioration versus older academic benchmarks | Multi-modal, diverse media requires a documented face-relevant subset |
| **Synthetic Generation of Face Videos With Plethysmograph Physiology, CVPR 2022** [8] | Synthetic videos can contain plausible physiological signals | Neither presence nor absence of pulse is an authenticity certificate |
| **C2PA and NIST** [9–13] | Use provenance, protected capture, liveness and layered controls for bounded assurance | These do not establish universal semantic truth |

I deliberately do **not** combine paper scores into a “world-best” leaderboard: frame/video units, training data, target datasets, preprocessing, modalities and tuning protocols differ.

---

## 5. The detector I would actually build on your RTX 4060 8GB

### Phase A: a deliberately small core

```text
Original video
  -> deterministic decoding, face tracking, timestamps and quality checks
  -> semantic face crops
  -> DINOv3 ViT-S/16, or DINOv2 ViT-S/14 after license/compatibility review
  -> normalized features + small classifier
  -> fixed, validated video aggregation
  -> score + quality flags + uncertain/no-evidence/suspicious decision
```

Start with:

- **B0:** a simple CNN baseline under identical preprocessing;
- **B1:** frozen foundation encoder + trained linear head;
- **B2:** same encoder with LayerNorm-only adaptation;
- **B3:** small adapter/LoRA only if B2 leaves a demonstrated gap.

A frozen encoder plus trained classifier is a **linear probe**, not zero-shot detection.

**Important memory detail:** cached embeddings are valid for training a head or temporal module only while the encoder is frozen. Once LayerNorm/adapters inside the encoder train, embeddings change and backpropagation needs encoder activations. A tiny trainable-parameter count does **not** imply negligible VRAM. Use small microbatches, mixed precision, gradient accumulation/checkpointing where supported, and measure memory. Re-cache after adaptation is frozen.

### Phase B: fix the learning problem

#### 1. Source-matched supervision

For each fake with known ancestry, include its genuine source counterpart. Keep all related originals, derivatives, donor/target identities where available, and codec variants within the appropriate split group.

- Never pair unrelated real and fake videos just because their indices match.
- Document when source or identity metadata is unavailable.
- If strict identity grouping conflicts with a benchmark's official split, report a separate protocol rather than claiming both.
- If using a metric loss, specify the actual positive/negative relationships. **Source pairing does not mean pulling genuine and fake embeddings together as if they shared a label.**

The objective is to make content/source less predictive of the class, not to erase the manipulation distinction. [1]

#### 2. Class-balanced processing controls

Use the same distribution of resolutions, codecs, crop settings and augmentations for real and fake examples. Pair benign processing variants where useful. Do not let “all real videos are clean; all fakes are compressed” become the task.

Test a **quality/metadata-only classifier**. If it predicts the label strongly, inspect dataset confounds; that result is a warning, not independent forgery evidence.

#### 3. More relevant diversity, not just more frames

Cover face swap, reenactment/lip editing, and whole-face generation separately. Add consented/licensed modern examples and held-out generator families. Include hard genuine controls: low light, heavy codecs, varied cameras, occlusion, skin tones and legitimate filters consistent with your label policy.

Use DF40 for broader technique coverage where the media type fits. Treat AIGVDBench/FVBench as **secondary, curated face-visible generation/editing tests**, not ready-made face-manipulation coverage. [6, 15, 16]

SBI is a low-cost optional augmentation. If adding image-level synthetic blending to a video model, do not introduce unnatural independent frame-to-frame augmentations and then reward the model for detecting augmentation flicker. [4]

#### 4. Label-preserving codec consistency

Train on realistic video-level transformations, not only JPEG augmentation after face extraction. Encourage stable predictions across transformations **that preserve usable evidence**. If extreme degradation destroys the evidence, the correct behavior may be uncertainty—not forced agreement with a confident clean-clip score.

### Phase C: add one component at a time

1. **Small temporal head:** start with a small GRU/TCN over features from a frozen adapted encoder. Record real timestamps and frame spacing. Compare with the frame-only model under the same sample budget.
2. **Residual branch:** add only if it brings complementary cross-domain benefit. Preserve decoded evidence; avoid introducing unnecessary JPEG/WebP recompression before SRM/FFT.
3. **Quality-aware fusion:** test whether it beats a simple combination. Use missing-modality training and inspect gates for source/codec shortcuts.
4. **rPPG and audio:** later experiments, not required components. Remove them if benefits disappear under realistic quality or demographic slices.
5. **Localization/adversarial training:** separate later milestones. Neither fixes an untrustworthy baseline by itself.

Do not assume branch errors are independent. Six models learning the same source shortcut can agree confidently and still be wrong.

---

## 6. Corrections needed in your README

| Priority | Current issue | Required correction |
|---|---|---|
| **P0** | Select adaptation on cross-domain scores while the same target datasets are described as locked tests | Create development domains, a calibration partition, and untouched final target domains. Any target used for selection is development data thereafter |
| **P0** | Missing optional rPPG/audio sometimes forces whole-system abstention | Missing optional evidence should disable that branch. Reject the whole input only when core evidence or the applicable task policy is inadequate |
| **P0** | FFHQ listed as an rPPG baseline | FFHQ contains still images. Use real temporal recordings with reference PPG/ECG for pulse validation |
| **P0** | Risk of feeding the short temporal window into pulse analysis | Sixteen consecutive frames at 25–30 fps cover only about 0.5–0.6 seconds. rPPG needs its own contiguous track; validate duration/quality requirements empirically |
| **P0** | Lossy aligned crop cache proposed for all branches | Keep branch-appropriate inputs: semantic normalization must not silently alter physiological color signals or forensic residuals |
| **P1** | Max/top-10% pooling treated as intrinsically better | False positives may grow with video length/window count. Compare fixed-budget, quality-aware and length-calibrated aggregation |
| **P1** | `calibrated_risk`, `uncertainty`, `evidence_coverage` lack definitions | Specify formulas, calibration population, validity conditions and failure behavior; otherwise omit these numbers |
| **P1** | “Evidence fusion” presented as fundamentally outside discriminative learning | A supervised neural fusion classifier still learns a statistical decision rule. Naming branches “physics” or “evidence” does not create a guarantee |
| **P1** | Large-backbone paper findings used to justify small-backbone results | Treat DINOv3-S adaptation as a new experiment, not a reproduced GenD result |
| **P1** | WildDeepfake dataset units unclear | 3,805 real/3,509 fake are **face sequences**, not counts of independent source videos [17] |
| **P2** | Foundation linear probe called zero-shot | Rename it; task-label training is not zero-shot |
| **P2** | Numerical targets and “approved” wording could read as validation | Label targets as proposed, evidence pending. Architecture and publication novelty remain conditional |

Also review the exact DINOv3 model/weight license before distribution. The small backbone exists, but the custom license is not a blanket permissive-open-source license. [18]

---

## 7. The stronger solution: separate detection from verification

Your original v1 excludes legal provenance verification. The following is a **deliberate product-scope extension**, not evidence that the existing face-only detector has improved.

### Mode 1 — Arbitrary uploaded/internet video

You cannot retroactively create trusted capture history.

Use:

- forensic detector and explicit uncertainty;
- original-file request when a copy is badly processed;
- existing Content Credentials/signature checks, if present;
- source/context corroboration and human review when warranted.

Possible detector messages:

- **Suspected manipulation** — evidence exceeds a validated threshold;
- **No manipulation detected** — not a claim of verified authenticity;
- **Insufficient evidence** — cannot support a dependable decision.

Keep provenance separate:

`validation_state = valid | invalid | missing | unsupported | unavailable`

`signer_trust = trusted | untrusted | unknown`

Report applicable capture/edit assertions and policy checks independently. A valid credential may explicitly describe **AI-generated media**. Missing credentials do not make a genuine video fake. Invalid credentials mean provenance validation failed—not necessarily that the facial content was generated. [9, 10]

### Mode 2 — Video captured through your own application

Here you can improve assurance by controlling the workflow:

1. Server issues a fresh, short-lived, session-bound challenge.
2. Approved capture software/device binds the session challenge and media to a verifiable capture record.
3. Where supported, use hardware-protected signing keys and device/capture attestation.
4. Verify freshness, media binding, signer/device trust, revocation and protected transport.
5. Add liveness/PAD and anti-injection checks appropriate to the threat model.
6. Verify identity and authorization separately when required.
7. For consequential actions, obtain independent confirmation through an already-trusted channel—not contact details supplied in the suspicious video/message.

A random blink/head-turn instruction alone is not robust proof: modern real-time systems may respond. A nonce proves session freshness/secret possession only to the extent the protocol and trusted components enforce it; it does not prove human identity or scene truth. [12, 13]

**Residual risks remain:** compromised devices/keys, capture software signing untrusted input, cameras filming a screen, staged genuine scenes, coercion, and broken trust configuration. Secure capture gives stronger bounded claims, not universal certainty.

### Mode 3 — High-stakes transaction

For payment approval, account recovery or identity-sensitive decisions, **never let a face fake-score be the sole authorization control**. Combine authenticated channels, transaction-specific approval, appropriate identity proofing and independent confirmation. The business goal is preventing an unauthorized action, not winning every visual guessing contest.

---

## 8. Evaluation that makes stronger claims credible

### Separate two experiments

**Research generalization track:** restricted, documented training sources; held-out manipulation families/development domains; freeze all choices before final external testing. Do not add target-domain training samples and continue calling the result the original FF++-only zero-shot transfer experiment.

**Deployment track:** broader consented/licensed training distribution, realistic prevalence, time-separated and source-disjoint future tests, monitored model versions. Any retraining or recalibration creates a new version requiring fresh validation.

### Minimal ablation sequence

| Run | Change | Question |
|---|---|---|
| E0 | Current model with audited evaluation | What does 60–65% actually measure? |
| E1 | Frozen small foundation encoder + linear head | Does a simpler representation transfer better? |
| E2 | LayerNorm-only adaptation under matched data/protocol | Does conservative tuning preserve useful generality? |
| E3 | Source-pairing ablation | Is source/content shortcut reduction helping? |
| E4 | Class-balanced codec bank + consistency | Does performance survive realistic processing? |
| E5 | SBI or added modern-generator training diversity, separately | Does new diversity improve held-out families? |
| E6 | Small temporal head on the winning core | Is there useful additional temporal evidence? |
| E7 | Residual/fusion addition only if justified | Does complexity beat the strongest simpler system? |

Choose on **development-domain** results, considering both macro and worst-domain performance. Run multiple seeds for finalists; use paired source-group bootstrap comparisons where appropriate. Final test datasets are not a repeatedly consulted leaderboard.

### Report reliability, not just AUC

At minimum report:

- video AUROC/AP and accuracy at the frozen threshold;
- TPR at prespecified FPR budgets, with uncertainty intervals;
- per-domain and condition performance, including worst-domain results;
- calibration metrics and the population used for calibration;
- **coverage versus error**, with class-specific accepted errors and abstention rates;
- missing-face/unsupported-input rates over **all attempted inputs**;
- peak VRAM and measured end-to-end latency.

Temperature scaling, ensemble disagreement, OOD scores and conformal methods are not future-generator guarantees. Standard conformal validity depends on exchangeability; distribution-shift extensions require their own assumptions. [14]

### Two sanity checks for “high accuracy”

**Low prevalence:** imagine 10,000 videos, 1% actually fake, 90% fake recall and 1% false-positive rate. The model flags about 90 true fakes and 99 genuine videos. Only about **48% of flags are correct**. This hypothetical example shows why prevalence and false positives matter.

**Zero observed errors:** with zero errors among 300 independent relevant test examples, a one-sided 95% binomial upper error bound is roughly 1%; with 3,000 it is roughly 0.1%. These are not zero-risk certificates and do not survive arbitrary population shift. Thousands of correlated frames do not equal thousands of independent cases.

Never improve headline selective accuracy by silently excluding almost everything. Publish the coverage and failure denominator.

---

## 9. Practical milestones and stop rules

### Milestone 1 — Establish truth about the current system

Deliver: prediction export, inspected failure examples, split audit, identical preprocessing checks, metric report. No new branches.

**Stop rule:** if evaluation or preprocessing is broken, fix that before interpreting architecture gains.

### Milestone 2 — Establish the strongest inexpensive baseline

Deliver: reproducible E1–E4 comparisons, resource measurements, development-domain selection log. Keep train/calibration/final-test boundaries explicit.

**Stop rule:** if paired/debiased training does not help, inspect family coverage and genuine-video controls rather than assuming a larger backbone is the answer.

### Milestone 3 — Earn each additional component

Deliver: modern-family tests, temporal ablation, optional residual ablation, all with matched compute/input budgets where practical.

**Stop rule:** remove a branch that lacks repeatable benefit or materially worsens false positives, worst-domain performance or resource use.

### Milestone 4 — Make decisions safer

Deliver: documented score semantics, frozen thresholds, coverage-risk results, provenance displayed separately, and an escalation path. Prototype controlled-capture verification only as a separately scoped extension.

**Stop rule:** if required risk at acceptable coverage is not met, narrow supported conditions or require more evidence. Do not simply relabel a high score as “verified real.”

### Maintenance

Keep a versioned challenge set of newly verified failures, including difficult genuine videos. Revisit old domains after each update to detect regression. Retire any challenge set used for tuning from the untouched-test role, and replenish future tests. Use controlled retention/access for face data, licenses/consent, and removal procedures; do not treat scraped faces as automatically reusable training material.

---

## 10. Final recommendation

**For your research project:** prioritize a clean source-paired, codec-controlled, small-foundation-model baseline with honest held-out evaluation. Add temporal evidence only when the experiment earns it. Your strongest contribution could be reliable behavior under shift at consumer-GPU cost—but novelty and superiority must be established empirically.

**For a real product:** build two clearly labeled layers:

> **FORGE Detect:** statistical evidence of manipulation, with uncertainty.  
> **FORGE Verify:** provenance/capture-policy evidence where available, with explicit trust assumptions.

**For the 100% requirement:** replace “always decide correctly” with “never represent unsupported authenticity as verified, and never authorize a critical action from a face score alone.” That is a sound design policy—not a promise that software cannot fail.

**Next implementation input needed:** your actual training/evaluation code and configuration. With those, the first task is tracing the reported 60–65% to its metric, data split, preprocessing and failure slices—not adding another model based only on this PRD.

---

## Primary sources

1. **GenD — Deepfake Detection that Generalizes Across Benchmarks.** [WACV 2026 record](https://wacv.thecvf.com/virtual/2026/poster/457); [revised primary paper, May 2026](https://arxiv.org/html/2508.06248v4). LN adaptation, normalized/metric learning and source-pairing evidence; larger encoders than the proposed small backbone.
2. **Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection**, CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.html).
3. **Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model**, CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Han_Towards_More_General_Video-based_Deepfake_Detection_through_Facial_Component_Guided_CVPR_2025_paper.html).
4. **Detecting Deepfakes With Self-Blended Images**, CVPR 2022. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html).
5. **FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing**, CVPR 2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html).
6. **DF40: Toward Next-Generation Deepfake Detection**, NeurIPS 2024. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/34239f60eca7ce9bee5280aaf81362d8-Abstract-Datasets_and_Benchmarks_Track.html).
7. **Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024.** [Primary paper, v5, May 2026](https://arxiv.org/abs/2503.02857v5). Evidence of substantial in-the-wild deterioration; not a source for promised project gains.
8. **Synthetic Generation of Face Videos With Plethysmograph Physiology**, CVPR 2022. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Synthetic_Generation_of_Face_Videos_With_Plethysmograph_Physiology_CVPR_2022_paper.html).
9. **C2PA v2.4 official documentation.** [Explainer](https://spec.c2pa.org/specifications/specifications/2.4/explainer/Explainer.html); [technical specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html).
10. **C2PA v2.4 Security Considerations.** [Official threat model](https://spec.c2pa.org/specifications/specifications/2.4/security/Security_Considerations.html). Includes signing-key compromise, misuse and provenance stripping.
11. **NIST AI 100-4: Reducing Risks Posed by Synthetic Content**, 2024. [Report](https://doi.org/10.6028/NIST.AI.100-4). Transparency, provenance, watermarking, detection and defense-in-depth limitations.
12. **NIST SP 800-63A-4: Identity Proofing and Enrollment**, 2025. [Guidance](https://pages.nist.gov/800-63-4/sp800-63a.html). Presentation/injection distinctions and layered controls; not a universal video-authenticity standard.
13. **NIST FATE Part 10: Passive Software-Based PAD**, 2023. [Evaluation](https://doi.org/10.6028/NIST.IR.8491); [NIST challenge-response definition](https://csrc.nist.gov/glossary/term/challenge_response_protocol). PAD study does not evaluate all injection or active-capture threats.
14. **Barber, Candès, Ramdas and Tibshirani: Conformal Prediction Beyond Exchangeability.** [Primary paper](https://www.stat.berkeley.edu/~ryantibs/papers/nexcp.pdf). Assumptions behind statistical coverage guarantees.
15. **AIGVDBench: Your One-Stop Solution for AI-Generated Video Detection**, January 2026. [Paper](https://arxiv.org/html/2601.11035v1). Broad generation benchmark; face subset requires explicit curation.
16. **FVBench**, CVPR 2026. [Final conference record](https://cvpr.thecvf.com/virtual/2026/poster/39030). Broad real/edited/generated-video benchmark; do not infer face-specific coverage from total scale.
17. **WildDeepfake.** [Primary paper](https://arxiv.org/html/2101.01456v2). Face-sequence counts and in-the-wild scope.
18. **DINOv3 official model card and license.** [Model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md); [license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
19. **DeepfakeBench**, NeurIPS 2023. [Conference record](https://neurips.cc/virtual/2023/poster/73502). Standardized preprocessing, implementations and evaluation protocols.

**Evidence boundary:** literature findings, engineering proposals, user-reported measurements and verified implementation results are different categories. This report contains the first two and your reported 60–65% result. No FORGE-Lite model was trained or benchmarked during this review.
