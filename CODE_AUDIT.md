# FORGE-Lite: Code-Level Audit and Repair Order

**15 September 2026 · Follow-up to the research improvement plan**

> **Bottom line:** There are concrete implementation and measurement problems in the pasted code. Fix and verify these before interpreting 60–65% as a limit of your architecture or of deepfake detection. The exact cause of that reported score is not yet established.

## What was actually inspected and tested

- Inspected the nine pasted Python modules, not a checked-out repository.
- Ran **14 isolated checks** on the excerpts, mathematical properties, and explicitly marked test doubles. All confirmed the behavior they were designed to examine; they are **not 14 failed project unit tests**.
- Wrote small repair helpers and ran **10 helper tests; all passed**.
- Used CPU PyTorch. No model training, real face detection, real-video inference, checkpoint evaluation, GPU validation or dataset benchmark was performed.
- Your advertised 36-test suite was not supplied or executed.

The missing `models/forge_lite.py` is particularly important: it connects branches to calibration, optional modalities and final decisions. Other missing pieces include phase 1–4 trainers, config, data loader/splits, `predict_video`, `evaluate_cross_domain.py`, checkpoints and prediction logs. Some findings below can be resolved by those omitted files; others are explicit in the pasted implementation.

## 1. Triage: accuracy bugs versus reliability bugs

| Priority | Finding | Evidence level | Possible impact |
|---|---|---|---|
| P0 | CPU-only inference skips checkpoint loading | Confirmed control flow; reproduced with explicit stubs | Evaluates default/initial task weights instead of the requested checkpoint |
| P0 | Phase 5 constructs a fresh model; no explicit earlier-phase weight loading is shown | Handover unverified; constructor/other trainers omitted | If no internal loader exists, freezes the wrong branch states |
| P0 | Padded-image landmarks are applied to the original image | Confirmed control flow; reproduced with stubs | Incorrect face alignment on fallback inputs |
| P0 | No-face input is resized and sent to a face classifier | Confirmed control flow; reproduced with stubs | Unsupported inputs receive ordinary classifications |
| P1 | Phase 5 does not activate codec or metric loss; temporal loss is unused | Confirmed call graph and loss execution | Advertised generalization training is absent from this phase |
| P1 | Uncertainty head receives no gradient from shown objective | Confirmed with autograd | Unvalidated abstention and confidence reporting |
| P1 | Availability counts semantic/forensic/temporal as always present | Confirmed numerical check | Minimum coverage is 0.6; a 0.3 coverage gate never rejects |
| P1 | Image-only, accepted-only benchmark is treated too broadly | Confirmed evaluator semantics | Does not establish full unseen-video performance |
| P1 | Frozen parameters remain in train mode | Confirmed PyTorch behavior; actual branch layers omitted | BN statistics/dropout can change frozen feature behavior |
| P2 | ECE omits probability-zero samples and can crash | Confirmed numerical check | Underreported calibration error or failed evaluation |

**Do not add up these findings as explanations for a particular accuracy loss.** Some alter predictions; some alter rejection/measurement; some would stop execution entirely. Their relevance depends on which path produced your 60–65%.

---

## 2. Checkpoint loading must be fixed first

### Confirmed CPU-only bug

In `service/inference.py`:

```python
self.model = FORGELite()
if torch.cuda.is_available():
    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
```

On a CPU-only host, this never loads the requested file. It can appear to run successfully even when that checkpoint is absent. A constructor might initialize a pretrained backbone, but that is **not** loading your trained detector checkpoint.

**Fix:** always load the checkpoint, independent of CUDA availability. Missing/incompatible weights must be fatal initialization errors, not a silent fallback to an unfitted model. Construct with the exact training config, load strictly, then move to the selected device and set inference mode.

The supplied `reference_fixes.py::load_raw_checkpoint` implements this for the **raw state-dictionary format shown in your trainer**. It is not a universal loader for unknown checkpoint formats. Its tests verify exact CPU weight restoration and rejection of missing/mismatched files.

Also record absolute checkpoint path, file hash, training/config version and load success in each evaluation run. The current working-directory-relative `outputs/checkpoints/best.ckpt` can point to a different or stale artifact when commands run from different directories.

### Phase-to-phase transfer: highest-priority missing proof

`run_training.py` calls five functions in sequence, but passes no model or checkpoint between them. `run_full_fusion()` constructs `FORGELite(...)` anew and immediately freezes most parameters. No explicit `load_state_dict` appears before freezing.

This does **not** prove handover is absent: the omitted constructor or trainers might load internally. But calling phases sequentially does not itself transfer weights.

Require a handover record for every phase:

1. Exact input checkpoint(s) and config.
2. Explicit key mapping if branch checkpoints use different prefixes.
3. Assertions for expected keys/shapes; no unexplained `strict=False`.
4. Equality/hash checks showing loaded branch tensors match saved tensors.
5. Trainable parameter list and output checkpoint identity.

If no internal loader exists, Phase 5 currently trains fusion over pretrained/default or freshly initialized branches rather than the intended phase-trained encoders. That would be a major implementation failure—not evidence that the staged architecture reached its scientific limit.

---

## 3. Your actual Phase 5 objective is much smaller than advertised

The trainer passes only:

```text
fake_logit, labels, pred_mask, gt_mask, mask_available
```

Therefore, in the shown path:

- classification loss is active;
- localization is active only when both predicted and ground-truth masks exist;
- codec consistency is inactive;
- source/class metric loss is inactive;
- `frame_scores` is never used by `ForgeLoss`;
- `temporal_weight` is stored but never read;
- there is no uncertainty objective.

The isolated no-mask run returned only `classification` and `total`. This says nothing about losses that might run in missing earlier phases.

### Repair the gradient path, not just the argument list

Simply passing frozen semantic embeddings into a metric loss is insufficient. If those embeddings cannot backpropagate to any trainable parameter, the term cannot train the intended representation.

- Representation/metric learning belongs in a phase where the relevant encoder/adapters/projections are trainable.
- Fusion-stage codec consistency can compare predictions from clean and transformed inputs, with a gradient path to fusion.
- Log each active term's value, valid-example count and gradient contribution to intended parameters.
- A zero mask count must be reported as “localization supervision unavailable,” not evidence that a localization head trained.
- Do not add indiscriminate frame-score smoothing: real manipulation boundaries may genuinely change over time.

### Two loss-definition corrections

**Focal alpha:** `self.alpha * (1 - pt)**gamma` multiplies both classes by 0.25. It is a globally scaled focal loss, not class-balanced alpha weighting. It is not inherently invalid, but it does not fix class imbalance and changes classification's strength relative to auxiliary losses. Start with ordinary BCE as a clear baseline; if class-balanced focal is justified, use `alpha_t = alpha*y + (1-alpha)*(1-y)` and validate the choice. An alpha of 0.25 downweights positive/fake examples when fake is label 1; it is not an automatically correct default.

**SourcePairMetricLoss:** it receives embeddings and authenticity labels only. It clusters same-label real/fake examples; it does not inspect source IDs. A separate source-paired sampler could exist, but the loss alone does not establish identity invariance. Audit the sampler, ancestry manifest and split grouping. Do not treat genuine/fake counterparts as the same class merely because they share a source.

**Codec direction:** the augmented view is detached, so consistency gradients go to the clean view. This is a deliberate one-sided objective, not inherently an error. If the intended design is “clean teacher supervises degraded student,” reverse the stop-gradient direction. Whichever design is used, retain classification supervision and do not force confidence when extreme degradation destroys evidence.

---

## 4. Uncertainty and evidence coverage currently do not mean what the UI suggests

### Uncertainty head: no learning signal in the shown objective

```python
self.uncertainty_head = nn.Sequential(nn.Linear(..., 1), nn.Softplus())
```

A Softplus output is just a nonnegative number. It is not automatically uncertainty, calibration or probability of error.

Autograd on the pasted fusion plus shown classification objective produced:

```text
fake_logit_head.weight.grad: present
uncertainty_head weight/bias gradients: absent
```

Shared fusion features can change as classification trains, so the uncertainty output can change even while its own head parameters receive no training. That does **not** make the output a learned uncertainty estimate. Prior pretraining/loading could change this conclusion about initialization, but none was supplied.

Do not expose this head as validated uncertainty or use its arbitrary scale to justify `max_uncertainty=0.35`. Replace the policy in a controlled development experiment with a defined confidence/uncertainty method and validated coverage-risk thresholds. Do not simply remove a safety gate in deployment without revalidation.

### MC-dropout is not demonstrated

The service sets `eval()` and makes one visible forward pass. The pasted fusion is deterministic on repeated `eval()` calls. If `FORGELite.forward()` performs multiple passes and selectively enables dropout, that must be inspected; the supplied code does not show it.

A proper MC-dropout implementation needs multiple passes, selected dropout layers enabled, BatchNorm kept in evaluation mode, a stated aggregation/dispersion measure and empirical validation. Dispersion still is not a universal OOD guarantee.

### Coverage gate is mathematically ineffective in ordinary shown calls

The first three branch flags are hardcoded to one:

```text
availability = [1, 1, 1, rppg_available, av_available]
coverage = mean(availability)
```

So coverage can only be approximately 0.6, 0.8 or 1.0. It cannot fall below `min_coverage=0.3`.

For a single image, a temporal encoder may produce a vector, but there is no measured inter-frame evidence. A computed vector is not the same as valid evidence.

**Fix:** distinguish branch computational availability, task applicability and input quality. Pass a true temporal-validity flag; reject invalid core face inputs upstream. Keep missing optional audio/rPPG from automatically invalidating otherwise usable visual evidence. A modality count should not be presented as a probability of correctness.

Also sanitize missing features **before** projecting them. `NaN * 0` remains NaN: a synthetic unavailable-rPPG/NaN test poisoned the final fake logit despite `rppg_avail=0`. This is a robustness defect if upstream uses NaN placeholders; the actual upstream behavior is unknown.

### Disagreement is also unverified

The fusion output returns no per-branch predictions or disagreement value, while abstention requires one. Different embedding vectors or branch weights are not automatically comparable probability estimates. Inspect the missing wrapper; if branch disagreement is claimed, define and validate it explicitly.

---

## 5. Fix the inference/preprocessing contract

### Padded coordinates belong to the padded image

The fallback detects landmarks on a 50-pixel-padded image, then aligns the original image using those coordinates. The isolated control-flow test confirmed that mismatch.

Either:

- align the same padded image and record that fallback; or
- subtract the padding offset from landmarks and handle boundaries before aligning the original.

Use a regression test that checks actual crop coordinates. Retraining will not repair a deterministic coordinate-system error reliably.

### No face should not silently mean whole-image face inference

After both detection attempts fail, the code resizes the entire image to 224×224 and sends it through the face pipeline. For this face-scoped detector, return `insufficient_evidence` with reason `no_face`, unless you have a separately trained and evaluated whole-image fallback.

Specify a policy for multiple faces; blindly using `landmarks[0]` is not a verified dominant-track policy. Check BGR/RGB conversion, normalization, crop geometry and interpolation against training. Their correctness cannot be verified without `frames_to_tensor` and training preprocessing.

### Consistent error schema

Unreadable images return only `decision` and `reason`. The benchmark immediately indexes `calibrated_risk` and `uncertainty`, causing a `KeyError`.

Return a stable schema with unavailable scores set to `None`, retain every attempted input in the evaluation table, and handle missing scores explicitly. **Do not fabricate a score of zero or 0.5** to make metric code run.

The helper `abstained_result()` supplies a consistent unavailable-result structure, but the evaluator must also be changed to accept it.

---

## 6. The presented benchmark does not measure the full claimed task

`test_final_dataset.py` calls `predict_image()` on StyleGAN3/authentic images. The image path passes a tensor with **T=1**. This is a valid possible image benchmark, but it is not a temporal face-manipulation video benchmark. Training at T=8 and inference at T=1 can also change the temporal feature distribution even if the model runs without a shape error.

Evaluate separately:

1. **Image authenticity:** image-trained or validated image-mode head; temporal/rPPG/AV marked unavailable as appropriate.
2. **Face-manipulated video:** real tracks, documented frame spacing/window sampling, video-level aggregation.
3. **Fully synthetic facial imagery/video:** distinct generator-family and media-type tests.

Duplicating one image into eight frames does not create temporal evidence. For the advertised 8 FPS/T=8 window, only about one second is represented; pulse analysis requires a separately sampled, longer contiguous track if it is used at all.

### Accepted-only metrics are useful—but incomplete

The evaluator explicitly reports **Decided Accuracy/AUC** after removing abstentions. That is a selective metric, not intrinsically wrong. But do not call it overall accuracy, and do not compare it with another system evaluated on all inputs.

Report together:

- total attempts and readable/face-usable counts;
- all-score AUROC on every example with a valid score, including score-bearing abstentions;
- score coverage, and reasons for examples without any score;
- selective accuracy based on the **actual emitted labels**;
- acceptance coverage and abstention rates by real/fake class, generator and quality;
- confusion matrix among accepted predictions;
- false-positive rates both per all genuine attempts and per accepted genuine example, clearly distinguished.

The current benchmark reconstructs labels using `calibrated_risk >= 0.5`; this only evaluates actual decisions if the service uses the same boundary. Generator/difficulty columns are stored, but no slice summaries or confusion matrix are computed in the pasted body despite the description.

Handle empty datasets, zero accepted examples, one-class subsets and nonfinite scores. AUC is undefined without both classes; use `N/A` with a reason. In the genuine-only evaluator, zero accepted cases should produce **undefined accepted FPR**, not the currently returned 0.0. Also guard the division by total when there are no rows.

A genuine-only selfie test measures false positives, not fake recall or overall detection accuracy.

---

## 7. Calibration cannot repair ranking failure

For any finite positive temperature T:

```text
sigmoid(logit / T) >= 0.5  iff  logit >= 0
```

Positive temperature scaling preserves score ordering and the 0.5-threshold labels for the same evaluated examples, apart from finite-precision saturation/ties. Therefore it does **not** improve ordinary AUROC or fixed-0.5 accuracy. It may improve probability calibration and alter which examples a margin-based abstention rule accepts.

If AUROC is strong but accuracy is weak, investigate the operating threshold/class-prior mismatch. If ranking itself is weak, changing temperature is not the solution.

### Concrete ECE bug

The first bin excludes `p=0`. With:

```text
probabilities = [0.0, 0.9]
labels        = [1,   1]
```

The supplied function returns approximately **0.05** instead of **0.55** under its separated bins. If every probability is zero, it raises an `AttributeError` because `ece` remains a Python float and `.item()` is called.

The replacement `binary_ece()` includes both endpoints, validates inputs and returns `None` for an empty evaluation. This is positive-class probability calibration; it is not a top-label confidence-ECE estimator.

### Fit → persist → load → apply must be visible

`fit_temperature()` returns a scalar, but the supplied code does not show persistence or its use in inference. The model output name `calibrated_risk` does not prove calibration.

Require: detached logits from a separate calibration split; compatible tensor devices/dtypes; positive finite temperature; saved calibration artifact linked to checkpoint/config; inference reload; and a test that the returned score equals the specified transformation. Reassess after model or preprocessing changes. Brier score computation is also absent from the pasted calibration file.

---

## 8. Additional runtime/integration issues

- **Frozen modules:** `requires_grad=False` does not freeze BatchNorm buffers or disable dropout. After each parent `model.train()`, put truly frozen branches back into `eval()` while leaving trainable heads/adapters in the intended mode. A CPU behavior test confirmed BN running statistics changed under `train()` with all parameters frozen.
- **Localization devices:** mask targets and availability are passed without `.to(device)`. Ordinary CPU DataLoader outputs will conflict with CUDA predictions if localization is active, unless unseen code already moves them.
- **Training excerpt completeness:** `train_loader`, `val_loader` and `evaluate_validation` are undefined in the pasted module. If this is literally the whole file, Phase 5 cannot train as advertised.
- **CLI completeness:** the shown modules lack visible entry-point calls/argument parsing for several advertised commands. `run_training.py` defines `main()` but does not visibly call it. `--balanced` is not implemented in the pasted benchmark, `limit` is unused, and Phase 5 does not use `smoke_test` or `manifest_path` in its shown body.
- **Checkpoint creation:** ensure output directories exist and saving is atomic/versioned. Undefined or NaN validation AUC must not silently leave an old `best.ckpt` to be reused.

These are conditional repository findings where omitted code could supply the missing pieces. A command list is not execution evidence.

---

## 9. Repair order I recommend

### Gate A — prove you are evaluating the intended trained system

1. Fix always-load checkpoint behavior and config matching.
2. Prove phase handover with tensor/key checks.
3. Fix padded alignment, no-face behavior and error-result schema.
4. Separate image and video inference contracts.
5. Save raw per-example logits, emitted decisions, abstention reasons, quality, checkpoint/config identity and source groups.

**First rerun:** the existing benchmark, unchanged in membership, reporting all-score and selective metrics separately. This isolates implementation effects before any new training.

### Gate B — train the objective you claim

1. Establish an ordinary BCE baseline with correct splits.
2. Prove gradients for every intended trainable component.
3. Activate source-paired/codec training in the appropriate trainable phases.
4. Keep frozen feature branches behaviorally frozen.
5. Log losses and active supervision counts; use missing modalities deliberately during development.

### Gate C — earn reliability claims

1. Replace undefined uncertainty/disagreement outputs with defined, validated measures.
2. Implement true evidence-validity flags and core quality rejection.
3. Fit and persist calibration; choose thresholds on development/calibration data only.
4. Evaluate coverage-risk, genuine false positives and held-out generator families.
5. Only then compare additional architecture components against the repaired baseline.

No accuracy improvement is promised. The first success criterion is **a reproducible, correctly loaded, correctly measured system**.

## 10. Included files and the next missing input

The accompanying bundle includes:

- this audit;
- the copied excerpts and extracted modules;
- `scripts/audit_snippets.py` — 14 isolated checks;
- `code-review/isolated-checks.json` — recorded evidence;
- `code-review/reference_fixes.py` — narrow repair helpers;
- `scripts/test_reference_fixes.py` — 10 helper tests;
- `code-review/repair-checks.json` — recorded helper-test result.

These helpers are **not a patched full repository**. I have not replaced missing model/training logic with invented implementations.

**Most useful next file: the actual `models/forge_lite.py`.** It will show whether checkpoint loading, image/video handling, MC-dropout, disagreement, temperature and final decision rules are genuinely wired together. An actual repository ZIP would allow integration-level verification beyond these excerpts.
