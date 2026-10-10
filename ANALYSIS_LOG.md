# ANALYSIS_LOG.md — Post-Hoc Analysis Deviations
**Rule:** Any analysis performed after seeing results, or any protocol deviation, is logged here with date and reason.

---

## 2026-10-10 — Loss Polarity Finding (Phase A)

**Finding:** `pos_weight=18.4` in `configs/train.yaml` and `training/losses.py` upweights **fake** samples (label=1), not real samples as stated in the code comments. Label convention is 0=real, 1=fake (`data/dataset.py` L45). Combined with `balanced_sampling=true`, fakes receive disproportionate loss weight even in balanced batches.

**Impact on Paper 1:** Does not affect the audit results (all numbers come from post-hoc inference, not from re-training). Must be disclosed as a known training confound of the best.ckpt checkpoint.

**Action:** Added note to phase_a_log. Will disclose in paper Limitations section. No re-training in Paper 1.

**Decision:** Pre-registered? No — this was a diagnostic finding from Phase A code audit, not a hypothesis test. Logging here per rule 2.
