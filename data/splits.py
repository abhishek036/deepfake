"""
data/splits.py
Deterministic, leakage-free split generation for FORGE-Lite.

PRD Rules (Section 7, Step 3):
- Never split frames from one video across train/test
- Keep original-real and manipulated versions in the same split
- Split by identity where labels exist
- Hold out entire manipulation families for zero-shot tests
- Hold out generator families for unseen-generator tests
- Save split files permanently
- Never tune thresholds on the final test set

newimp.md P0 corrections:
- Create development domains, a CALIBRATION partition, and untouched final target domains.
- Any target used for adaptation/threshold selection is development data thereafter.
- Enforce source-disjoint splits: no source_group appears in two partitions.
- Valid split names: 'train' | 'calibration' | 'val' | 'test' | 'zero_shot_test' | 'generator_test'
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _stable_hash_split(key: str, seed: int = 42) -> float:
    """Hash a string key to a stable float in [0, 1) for splitting."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class SplitGenerator:
    """
    Deterministic split generator that enforces all PRD leakage rules.

    The split is always done at the video/source level, never at the frame level.

    Partition layout (newimp.md §8 / P0 correction):
      train       → model training
      calibration → threshold tuning and temperature calibration (NEVER model selection)
      val         → model/architecture selection (development domain)
      test        → locked final evaluation — no access until Step 15
    """

    # All valid split names
    VALID_SPLITS = frozenset({
        "train", "calibration", "val", "test",
        "zero_shot_test", "generator_test",
    })

    def __init__(
        self,
        splits_dir: str = "data/splits",
        val_fraction: float = 0.10,
        calibration_fraction: float = 0.10,
        test_fraction: float = 0.14,
        seed: int = 42,
    ) -> None:
        self.splits_dir = Path(splits_dir)
        self.val_fraction = val_fraction
        self.calibration_fraction = calibration_fraction
        self.test_fraction = test_fraction
        self.train_fraction = 1.0 - val_fraction - calibration_fraction - test_fraction
        self.seed = seed
        if self.train_fraction <= 0:
            raise ValueError(
                f"Fractions sum to >= 1.0: val={val_fraction} cal={calibration_fraction} "
                f"test={test_fraction}. Reduce to leave room for train."
            )

    def generate(
        self,
        manifest_records: list[dict],
        dataset_name: str,
        split_by: str = "source_id",   # "source_id" | "identity_id" | "video_id"
        hold_out_manipulations: Optional[list[str]] = None,  # for zero-shot splits
        hold_out_generators: Optional[list[str]] = None,
        lock_test: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Assign train/calibration/val/test splits to all records.

        Args:
            manifest_records: List of manifest record dicts
            dataset_name: Used for saving split files
            split_by: Key to group by before splitting
            hold_out_manipulations: Manipulation families held out for zero-shot
            hold_out_generators: Generator families held out for unseen-generator test
            lock_test: If True, test split is saved and cannot be accessed during training

        Returns:
            dict with keys "train", "calibration", "val", "test" → list of records

        newimp.md P0:
          Once any target domain is used for adaptation selection it becomes
          development data. Track this in split metadata.
        """
        # Group records by the split key
        groups: dict[str, list[dict]] = {}
        for rec in manifest_records:
            key = rec.get(split_by, rec["video_id"])
            groups.setdefault(key, []).append(rec)

        splits: dict[str, list[dict]] = {"train": [], "calibration": [], "val": [], "test": []}

        assigned_sources: set[str] = set()  # track for source-disjoint validation

        for group_key, records in groups.items():
            # Check zero-shot hold-outs
            manip = records[0].get("manipulation_family", "")
            gen = records[0].get("generator_family", "")

            if hold_out_manipulations and manip in hold_out_manipulations:
                for r in records:
                    r["split"] = "zero_shot_test"
                splits.setdefault("zero_shot_test", []).extend(records)
                continue

            if hold_out_generators and gen in hold_out_generators:
                for r in records:
                    r["split"] = "generator_test"
                splits.setdefault("generator_test", []).extend(records)
                continue

            # Stable hash-based assignment
            h = _stable_hash_split(group_key, self.seed)
            if h < self.train_fraction:
                split_name = "train"
            elif h < self.train_fraction + self.calibration_fraction:
                split_name = "calibration"
            elif h < self.train_fraction + self.calibration_fraction + self.val_fraction:
                split_name = "val"
            else:
                split_name = "test"

            for r in records:
                r["split"] = split_name
                r["source_group_id"] = group_key  # propagate for bootstrap CI
            splits[split_name].extend(records)
            assigned_sources.add(group_key)

        # Validate source-disjoint property
        self._validate_source_disjoint(splits)

        # Log split statistics
        for split_name, recs in splits.items():
            n_fake = sum(1 for r in recs if r.get("label") == "fake")
            n_real = sum(1 for r in recs if r.get("label") == "real")
            logger.info("Split %-20s: %4d videos (%d fake, %d real)",
                        split_name, len(recs), n_fake, n_real)

        # Save split files permanently
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name, recs in splits.items():
            split_file = self.splits_dir / f"{dataset_name}_{split_name}.json"
            with open(split_file, "w") as f:
                json.dump([r["video_id"] for r in recs], f, indent=2)
            logger.info("Saved split file: %s", split_file)

        # Lock the test set (write metadata)
        if lock_test and "test" in splits:
            lock_file = self.splits_dir / "locked_test.json"
            locked_ids = set()
            if lock_file.exists():
                with open(lock_file) as f:
                    existing = json.load(f)
                locked_ids = set(existing.get("video_ids", []))
            locked_ids.update(r["video_id"] for r in splits["test"])
            with open(lock_file, "w") as f:
                json.dump({
                    "dataset": dataset_name,
                    "video_ids": sorted(locked_ids),
                    "locked": True,
                    "note": "NEVER ACCESS BEFORE STEP 15 (locked evaluation)",
                }, f, indent=2)
            logger.info("Test set locked: %s", lock_file)

        return splits

    def load_split(self, dataset_name: str, split_name: str) -> list[str]:
        """Load a saved split file and return list of video_ids."""
        split_file = self.splits_dir / f"{dataset_name}_{split_name}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            return json.load(f)

    def is_test_locked(self) -> bool:
        """Check if the test set is currently locked."""
        lock_file = self.splits_dir / "locked_test.json"
        if not lock_file.exists():
            return False
        with open(lock_file) as f:
            data = json.load(f)
        return data.get("locked", False)

    def unlock_test(self, reason: str) -> None:
        """Unlock the test set for final evaluation (Step 15 only)."""
        lock_file = self.splits_dir / "locked_test.json"
        if not lock_file.exists():
            logger.warning("No lock file found — test set was never locked")
            return
        with open(lock_file) as f:
            data = json.load(f)
        data["locked"] = False
        data["unlock_reason"] = reason
        with open(lock_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.warning("⚠️  Test set UNLOCKED for final evaluation. Reason: %s", reason)

    def _validate_source_disjoint(self, splits: dict[str, list[dict]]) -> None:
        """
        Verify no source_group_id appears in more than one partition.

        Per newimp.md P0: any target used for selection becomes development data.
        A source group spanning train+test is a critical data leakage bug.
        """
        partition_sources: dict[str, set[str]] = {}
        for split_name, recs in splits.items():
            partition_sources[split_name] = {
                r.get("source_group_id", r.get("video_id", "?")) for r in recs
            }

        split_names = list(partition_sources.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                a, b = split_names[i], split_names[j]
                overlap = partition_sources[a] & partition_sources[b]
                if overlap:
                    logger.error(
                        "⚠️  SOURCE LEAKAGE DETECTED: %d source_group_id(s) appear in "
                        "both '%s' and '%s' splits. Examples: %s",
                        len(overlap), a, b, list(overlap)[:5]
                    )
                else:
                    logger.debug("Source-disjoint check OK: %s vs %s", a, b)

    def mark_domain_as_development(self, domain_name: str, reason: str) -> None:
        """
        Register that a target domain has been used for adaptation/selection.

        Per newimp.md P0: once a target domain is used for any kind of tuning or
        adaptation selection, it must be treated as development data thereafter.
        The untouched final test role must be filled by a different domain.

        This creates a permanent record in the splits directory.
        """
        dev_file = self.splits_dir / "development_domains.json"
        dev_domains: dict = {}
        if dev_file.exists():
            with open(dev_file) as f:
                dev_domains = json.load(f)
        dev_domains[domain_name] = {
            "reason": reason,
            "note": "This domain is now DEVELOPMENT data. Do not use as locked test.",
        }
        with open(dev_file, "w") as f:
            json.dump(dev_domains, f, indent=2)
        logger.warning(
            "Domain '%s' marked as DEVELOPMENT (reason: %s). "
            "It can no longer serve as a locked final test domain.",
            domain_name, reason
        )
