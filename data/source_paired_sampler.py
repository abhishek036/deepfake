"""
data/source_paired_sampler.py
Source-Paired Batch Sampler for FORGE-Lite.

Implements newimp.md Phase B §1: Source-matched supervision.

Key rules (direct quotes from newimp.md):
  "For each fake with known ancestry, include its genuine source counterpart."
  "Never pair unrelated real and fake videos just because their indices match."
  "Document when source or identity metadata is unavailable."
  "If using a metric loss, specify the actual positive/negative relationships.
   Source pairing does NOT mean pulling genuine and fake embeddings together
   as if they shared a label."

The objective: make content/source LESS predictive of the class,
not to erase the manipulation distinction.

Usage:
    sampler = SourcePairedBatchSampler(dataset, batch_size=8, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_forge)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Iterator, Optional

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class SourcePairedBatchSampler(Sampler):
    """
    Batch sampler that groups real/fake pairs by source_group_id.

    Each batch is assembled to contain:
      - At least one real and one fake sample from the SAME source group
        (when source metadata is available and pairs exist)
      - Remaining slots filled with other samples (balanced across classes)

    This directly implements newimp.md Phase B §1:
      Source-matched supervision prevents content/identity from being the
      discriminating feature between real and fake.

    When source metadata is unavailable:
      - Logs a warning (per newimp.md: "Document when source or identity
        metadata is unavailable")
      - Falls back to class-balanced sampling without pairing

    Args:
        dataset: ForgeDataset with records that have source_group_id
        batch_size: total samples per batch (should be even for balance)
        drop_last: drop incomplete final batch
        pairs_per_batch: how many source-paired (real, fake) pairs per batch.
                         Remaining slots are filled with balanced samples.
        seed: reproducibility
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 8,
        drop_last: bool = True,
        pairs_per_batch: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed
        self._rng = random.Random(seed)

        # Build indices organized by source_group_id and label
        self._source_real: dict[str, list[int]] = defaultdict(list)
        self._source_fake: dict[str, list[int]] = defaultdict(list)
        self._all_real: list[int] = []
        self._all_fake: list[int] = []
        self._no_source_metadata_count: int = 0

        self._build_indices()

    def _build_indices(self) -> None:
        """Scan dataset records and build source-grouped index maps."""
        records = getattr(self.dataset, "records", [])
        for i, record in enumerate(records):
            label = int(record.get("label_int", record.get("label", 0)))
            src_id = record.get("source_group_id", "")

            if not src_id:
                self._no_source_metadata_count += 1
                # Use video_id as degenerate single-element source group
                src_id = record.get("video_id", f"__idx_{i}")

            if label == 1:
                self._source_fake[src_id].append(i)
                self._all_fake.append(i)
            else:
                self._source_real[src_id].append(i)
                self._all_real.append(i)

        # Find sources that have BOTH real and fake members (true pairs)
        all_sources = set(self._source_real.keys()) | set(self._source_fake.keys())
        self._paired_sources: list[str] = [
            s for s in all_sources
            if self._source_real.get(s) and self._source_fake.get(s)
        ]

        pct_missing = 100 * self._no_source_metadata_count / max(len(records), 1)
        if self._no_source_metadata_count > 0:
            logger.warning(
                "SourcePairedBatchSampler: %d / %d records (%.1f%%) missing source_group_id. "
                "These will be sampled without source pairing. "
                "Per newimp.md: document when source metadata is unavailable.",
                self._no_source_metadata_count, len(records), pct_missing
            )

        logger.info(
            "SourcePairedBatchSampler: %d real, %d fake, %d paired sources, "
            "%d unpaired sources",
            len(self._all_real), len(self._all_fake),
            len(self._paired_sources),
            len(all_sources) - len(self._paired_sources),
        )

        if not self._paired_sources:
            logger.warning(
                "No source-paired (real+fake) groups found. "
                "Falling back to class-balanced sampling without source pairing. "
                "Check that manifest records include 'source_group_id'."
            )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches with source-paired samples."""
        self._rng = random.Random(self.seed + 1)  # re-shuffle each epoch

        # Shuffle all pools
        all_real = self._all_real.copy()
        all_fake = self._all_fake.copy()
        paired_sources = self._paired_sources.copy()
        self._rng.shuffle(all_real)
        self._rng.shuffle(all_fake)
        self._rng.shuffle(paired_sources)

        real_pool = list(all_real)
        fake_pool = list(all_fake)
        paired_pool = list(paired_sources)

        total = len(self.dataset)
        n_batches = total // self.batch_size

        for _ in range(n_batches):
            batch: list[int] = []

            # 1. Fill with source-paired samples (real+fake from same source)
            n_pairs_added = 0
            while n_pairs_added < self.pairs_per_batch and paired_pool:
                src = paired_pool.pop()
                reals = self._source_real.get(src, [])
                fakes = self._source_fake.get(src, [])
                if reals and fakes:
                    batch.append(self._rng.choice(reals))
                    batch.append(self._rng.choice(fakes))
                    n_pairs_added += 1
                if not paired_pool:
                    # Reshuffle when exhausted
                    paired_pool = self._paired_sources.copy()
                    self._rng.shuffle(paired_pool)

            # 2. Fill remaining slots with class-balanced samples
            remaining = self.batch_size - len(batch)
            half = remaining // 2
            extra = remaining - half

            def _pop(pool: list[int], fallback: list[int], n: int) -> list[int]:
                """Pop n items from pool, refill from fallback if needed."""
                result = []
                for _ in range(n):
                    if not pool:
                        pool.extend(fallback)
                        self._rng.shuffle(pool)
                    if pool:
                        result.append(pool.pop())
                return result

            batch.extend(_pop(real_pool, all_real, half))
            batch.extend(_pop(fake_pool, all_fake, extra))

            # Shuffle within batch to avoid position bias
            self._rng.shuffle(batch)

            if len(batch) == self.batch_size:
                yield batch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


class ClassBalancedSampler(Sampler):
    """
    Simple class-balanced sampler for use when source metadata is unavailable.

    Per newimp.md Phase B §2: Use the SAME distribution of resolutions, codecs,
    crop settings and augmentations for real and fake examples.
    """

    def __init__(
        self,
        dataset,
        oversample_minority: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.oversample_minority = oversample_minority
        self.seed = seed

        records = getattr(dataset, "records", [])
        self._real_indices = [
            i for i, r in enumerate(records)
            if int(r.get("label_int", r.get("label", 0))) == 0
        ]
        self._fake_indices = [
            i for i, r in enumerate(records)
            if int(r.get("label_int", r.get("label", 0))) == 1
        ]

        majority = max(len(self._real_indices), len(self._fake_indices))
        minority_label = "fake" if len(self._fake_indices) < len(self._real_indices) else "real"
        logger.info(
            "ClassBalancedSampler: %d real, %d fake | minority=%s | oversample=%s",
            len(self._real_indices), len(self._fake_indices),
            minority_label, oversample_minority
        )

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        n = max(len(self._real_indices), len(self._fake_indices))

        real_pool = self._real_indices.copy()
        fake_pool = self._fake_indices.copy()

        # Oversample minority class with replacement
        while len(real_pool) < n:
            real_pool.extend(self._real_indices)
        while len(fake_pool) < n:
            fake_pool.extend(self._fake_indices)

        real_pool = real_pool[:n]
        fake_pool = fake_pool[:n]
        rng.shuffle(real_pool)
        rng.shuffle(fake_pool)

        # Interleave real and fake
        combined = []
        for r, f in zip(real_pool, fake_pool):
            combined.extend([r, f])
        rng.shuffle(combined)
        return iter(combined)

    def __len__(self) -> int:
        return 2 * max(len(self._real_indices), len(self._fake_indices))
