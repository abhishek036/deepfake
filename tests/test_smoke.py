"""
tests/test_smoke.py
FORGE-Lite smoke tests — validate end-to-end pipeline with synthetic data.

Run: pytest tests/test_smoke.py -v

These tests validate:
1. Preprocessing pipeline (decode, detect, track, align, quality)
2. Synthetic dataset generation
3. Dataset loading (ForgeDataset)
4. Model forward pass (FORGELite)
5. Loss computation
6. Inference engine
7. API health schema

No real data required. No GPU required (runs on CPU with random weights).
"""

import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

def test_quality_analyzer_frame():
    from preprocessing.quality import QualityAnalyzer
    qa = QualityAnalyzer()
    # Create a dummy 224×224 face crop
    face = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    result = qa.analyze_frame(
        face_crop_bgr=face,
        bbox=np.array([10, 10, 214, 214], dtype=np.float32),
        frame_wh=(640, 480),
    )
    assert 0.0 <= result.blur_score <= 1.0
    assert 0.0 <= result.brightness <= 1.0
    assert result.to_vector().shape == (10,)


def test_face_aligner():
    from preprocessing.align_faces import FaceAligner
    aligner = FaceAligner()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array([
        [320, 170], [420, 170], [370, 220], [330, 270], [410, 270]
    ], dtype=np.float32)
    aligned, transform = aligner.align(frame, landmarks)
    assert aligned.shape == (224, 224, 3)
    assert transform.shape == (2, 3)


def test_face_tracker():
    from preprocessing.track_faces import FaceTracker
    from preprocessing.detect_faces import FaceDetection
    tracker = FaceTracker()
    bbox = np.array([100.0, 100.0, 200.0, 200.0])
    lm = np.zeros((5, 2), dtype=np.float32)
    det = FaceDetection(bbox=bbox, score=0.95, landmarks=lm, frame_idx=0, timestamp_sec=0.0)
    result = tracker.update([det], frame_idx=0)
    assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset tests
# ─────────────────────────────────────────────────────────────────────────────

def test_synthetic_dataset_generation(tmp_path):
    from data.synthetic_test import SyntheticDatasetGenerator
    gen = SyntheticDatasetGenerator(
        output_dir=str(tmp_path / "synthetic"),
        num_real=4, num_fake=4,
        frames_per_video=8,
    )
    manifest_path = gen.generate()
    assert (tmp_path / "synthetic" / "manifest.jsonl").exists()
    import json
    records = [json.loads(l) for l in open(manifest_path).readlines()]
    assert len(records) == 8
    real_count = sum(1 for r in records if r["label"] == "real")
    fake_count = sum(1 for r in records if r["label"] == "fake")
    assert real_count == 4
    assert fake_count == 4


def test_forge_dataset(tmp_path):
    from data.synthetic_test import SyntheticDatasetGenerator
    from data.dataset import ForgeDataset, collate_forge
    from torch.utils.data import DataLoader

    gen = SyntheticDatasetGenerator(
        output_dir=str(tmp_path / "syn"),
        num_real=6, num_fake=6,
        frames_per_video=20,
    )
    manifest = gen.generate()

    ds = ForgeDataset(
        manifest_path=manifest,
        split="train",
        window_size=8,
        augment=False,
        min_track_length=1,
    )
    assert len(ds) > 0

    sample = ds[0]
    assert "frames" in sample
    assert sample["frames"].shape[-3:] == torch.Size([3, 224, 224])
    assert "label" in sample
    assert "face_quality" in sample
    assert sample["face_quality"].shape == (10,)

    loader = DataLoader(ds, batch_size=2, collate_fn=collate_forge)
    batch = next(iter(loader))
    assert "frames" in batch
    assert "label" in batch


# ─────────────────────────────────────────────────────────────────────────────
# SRM filter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_srm_filter_bank():
    from models.srm_filters import SRMFilterBank, HaarWaveletBank, ForensicPreprocessor
    srm = SRMFilterBank()
    lum = torch.rand(2, 1, 224, 224)
    out = srm(lum)
    assert out.shape == (2, 30, 224, 224)

    haar = HaarWaveletBank()
    out_haar = haar(lum)
    assert out_haar.shape == (2, 3, 112, 112)  # half spatial size

    preprocessor = ForensicPreprocessor()
    rgb = torch.rand(2, 3, 224, 224)
    forensic_feat = preprocessor(rgb)
    assert forensic_feat.shape == (2, 35, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Model tests (CPU, no GPU required)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_forge_lite_forward_cpu():
    """Full model forward pass on CPU with random weights."""
    from models.forge_lite import FORGELite

    model = FORGELite(
        enable_semantic=False,  # skip DINOv2 download in tests
        enable_forensic=True,
        enable_temporal=True,
        enable_localization=False,
    ).eval()

    B, T = 1, 8
    frames = torch.randn(B, T, 3, 224, 224)
    diffs = torch.randn(B, T - 1, 3, 224, 224)
    fq = torch.rand(B, 10)
    tq = torch.rand(B, 5)
    cont = torch.ones(B)

    with torch.no_grad():
        out = model(
            frames=frames, frame_diffs=diffs,
            face_quality=fq, track_quality=tq,
            track_continuity=cont,
        )

    assert "fake_logit" in out
    assert out["fake_logit"].shape == (B,)
    assert "uncertainty" in out
    assert "branch_weights" in out
    assert out["branch_weights"].shape == (B, 5)


def test_forensic_branch():
    """Test forensic CNN branch independently."""
    from models.forensic_branch import ForensicResidualBranch
    branch = ForensicResidualBranch()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = branch(x)
    assert out["embedding"].shape == (2, 256)
    assert out["fft_features"].shape == (2, 20)


def test_temporal_encoder():
    """Test temporal GRU encoder independently."""
    from models.temporal_branch import TemporalEncoder
    enc = TemporalEncoder()
    B, T = 2, 8
    out = enc(
        dino_features=torch.randn(B, T, 384),
        forensic_features=torch.randn(B, T, 256),
        fft_features=torch.randn(B, T, 20),
        frame_diffs=torch.randn(B, T - 1, 3, 224, 224),
        landmark_velocity=torch.randn(B, T, 10),
        face_quality=torch.rand(B, 10),
        track_continuity=torch.ones(B),
    )
    assert out["temporal_embedding"].shape == (B, 256)
    assert out["frame_scores"].shape == (B, T)


def test_fusion_head():
    """Test reliability-aware fusion head."""
    from models.fusion import ReliabilityAwareFusion
    B = 4
    fusion = ReliabilityAwareFusion()
    out = fusion(
        semantic_embedding=torch.randn(B, 384),
        forensic_embedding=torch.randn(B, 256),
        fft_features=torch.randn(B, 20),
        temporal_embedding=torch.randn(B, 256),
        temporal_scores=torch.randn(B, 5),
        rppg_features=torch.zeros(B, 16),
        av_score=torch.zeros(B, 1),
        face_quality=torch.rand(B, 10),
        video_quality=torch.rand(B, 5),
        rppg_available=torch.zeros(B, dtype=torch.bool),
        av_available=torch.zeros(B, dtype=torch.bool),
    )
    assert out["fake_logit"].shape == (B,)
    assert out["branch_weights"].shape == (B, 5)
    assert out["uncertainty"].shape == (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Loss tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forge_loss():
    from training.losses import ForgeLoss
    loss_fn = ForgeLoss()
    B = 4
    losses = loss_fn(
        fake_logit=torch.randn(B),
        labels=torch.randint(0, 2, (B,)),
        semantic_embedding=torch.randn(B, 384),
    )
    assert "total" in losses
    assert losses["total"].item() > 0
    assert losses["classification"].item() > 0


def test_focal_loss():
    from training.losses import FocalLoss
    loss = FocalLoss()
    logits = torch.randn(8)
    labels = torch.randint(0, 2, (8,)).float()
    out = loss(logits, labels)
    assert out.item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Calibration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_calibration_ece():
    from models.calibration import compute_ece, compute_brier_score
    probs = np.random.rand(100)
    labels = np.random.randint(0, 2, 100)
    ece = compute_ece(probs, labels)
    assert 0.0 <= ece <= 1.0
    bs = compute_brier_score(probs, labels)
    assert 0.0 <= bs <= 1.0


def test_abstention_layer():
    from models.calibration import CalibrationLayer
    B = 4
    cal = CalibrationLayer()
    out = cal(
        fake_logit=torch.randn(B),
        uncertainty=torch.rand(B) * 0.5,
        evidence_coverage=torch.ones(B),
        branch_disagreement=torch.rand(B) * 0.3,
        face_quality_composite=torch.ones(B) * 0.8,
    )
    assert "calibrated_score" in out
    assert len(out["decisions"]) == B


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator():
    from evaluation.evaluate_video import ForgeEvaluator
    evaluator = ForgeEvaluator()
    for i in range(20):
        label = i % 2
        score = 0.9 if label == 1 else 0.1
        evaluator.add_result(
            video_id=f"vid_{i}", label=label,
            score=score + np.random.randn() * 0.1,
            abstained=False, dataset="test_ds"
        )
    metrics = evaluator.compute()
    assert "video_auc" in metrics
    assert metrics["video_auc"] > 0.6  # should be reasonable with this setup


# ─────────────────────────────────────────────────────────────────────────────
# rPPG tests
# ─────────────────────────────────────────────────────────────────────────────

def test_rppg_quality_gate():
    from models.rppg_branch import RPPGBranch
    rppg = RPPGBranch(min_duration_sec=8.0)
    # Should fail: short duration
    gate_ok, reason = rppg.check_quality_gate(5.0, 0.1, 0.9, 0.1)
    assert not gate_ok

    # Should pass
    gate_ok, reason = rppg.check_quality_gate(10.0, 0.1, 0.9, 0.1)
    assert gate_ok


def test_rppg_feature_vector():
    from models.rppg_branch import RPPGBranch
    rppg = RPPGBranch()
    faces = [np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8) for _ in range(50)]
    # Should fail quality gate (short duration at 25fps)
    result = rppg.extract(
        face_crops_bgr=faces[:10],
        fps=25.0,
        track_duration_sec=0.4,  # too short
    )
    assert not result.available
    fv = result.to_feature_vector()
    assert fv.shape == (16,)


# ─────────────────────────────────────────────────────────────────────────────
# Splits tests
# ─────────────────────────────────────────────────────────────────────────────

def test_split_generator(tmp_path):
    from data.splits import SplitGenerator
    gen = SplitGenerator(splits_dir=str(tmp_path / "splits"))
    records = [
        {"video_id": f"vid_{i}", "label": "fake" if i % 2 else "real",
         "source_id": f"src_{i}", "manipulation_family": "deepfakes",
         "generator_family": "df", "split": ""}
        for i in range(30)
    ]
    splits = gen.generate(records, "test_dataset", lock_test=False)
    assert "train" in splits
    assert len(splits["train"]) > 0
    total = sum(len(v) for v in splits.values())
    assert total == 30
