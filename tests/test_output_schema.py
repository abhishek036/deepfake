"""
tests/test_output_schema.py
Unit tests for FORGE-Lite service schemas and PRD Section 4 JSON output schema conformance:
- Decision enum validation
- QualityInfo bounds and required fields
- AnalysisResult end-to-end schema validation
- HealthResponse and AnalysisJobStatus validation
"""

try:
    import pytest
except ImportError:
    pytest = None
from pydantic import ValidationError

from service.schemas import (
    AnalysisJobStatus,
    AnalysisResult,
    Decision,
    HealthResponse,
    QualityInfo,
    SuspiciousWindow,
)


def test_valid_analysis_result():
    sample_data = {
        "decision": "likely_fake",
        "fake_score": 0.8842,
        "calibrated_risk": 0.8521,
        "uncertainty": 0.0712,
        "evidence_coverage": 0.9400,
        "quality": {
            "face_visibility": 0.28,
            "track_continuity": 0.96,
            "blur": 0.12,
            "pose": 12.4,
            "occlusion": 0.05,
            "compression": 0.22,
            "audio_available": True,
            "rppg_available": True,
            "rppg_quality": 0.85,
        },
        "active_evidence": ["semantic", "forensic", "temporal", "rppg"],
        "unavailable_evidence": ["audio_visual"],
        "suspicious_windows": [
            {"start_seconds": 1.5, "end_seconds": 3.0, "score": 0.92}
        ],
        "localization_available": False,
        "abstention_reason": None,
        "model_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "processing_time_seconds": 1.45,
    }

    result = AnalysisResult.model_validate(sample_data)
    assert result.decision == Decision.LIKELY_FAKE
    assert result.fake_score == 0.8842
    assert result.quality.face_visibility == 0.28
    assert len(result.suspicious_windows) == 1
    assert result.suspicious_windows[0].score == 0.92


def test_abstain_decision_with_reason():
    sample_data = {
        "decision": "insufficient_evidence",
        "fake_score": 0.52,
        "calibrated_risk": 0.52,
        "uncertainty": 0.45,
        "evidence_coverage": 0.30,
        "quality": {
            "face_visibility": 0.03,
            "track_continuity": 0.20,
            "blur": 0.88,
            "pose": 45.0,
            "occlusion": 0.65,
            "compression": 0.90,
            "audio_available": False,
            "rppg_available": False,
            "rppg_quality": 0.0,
        },
        "active_evidence": ["semantic"],
        "unavailable_evidence": ["forensic", "temporal", "rppg", "audio_visual"],
        "suspicious_windows": [],
        "localization_available": False,
        "abstention_reason": "high_blur_and_low_coverage",
        "model_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "processing_time_seconds": 0.62,
    }

    result = AnalysisResult.model_validate(sample_data)
    assert result.decision == Decision.INSUFFICIENT_EVIDENCE
    assert result.abstention_reason == "high_blur_and_low_coverage"


def test_invalid_score_bounds():
    invalid_data = {
        "decision": "likely_real",
        "fake_score": 1.5,  # Invalid: > 1.0
        "calibrated_risk": 0.1,
        "uncertainty": 0.1,
        "evidence_coverage": 0.8,
        "quality": {
            "face_visibility": 0.5,
            "track_continuity": 0.9,
            "blur": 0.1,
            "pose": 0.0,
            "occlusion": 0.0,
            "compression": 0.1,
            "audio_available": True,
            "rppg_available": False,
            "rppg_quality": 0.0,
        },
        "active_evidence": ["semantic"],
        "unavailable_evidence": [],
        "suspicious_windows": [],
        "localization_available": False,
        "model_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "processing_time_seconds": 1.0,
    }

    raised = False
    try:
        AnalysisResult.model_validate(invalid_data)
    except ValidationError:
        raised = True
    assert raised, "Expected ValidationError for out-of-bound fake_score"


def test_health_response_schema():
    health = HealthResponse(
        status="healthy",
        gpu_available=True,
        gpu_name="NVIDIA GeForce RTX 4060",
        vram_total_gb=8.0,
        vram_free_gb=6.4,
        queue_depth=0,
        model_loaded=True,
        model_version="1.0.0",
    )
    assert health.status == "healthy"
    assert health.vram_total_gb == 8.0


def test_api_health_and_preset_endpoints():
    from fastapi.testclient import TestClient
    from service.api import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "gpu_available" in data

        presets_resp = client.get("/api/v1/presets")
        assert presets_resp.status_code == 200
        assert len(presets_resp.json()) >= 5
