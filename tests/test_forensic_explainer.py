"""
tests/test_forensic_explainer.py
Automated Unit and Integration Tests for FORGE-Lite V3 Forensic Explanation Layer.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import cv2

from service.forensic_explainer import ForensicExplainer, ForensicDossier
from utils.forensic_html_report import (
    generate_forensic_markdown,
    generate_forensic_html,
    save_forensic_dossier,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def explainer():
    """Initializes ForensicExplainer once for test module."""
    return ForensicExplainer(policy_path=REPO_ROOT / "configs" / "frozen_v3_policy.json")


def test_frozen_policy_loading_and_hash_verification(explainer):
    """Verifies that the policy loads read-only and validates the frozen V3 SHA-256."""
    expected_hash = "904E8A91391A337B5BEBD1B95EAFD45A8F5F819F2FA024067B68CF96B3BE5B03"
    assert explainer.checkpoint_sha256 == expected_hash
    assert explainer.platt_a > 0
    assert explainer.platt_b > 0
    assert explainer.tau_abst == 0.2661
    assert explainer.min_quality == 0.35
    assert explainer.review_lower == 0.40
    assert explainer.review_upper == 0.60


def test_explain_synthetic_image(explainer):
    """Tests explanation on an image and verifies all components of the forensic dossier."""
    # Create a synthetic face crop (224, 224, 3)
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    # Draw simple facial feature patterns (eyes, mouth) to create signal
    cv2.circle(img, (75, 85), 18, (230, 230, 230), -1)
    cv2.circle(img, (149, 85), 18, (230, 230, 230), -1)
    cv2.ellipse(img, (112, 160), (35, 15), 0, 0, 180, (210, 210, 210), -1)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_img_path = Path(f.name)
    try:
        cv2.imwrite(str(temp_img_path), img)
        dossier = explainer.explain_image(temp_img_path)

        # 1. Check Triage Verdict & Schema
        assert dossier.triage_verdict in ["REAL", "FAKE", "REVIEW"]
        assert dossier.operational_triage == dossier.triage_verdict
        assert dossier.model_assessment in ["LIKELY_MANIPULATED", "LIKELY_AUTHENTIC", "AMBIGUOUS"]
        assert isinstance(dossier.triage_reason_codes, list)
        assert len(dossier.triage_reason_codes) >= 1

        # 2. Check Provenance
        assert dossier.provenance.checkpoint_sha256 == explainer.EXPECTED_CHECKPOINT_SHA256
        assert len(dossier.provenance.input_sha256) == 64

        # 3. Check Counterfactual Attribution & Directional Evidence
        assert len(dossier.branch_attributions) == 5
        sem_branch = next(b for b in dossier.branch_attributions if b.name == "semantic")
        assert sem_branch.available is True
        assert 0.0 <= sem_branch.fake_probability <= 1.0
        assert 0.0 <= sem_branch.router_weight <= 1.0
        # Counterfactual logit delta and counterfactual_branch_effect property
        assert isinstance(sem_branch.counterfactual_logit_delta, float)
        assert sem_branch.counterfactual_branch_effect == sem_branch.counterfactual_logit_delta

        # 4. Check 4-factor uncertainty deconstruction
        u = dossier.uncertainty
        assert 0.0 <= u.composite_uncertainty <= 1.0
        assert 0.0 <= u.predictive_entropy <= 1.0
        assert 0.0 <= u.decision_margin_uncertainty <= 1.0
        assert 0.0 <= u.quality_defect <= 1.0
        assert u.branch_disagreement >= 0.0
        assert len(u.primary_uncertainty_driver) > 0

        # 5. Check quality audit
        q = dossier.quality
        assert q.quality_tier in ["OPTIMAL", "ACCEPTABLE", "DEGRADED"]
        assert "NOT evidence of manipulation" in q.reliability_assessment or "adequate" in q.reliability_assessment or "High-fidelity" in q.reliability_assessment

        # 6. Check visual diagnostics
        assert dossier.aligned_face_b64 is not None
        assert dossier.srm_residual_b64 is not None

        # 7. Check narrative and disclaimer
        assert len(dossier.forensic_narrative) > 20
        assert "DISCLAIMER" in dossier.operational_disclaimer

    finally:
        if temp_img_path.exists():
            temp_img_path.unlink()


def test_dossier_serialization_and_html_generation(explainer):
    """Tests that Markdown and HTML reporting formats generate properly."""
    img = np.full((224, 224, 3), 140, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_img_path = Path(f.name)
    try:
        cv2.imwrite(str(temp_img_path), img)
        dossier = explainer.explain_image(temp_img_path)

        # JSON
        d_dict = dossier.to_dict()
        assert isinstance(d_dict, dict)
        assert "model_assessment" in d_dict
        assert "operational_triage" in d_dict
        assert "triage_reason_codes" in d_dict

        d_json = dossier.to_json()
        parsed = json.loads(d_json)
        assert parsed["triage_verdict"] == dossier.triage_verdict
        assert parsed["model_assessment"] == dossier.model_assessment
        assert parsed["operational_triage"] == dossier.operational_triage
        assert parsed["triage_reason_codes"] == dossier.triage_reason_codes

        # Markdown
        md = generate_forensic_markdown(dossier)
        assert "# FORGE-Lite V3 Forensic Decision-Support Dossier" in md
        assert dossier.provenance.checkpoint_sha256 in md
        assert "Counterfactual Branch Effect" in md
        assert dossier.operational_triage in md

        # HTML
        html = generate_forensic_html(dossier)
        assert "<!DOCTYPE html>" in html
        assert dossier.operational_triage in html
        assert dossier.model_assessment in html
        assert dossier.provenance.checkpoint_sha256 in html
        assert "Counterfactual Branch Effect" in html

        # Save to disk
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_prefix = Path(tmp_dir) / "test_dossier"
            saved = save_forensic_dossier(dossier, out_prefix)
            assert Path(saved["json"]).exists()
            assert Path(saved["markdown"]).exists()
            assert Path(saved["html"]).exists()
    finally:
        if temp_img_path.exists():
            temp_img_path.unlink()


def test_quality_triggers_review(explainer):
    """Verifies that physical media degradation reduces reliability and triggers REVIEW."""
    # Create a severely degraded image (heavy blur ensuring Laplacian variance is near zero)
    base = np.full((224, 224, 3), 128, dtype=np.uint8)
    degraded = cv2.GaussianBlur(base, (51, 51), 30.0)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_img_path = Path(f.name)
    try:
        cv2.imwrite(str(temp_img_path), degraded)
        dossier = explainer.explain_image(temp_img_path)

        # Degraded quality should trigger REVIEW
        assert dossier.quality.quality_tier == "DEGRADED"
        assert dossier.operational_triage == "REVIEW"
        assert dossier.triage_verdict == "REVIEW"
        assert "QUALITY_BELOW_FLOOR" in dossier.triage_reason_codes
        assert "NOT evidence of manipulation" in dossier.quality.reliability_assessment
    finally:
        if temp_img_path.exists():
            temp_img_path.unlink()


def test_triage_policy_deterministic_edge_cases(explainer):
    """
    Exhaustive deterministic unit tests verifying the exact 3-way triage policy:
    REVIEW: U > tau_U (0.2661) OR Q < Q_floor (0.35) OR 0.40 <= p < 0.60 OR disagree > 0.20
    FAKE:   p >= 0.60 AND U <= tau_U AND Q >= Q_floor
    REAL:   p < 0.40  AND U <= tau_U AND Q >= Q_floor
    """
    from service.forensic_explainer import QualityReliabilityAudit, UncertaintyDeconstruction

    def _mock_quality(comp_q=0.90, tier="OPTIMAL"):
        return QualityReliabilityAudit(
            composite_quality=comp_q,
            blur_sharpness=0.9,
            compression_artifact_score=0.9,
            illumination_score=0.9,
            pose_yaw_degrees=5.0,
            face_resolution_ratio=1.0,
            track_continuity=1.0,
            quality_tier=tier,
            reliability_assessment="Good",
        )

    def _mock_uncertainty(comp_u=0.10):
        return UncertaintyDeconstruction(
            composite_uncertainty=comp_u,
            predictive_entropy=0.2,
            decision_margin_uncertainty=0.2,
            quality_defect=0.1,
            branch_disagreement=0.05,
            primary_uncertainty_driver="Predictive Entropy",
            abstention_triggered=(comp_u > 0.2661),
        )

    # 1. p=0.39, U low, Q good -> REAL (LIKELY_AUTHENTIC, CLEAR_SIGNAL_AUTHENTIC)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.39,
        uncertainty=_mock_uncertainty(0.10),
        quality=_mock_quality(0.90),
        disagreement=0.05,
    )
    assert triage == "REAL"
    assert assess == "LIKELY_AUTHENTIC"
    assert "CLEAR_SIGNAL_AUTHENTIC" in codes

    # 2. p=0.40, U low, Q good -> REVIEW (AMBIGUOUS, BORDERLINE_RISK)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.40,
        uncertainty=_mock_uncertainty(0.10),
        quality=_mock_quality(0.90),
        disagreement=0.05,
    )
    assert triage == "REVIEW"
    assert assess == "AMBIGUOUS"
    assert "BORDERLINE_RISK" in codes

    # 3. p=0.59, U low, Q good -> REVIEW (AMBIGUOUS, BORDERLINE_RISK)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.59,
        uncertainty=_mock_uncertainty(0.10),
        quality=_mock_quality(0.90),
        disagreement=0.05,
    )
    assert triage == "REVIEW"
    assert assess == "AMBIGUOUS"
    assert "BORDERLINE_RISK" in codes

    # 4. p=0.60, U low, Q good -> FAKE (LIKELY_MANIPULATED, CLEAR_SIGNAL_MANIPULATED)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.60,
        uncertainty=_mock_uncertainty(0.10),
        quality=_mock_quality(0.90),
        disagreement=0.05,
    )
    assert triage == "FAKE"
    assert assess == "LIKELY_MANIPULATED"
    assert "CLEAR_SIGNAL_MANIPULATED" in codes

    # 5. U > threshold (U=0.28 > 0.2661) -> REVIEW (HIGH_UNCERTAINTY)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.20,
        uncertainty=_mock_uncertainty(0.28),
        quality=_mock_quality(0.90),
        disagreement=0.05,
    )
    assert triage == "REVIEW"
    assert assess == "LIKELY_AUTHENTIC"
    assert "HIGH_UNCERTAINTY" in codes

    # 6. Q < floor (Q=0.34 < 0.35, tier="DEGRADED") -> REVIEW (QUALITY_BELOW_FLOOR)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.85,
        uncertainty=_mock_uncertainty(0.10),
        quality=_mock_quality(0.34, tier="DEGRADED"),
        disagreement=0.05,
    )
    assert triage == "REVIEW"
    assert assess == "LIKELY_MANIPULATED"
    assert "QUALITY_BELOW_FLOOR" in codes

    # 7. p high + Q low (p=0.92, Q=0.20) -> REVIEW (QUALITY_BELOW_FLOOR)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.92,
        uncertainty=_mock_uncertainty(0.08),
        quality=_mock_quality(0.20, tier="DEGRADED"),
        disagreement=0.02,
    )
    assert triage == "REVIEW"
    assert assess == "LIKELY_MANIPULATED"
    assert "QUALITY_BELOW_FLOOR" in codes

    # 8. p low + Q low (p=0.15, Q=0.20) -> REVIEW (QUALITY_BELOW_FLOOR)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.15,
        uncertainty=_mock_uncertainty(0.08),
        quality=_mock_quality(0.20, tier="DEGRADED"),
        disagreement=0.02,
    )
    assert triage == "REVIEW"
    assert assess == "LIKELY_AUTHENTIC"
    assert "QUALITY_BELOW_FLOOR" in codes

    # 9. Branch disagreement conflict (> 0.20) -> REVIEW (BRANCH_CONFLICT)
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.75,
        uncertainty=_mock_uncertainty(0.12),
        quality=_mock_quality(0.85),
        disagreement=0.25,
    )
    assert triage == "REVIEW"
    assert assess == "LIKELY_MANIPULATED"
    assert "BRANCH_CONFLICT" in codes

    # 10. Multi-trigger compound case
    triage, assess, codes = explainer._assign_triage_decision(
        cal_p=0.52,
        uncertainty=_mock_uncertainty(0.30),
        quality=_mock_quality(0.25, tier="DEGRADED"),
        disagreement=0.22,
    )
    assert triage == "REVIEW"
    assert assess == "AMBIGUOUS"
    assert set(["BORDERLINE_RISK", "HIGH_UNCERTAINTY", "QUALITY_BELOW_FLOOR", "BRANCH_CONFLICT"]).issubset(set(codes))


def test_html_escaping_and_security(explainer):
    """
    Verifies that all user/media-derived strings (filenames, narratives,
    device, reason codes, branch names) are strictly HTML-escaped against injection.
    """
    from service.forensic_explainer import (
        ForensicDossier,
        ForensicProvenanceHeader,
        UncertaintyDeconstruction,
        QualityReliabilityAudit,
        BranchForensicDetail,
    )

    xss_payload_1 = "<script>alert('xss')</script>"
    xss_payload_2 = '"><img src=x onerror=alert(1)>'

    provenance = ForensicProvenanceHeader(
        input_sha256=f"hash_{xss_payload_1}",
        checkpoint_sha256=explainer.EXPECTED_CHECKPOINT_SHA256,
        model_version=f"FORGE-{xss_payload_2}",
        calibration_version="Platt-V3",
        uncertainty_version="Composite-Platt-V3",
        report_schema_version="3.0",
        analysis_timestamp_utc="2026-09-18T00:00:00Z",
        media_type="image",
        execution_device="cuda",
    )

    dossier = ForensicDossier(
        provenance=provenance,
        triage_verdict="REVIEW",
        calibrated_fake_risk=0.8335,
        uncalibrated_raw_logit=0.0601,
        uncertainty=UncertaintyDeconstruction(
            composite_uncertainty=0.1475,
            predictive_entropy=0.6497,
            decision_margin_uncertainty=0.3331,
            quality_defect=0.6590,
            branch_disagreement=0.0693,
            primary_uncertainty_driver=f"Entropy_{xss_payload_1}",
            abstention_triggered=False,
        ),
        quality=QualityReliabilityAudit(
            composite_quality=0.3410,
            blur_sharpness=0.28,
            compression_artifact_score=0.1,
            illumination_score=0.8,
            pose_yaw_degrees=5.0,
            face_resolution_ratio=1.0,
            track_continuity=1.0,
            quality_tier="DEGRADED",
            reliability_assessment=f"Reliability note with {xss_payload_2}",
        ),
        branch_attributions=[
            BranchForensicDetail(
                name="semantic",
                available=True,
                router_weight=0.55,
                fake_probability=0.55,
                confidence_margin=0.10,
                directional_evidence=0.027,
                counterfactual_logit_delta=0.19,
                counterfactual_prob_delta=0.05,
                role_summary=f"Role summary with {xss_payload_1}",
            )
        ],
        dominant_supporting_branch="semantic",
        dominant_contradictory_branch=None,
        temporal_aggregation=None,
        forensic_narrative=f"Forensic narrative with {xss_payload_1}",
        operational_disclaimer=f"Legal disclaimer with {xss_payload_2}",
        model_assessment="LIKELY_MANIPULATED",
        operational_triage="REVIEW",
        triage_reason_codes=["QUALITY_BELOW_FLOOR", f"CUSTOM_{xss_payload_2}"],
    )

    rendered_html = generate_forensic_html(dossier, title=f"Title {xss_payload_1}")

    # Script tag must not appear unescaped anywhere in the HTML
    assert "<script>alert('xss')</script>" not in rendered_html
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in rendered_html or "&lt;script&gt;alert('xss')&lt;/script&gt;" in rendered_html

    # Img onerror tag must not appear unescaped anywhere in the HTML
    assert '<img src=x onerror=alert(1)>' not in rendered_html
    assert "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;" in rendered_html or "&lt;img src=x onerror=alert(1)&gt;" in rendered_html

    # HTML-escaped against the tested injection cases verified
    assert "HTML-escaped against tested injection cases" in rendered_html


def test_counterfactual_attribution_directionality(explainer):
    """
    Deterministic attribution test verifying directional evidence semantics:
    - Delta_z > 0: branch supports FAKE (elevates fake logit relative to ablation baseline)
    - Delta_z < 0: branch contradicts FAKE (suppresses fake logit, acts as contradictory evidence)
    - max |Delta_z|: largest counterfactual branch effect
    - Verifies that negative Delta_z is NEVER assigned as dominant supporting branch.
    """
    from service.forensic_explainer import BranchForensicDetail

    branches = [
        BranchForensicDetail(
            name="semantic",
            available=True,
            router_weight=0.35,
            fake_probability=0.88,
            confidence_margin=0.76,
            directional_evidence=0.133,
            counterfactual_logit_delta=1.070,
            counterfactual_prob_delta=0.25,
            role_summary="Semantic elevated fake logit",
            evidence_direction="SUPPORTS MANIPULATION",
        ),
        BranchForensicDetail(
            name="forensic",
            available=True,
            router_weight=0.35,
            fake_probability=0.15,
            confidence_margin=0.70,
            directional_evidence=-0.122,
            counterfactual_logit_delta=-1.123,
            counterfactual_prob_delta=-0.28,
            role_summary="Forensic suppressed fake logit",
            evidence_direction="AGAINST MANIPULATION",
        ),
        BranchForensicDetail(
            name="temporal",
            available=True,
            router_weight=0.30,
            fake_probability=0.60,
            confidence_margin=0.20,
            directional_evidence=0.030,
            counterfactual_logit_delta=0.450,
            counterfactual_prob_delta=0.08,
            role_summary="Temporal moderately elevated fake logit",
            evidence_direction="SUPPORTS MANIPULATION",
        ),
    ]

    analysis = explainer._analyze_branch_attributions(branches)

    # 1. Delta_z > 0 -> branch supports FAKE
    supporting_names = [b["name"] for b in analysis["supporting_branches"]]
    assert "semantic" in supporting_names
    assert "temporal" in supporting_names
    assert "forensic" not in supporting_names

    # 2. Delta_z < 0 -> branch contradicts FAKE
    contradictory_names = [b["name"] for b in analysis["contradictory_branches"]]
    assert "forensic" in contradictory_names
    assert "semantic" not in contradictory_names
    assert "temporal" not in contradictory_names

    # 3. |Delta_z| largest -> largest counterfactual effect
    assert analysis["largest_effect_branch"] == "forensic"
    assert abs(analysis["largest_effect_delta"]) == pytest.approx(1.123, abs=1e-4)
    assert analysis["largest_effect_direction"] == "AGAINST MANIPULATION"
    assert "suppresses the fake score" in analysis["largest_effect_interpretation"]

    # 4. Crucial invariant: dominant_supporting_branch MUST NOT be forensic (-1.123)
    assert analysis["dominant_supporting_branch"] == "semantic"
    assert analysis["dominant_contradictory_branch"] == "forensic"


def test_uncertainty_deconstruction_decisive_vs_degraded(explainer):
    """
    Verifies that a decisive model prediction (e.g. p=0.9867, U=0.0477)
    with degraded media quality (Q=0.252 < 0.35):
    1. Triggers operational triage REVIEW with primary trigger QUALITY_BELOW_FLOOR
    2. Reports Uncertainty Level as LOW, NOT attributing review to Predictive Entropy
    3. Provides a coherent narrative: model is decisive, but quality is insufficient for automatic disposition.
    """
    cal_p = 0.9867
    comp_u = 0.0477
    entropy = - (cal_p * np.log2(cal_p) + (1 - cal_p) * np.log2(1 - cal_p))
    margin = 1.0 - 2.0 * abs(cal_p - 0.5)
    qual_defect = 1.0 - 0.252
    disagreement = 0.0

    u = explainer._deconstruct_uncertainty(
        composite_u=comp_u,
        entropy=entropy,
        margin=margin,
        qual_defect=qual_defect,
        disagreement=disagreement,
    )

    # Uncertainty must be characterized as LOW because U <= tau_abst (0.2661)
    assert u.uncertainty_level == "LOW"
    assert u.abstention_triggered is False
    assert "Low Model Uncertainty" in u.primary_uncertainty_driver
    assert "Predictive Entropy (Ambivalence near 0.5)" not in u.primary_uncertainty_driver
    assert "insufficient for automatic disposition" in u.uncertainty_interpretation

    # In operational triage, quality floor violation is the review trigger
    from service.forensic_explainer import QualityReliabilityAudit
    mock_qual = QualityReliabilityAudit(
        composite_quality=0.252,
        blur_sharpness=0.2,
        compression_artifact_score=0.1,
        illumination_score=0.7,
        pose_yaw_degrees=5.0,
        face_resolution_ratio=0.8,
        track_continuity=1.0,
        quality_tier="DEGRADED",
        reliability_assessment="Severe blur/compression reduces reliability, not evidence of manipulation",
    )

    triage, assess, reason_codes = explainer._assign_triage_decision(
        cal_p=cal_p,
        uncertainty=u,
        quality=mock_qual,
        disagreement=disagreement,
    )

    assert triage == "REVIEW"
    assert assess == "LIKELY_MANIPULATED"
    assert reason_codes[0] == "QUALITY_BELOW_FLOOR"

