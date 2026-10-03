"""
utils/forensic_html_report.py
Interactive HTML & Markdown Forensic Dossier Generator for FORGE-Lite V3.

Produces:
1. High-fidelity, self-contained, responsive HTML forensic dossiers
2. Clean Executive Markdown summaries for inclusion in briefs, reports, and legal packages.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Union

from service.forensic_explainer import ForensicDossier


def _esc(val: Any) -> str:
    """Escapes user and media-derived strings for safe HTML embedding."""
    if val is None:
        return ""
    return html.escape(str(val), quote=True)


def generate_forensic_markdown(dossier: ForensicDossier) -> str:
    """Generates an executive forensic markdown summary."""
    p = dossier.provenance
    u = dossier.uncertainty
    q = dossier.quality

    op_triage = getattr(dossier, "operational_triage", dossier.triage_verdict)
    model_assess = getattr(dossier, "model_assessment", "AMBIGUOUS")
    reason_codes = getattr(dossier, "triage_reason_codes", [])

    verdict_emoji = {
        "REAL": "🟢 REAL (LIKELY AUTHENTIC)",
        "FAKE": "🔴 FAKE (LIKELY MANIPULATED)",
        "REVIEW": "🟡 REVIEW (ROUTED TO HUMAN AUDIT)",
    }.get(op_triage, op_triage)

    reasons_str = ", ".join(f"`{r}`" for r in reason_codes) if reason_codes else "`NONE`"
    primary_trig = getattr(dossier, "primary_review_trigger", None) or "NONE"
    unc_level = getattr(u, "uncertainty_level", "LOW")
    unc_interp = getattr(u, "uncertainty_interpretation", "")

    md = f"""# FORGE-Lite V3 Forensic Decision-Support Dossier

## 1. Executive Triage Verdict
- **Operational Triage:** **{verdict_emoji}**  
- **Model Assessment:** `{model_assess}` (Calibrated Fake Risk: `{dossier.calibrated_fake_risk:.4f}`, Raw Logit: `{dossier.uncalibrated_raw_logit:+.4f}`)
- **Primary Review Trigger:** `{primary_trig}`
- **Triage Reason Codes:** {reasons_str}
- **Model Uncertainty:** `{unc_level} (U = {u.composite_uncertainty:.4f})` (Abstention Threshold: `0.2661`)
- **Media Quality Status:** `{q.quality_tier}` (Composite Quality: `{q.composite_quality:.4f}`, Quality Floor: `0.35`)

> **Forensic Commentary:**  
> {dossier.forensic_narrative}

---

## 2. Multi-Branch Forensic Evidence

Directional evidence reflects branch confidence and router weighting. Counterfactual branch effect measures the exact change in the final fusion logit when each branch is ablated ($\\Delta z_i = z_{{\\text{{full}}}} - z_{{\\text{{without\\_}}i}}$).
- **$\\Delta z_i > 0$:** Branch **supports manipulation** (elevates fake logit relative to ablation baseline)
- **$\\Delta z_i < 0$:** Branch **contradicts manipulation** (suppresses fake logit / supports authenticity)

| Modality Branch | Status | Fake Prob ($p_i$) | Router Weight ($w_i$) | Directional Evidence ($C_i$) | Counterfactual Branch Effect ($\\Delta z_i$) | Evidence Direction | Role Summary |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""

    for b in dossier.branch_attributions:
        status = "Active" if b.available else "N/A"
        dir_label = getattr(b, "evidence_direction", "NEUTRAL")
        md += (
            f"| **{b.name.capitalize()}** | {status} | {b.fake_probability:.2f} | "
            f"{b.router_weight*100:.1f}% | {b.directional_evidence:+.4f} | "
            f"**`{b.counterfactual_logit_delta:+.4f}`** | `{dir_label}` | {b.role_summary} |\n"
        )

    # Build Supporting / Contradictory evidence lists
    supporting_lines = []
    for sb in getattr(dossier, "supporting_branches", []):
        supporting_lines.append(f"  - **{sb['name'].capitalize()}**: `+{abs(sb['counterfactual_logit_delta']):.4f}`")
    if not supporting_lines:
        supporting_lines.append("  - *None (no branches elevated fake logit)*")

    contradictory_lines = []
    for cb in getattr(dossier, "contradictory_branches", []):
        contradictory_lines.append(f"  - **{cb['name'].capitalize()}**: `{cb['counterfactual_logit_delta']:+.4f}`")
    if not contradictory_lines:
        contradictory_lines.append("  - *None (no branches suppressed fake logit)*")

    supporting_block = "\n".join(supporting_lines)
    contradictory_block = "\n".join(contradictory_lines)

    largest_branch = getattr(dossier, "largest_effect_branch", "None").capitalize()
    largest_delta = getattr(dossier, "largest_effect_delta", 0.0)
    largest_dir = getattr(dossier, "largest_effect_direction", "NEUTRAL")
    largest_interp = getattr(dossier, "largest_effect_interpretation", "")

    md += f"""
### Directional Evidence Breakdown

**Supporting Evidence ($\\Delta z_i > 0$):**
{supporting_block}

**Contradictory Evidence ($\\Delta z_i < 0$):**
{contradictory_block}

**Largest Branch Effect:**
- **Modality:** `{largest_branch}`
- **Magnitude:** `|Δz| = {abs(largest_delta):.4f}` (Signed `Δz = {largest_delta:+.4f}`)
- **Direction:** **`{largest_dir}`**
- **Interpretation:** *{largest_interp}*

---

## 3. Uncertainty Deconstruction & Review Audit

- **Primary Review Trigger:** `{primary_trig}`
- **Uncertainty Level:** `{unc_level} (U = {u.composite_uncertainty:.4f})`
- **Interpretation:** *{unc_interp}*

Composite uncertainty ($U = {u.composite_uncertainty:.4f}$) is deconstructed into:
- **Predictive Entropy $H(p)$:** `{u.predictive_entropy:.4f}` (ambivalence near decision boundary)
- **Decision Margin Proximity:** `{u.decision_margin_uncertainty:.4f}` ($1 - 2|p - 0.5|$)
- **Media Quality Defect:** `{u.quality_defect:.4f}` ($1 - Q$)
- **Branch Disagreement:** `{u.branch_disagreement:.4f}` (cross-modal dispersion $\\sigma$)
- **Primary Uncertainty Driver:** `{u.primary_uncertainty_driver}`

---

## 4. Media Quality & Reliability Audit

| Metric | Score | Assessment |
| :--- | :---: | :--- |
| **Sharpness / Blur Level** | `{q.blur_sharpness:.4f}` | Laplacian variance proxy |
| **Compression Artifact Severity** | `{q.compression_artifact_score:.4f}` | High-frequency block quantization |
| **Illumination Uniformity** | `{q.illumination_score:.4f}` | Mean luminance balance |
| **Face Resolution Ratio** | `{q.face_resolution_ratio:.4f}` | Bounding box area ratio |
| **Track Continuity** | `{q.track_continuity:.4f}` | Face tracker temporal persistence |

> [!NOTE]
> **Reliability Clarification**  
> {q.reliability_assessment}

---
"""

    if dossier.temporal_aggregation:
        ta = dossier.temporal_aggregation
        md += f"""## 5. Video Temporal Track Aggregation
- **Frames Analyzed:** {ta.total_frames_analyzed}
- **Windows Evaluated:** {ta.total_windows_evaluated}
- **Median Fake Risk:** `{ta.median_fake_score:.4f}` (75th percentile: `{ta.p75_fake_score:.4f}`, dispersion: `{ta.score_dispersion:.4f}`)
"""
        if ta.peak_anomaly_windows:
            md += "\n**Peak Anomaly Intervals:**\n"
            for pw in ta.peak_anomaly_windows:
                md += f"- Interval `{pw['window_label']}`: Calibrated Risk `{pw['calibrated_fake_risk']:.4f}`\n"
        md += "\n---\n"

    md += f"""## 6. Cryptographic Audit & Provenance Trail
```text
Input SHA-256:          {p.input_sha256}
Checkpoint SHA-256:     {p.checkpoint_sha256}
Model Version:          {p.model_version}
Calibration:            {p.calibration_version}
Uncertainty Engine:     {p.uncertainty_version}
Schema Version:         {p.report_schema_version}
Analysis Timestamp:     {p.analysis_timestamp_utc}
Media Type:             {p.media_type.upper()}
Execution Device:       {p.execution_device}
```

> **Forensic Decision-Support Disclaimer**  
> {dossier.operational_disclaimer}
"""
    return md


def generate_forensic_html(dossier: ForensicDossier, title: str = "FORGE-Lite V3 Forensic Dossier") -> str:
    """
    Generates a standalone, responsive, modern dark-mode HTML forensic dossier.
    All dynamic strings are escaped to protect against markup/script injection.
    """
    p = dossier.provenance
    u = dossier.uncertainty
    q = dossier.quality

    op_triage = getattr(dossier, "operational_triage", dossier.triage_verdict)
    model_assess = getattr(dossier, "model_assessment", "AMBIGUOUS")
    reason_codes = getattr(dossier, "triage_reason_codes", [])

    verdict_color = {
        "REAL": "#22c55e",
        "FAKE": "#ef4444",
        "REVIEW": "#f59e0b",
    }.get(op_triage, "#64748b")

    verdict_sub = {
        "REAL": "LIKELY AUTHENTIC BIOMETRICS",
        "FAKE": "LIKELY MANIPULATED MEDIA",
        "REVIEW": "ROUTED TO HUMAN REVIEW (SELECTIVE ABSTENTION)",
    }.get(op_triage, op_triage)

    # Operational highlight banner for quality-limited or ambiguous cases
    op_highlight = ""
    if op_triage == "REVIEW":
        if dossier.calibrated_fake_risk >= 0.60:
            op_highlight = (
                '<div class="op-highlight">'
                '<strong>Operational Notice:</strong> High manipulation-risk signal detected, '
                'but evidence quality is insufficient for automatic disposition.'
                '</div>'
            )
        elif dossier.calibrated_fake_risk < 0.40:
            op_highlight = (
                '<div class="op-highlight">'
                '<strong>Operational Notice:</strong> Authentic baseline pattern detected, '
                'but evidence quality is insufficient for automatic disposition.'
                '</div>'
            )

    # Reason code badge pills
    reason_badges = "".join(
        f'<span class="badge badge-reason">{_esc(rc)}</span>'
        for rc in reason_codes
    ) if reason_codes else '<span class="badge badge-reason">CLEAR_SIGNAL</span>'

    # Build branch rows
    branch_rows = ""
    for b in dossier.branch_attributions:
        avail_badge = (
            '<span class="badge badge-active">ACTIVE</span>'
            if b.available
            else '<span class="badge badge-na">N/A</span>'
        )
        bar_pct = max(0, min(100, int(b.fake_probability * 100)))
        bar_color = "#ef4444" if b.fake_probability >= 0.5 else "#22c55e"
        cf_color = "#ef4444" if b.counterfactual_logit_delta > 0 else "#22c55e"
        dir_label = getattr(b, "evidence_direction", "NEUTRAL")
        dir_color = "#ef4444" if b.counterfactual_logit_delta > 0 else ("#38bdf8" if b.counterfactual_logit_delta < 0 else "#94a3b8")

        branch_rows += f"""
        <tr>
            <td><strong>{_esc(b.name.capitalize())}</strong></td>
            <td>{avail_badge}</td>
            <td>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div class="progress-bar-bg" style="width:70px;">
                        <div class="progress-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
                    </div>
                    <span>{b.fake_probability:.2f}</span>
                </div>
            </td>
            <td>{b.router_weight * 100:.1f}%</td>
            <td>{b.directional_evidence:+.3f}</td>
            <td style="color:{cf_color}; font-weight:700;">{b.counterfactual_logit_delta:+.3f}</td>
            <td style="color:{dir_color}; font-weight:600; font-size:0.8rem;">{_esc(dir_label)}</td>
            <td style="font-size:0.85rem; color:#94a3b8;">{_esc(b.role_summary)}</td>
        </tr>
        """

    # Build supporting / contradictory evidence HTML blocks
    supporting_html = "".join(
        f'<div style="margin-bottom:6px; display:flex; justify-content:space-between; max-width:280px;">'
        f'<span><strong>{_esc(sb["name"].capitalize())}</strong></span>'
        f'<span style="color:#ef4444; font-family:monospace; font-weight:700;">+{abs(sb["counterfactual_logit_delta"]):.3f}</span>'
        f'</div>'
        for sb in getattr(dossier, "supporting_branches", [])
    ) or '<div style="color:#64748b; font-style:italic;">None (no branches elevated fake logit)</div>'

    contradictory_html = "".join(
        f'<div style="margin-bottom:6px; display:flex; justify-content:space-between; max-width:280px;">'
        f'<span><strong>{_esc(cb["name"].capitalize())}</strong></span>'
        f'<span style="color:#38bdf8; font-family:monospace; font-weight:700;">{cb["counterfactual_logit_delta"]:+.3f}</span>'
        f'</div>'
        for cb in getattr(dossier, "contradictory_branches", [])
    ) or '<div style="color:#64748b; font-style:italic;">None (no branches suppressed fake logit)</div>'

    largest_branch_str = getattr(dossier, "largest_effect_branch", "None").capitalize()
    largest_delta_val = getattr(dossier, "largest_effect_delta", 0.0)
    largest_dir_str = getattr(dossier, "largest_effect_direction", "NEUTRAL")
    largest_interp_str = getattr(dossier, "largest_effect_interpretation", "")
    largest_dir_color = "#ef4444" if largest_delta_val > 0 else ("#38bdf8" if largest_delta_val < 0 else "#94a3b8")

    primary_trigger_badge = (
        f'<div style="margin-top:6px; font-size:0.82rem; color:#f59e0b;">'
        f'Primary Review Trigger: <strong>{_esc(dossier.primary_review_trigger)}</strong>'
        f'</div>'
    ) if getattr(dossier, "primary_review_trigger", None) else ''

    # Video temporal section
    video_section = ""
    if dossier.temporal_aggregation:
        ta = dossier.temporal_aggregation
        peak_items = ""
        for pw in ta.peak_anomaly_windows:
            peak_items += f"""
            <div class="anomaly-pill">
                <span>Interval <strong>{_esc(pw['window_label'])}</strong></span>
                <span style="color:#ef4444; font-weight:700;">Risk {pw['calibrated_fake_risk']:.2f}</span>
            </div>
            """
        video_section = f"""
        <div class="card">
            <h3>Video Temporal Track Aggregation</h3>
            <div class="grid grid-3" style="margin-bottom:12px;">
                <div class="metric-box">
                    <span class="metric-label">Frames Analyzed</span>
                    <span class="metric-val">{ta.total_frames_analyzed}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Track Median Risk</span>
                    <span class="metric-val">{ta.median_fake_score:.2f}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Score Dispersion (std)</span>
                    <span class="metric-val">{ta.score_dispersion:.3f}</span>
                </div>
            </div>
            {f'<div style="margin-top:10px;"><strong>Peak Anomaly Windows:</strong><div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">{peak_items}</div></div>' if peak_items else ''}
        </div>
        """

    # Image Diagnostics gallery
    images_gallery = ""
    if dossier.aligned_face_b64 or dossier.srm_residual_b64 or dossier.attention_heatmap_b64:
        images_gallery = f"""
        <div class="card">
            <h3>Visual Diagnostics (Diagnostic Evidence Only)</h3>
            <div class="grid grid-3" style="margin-top:12px;">
                {f'''<div class="img-card">
                    <img src="{dossier.aligned_face_b64}" alt="Aligned Face Crop" />
                    <div class="img-label">Aligned Facial Crop</div>
                </div>''' if dossier.aligned_face_b64 else ''}
                {f'''<div class="img-card">
                    <img src="{dossier.srm_residual_b64}" alt="SRM Residual Noise" />
                    <div class="img-label">SRM High-Frequency Residual</div>
                </div>''' if dossier.srm_residual_b64 else ''}
                {f'''<div class="img-card">
                    <img src="{dossier.attention_heatmap_b64}" alt="Attention Heatmap" />
                    <div class="img-label">Model Attention / Anomaly Saliency</div>
                </div>''' if dossier.attention_heatmap_b64 else ''}
            </div>
        </div>
        """

    dominant_name = _esc(dossier.dominant_supporting_branch.capitalize())
    contra_name = _esc(dossier.dominant_contradictory_branch.capitalize()) if dossier.dominant_contradictory_branch else "None"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_esc(title)}</title>
    <style>
        :root {{
            --bg: #090d16;
            --card-bg: #111827;
            --border: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --verdict-color: {verdict_color};
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
        body {{ background: var(--bg); color: var(--text); padding: 24px; font-size: 15px; line-height: 1.5; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }}
        .logo-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.5px; }}
        .logo-sub {{ font-size: 0.8rem; color: var(--primary); text-transform: uppercase; letter-spacing: 1px; }}
        .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
        .verdict-card {{ border-left: 6px solid var(--verdict-color); display: flex; justify-content: space-between; align-items: center; gap: 20px; }}
        .verdict-badge {{ font-size: 2.2rem; font-weight: 800; color: var(--verdict-color); letter-spacing: 0.5px; line-height: 1.1; }}
        .verdict-sub {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
        .model-assess-tag {{ font-size: 0.85rem; color: #cbd5e1; margin-top: 6px; }}
        .model-assess-val {{ font-weight: 700; color: #fff; }}
        .op-highlight {{ background: rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 4px; font-size: 0.88rem; color: #fef3c7; margin-top: 14px; }}
        .grid {{ display: grid; gap: 16px; }}
        .grid-2 {{ grid-template-columns: 1fr 1fr; }}
        .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
        .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
        .metric-box {{ background: #1e293b; border-radius: 8px; padding: 12px; text-align: center; }}
        .metric-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; display: block; }}
        .metric-val {{ font-size: 1.4rem; font-weight: 700; color: #fff; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #1e293b; font-size: 0.9rem; }}
        th {{ background: #1e293b; color: var(--text-muted); text-transform: uppercase; font-size: 0.75rem; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }}
        .badge-active {{ background: #065f46; color: #34d399; }}
        .badge-na {{ background: #374151; color: #9ca3af; }}
        .badge-reason {{ background: #1e293b; border: 1px solid #475569; color: #cbd5e1; font-family: monospace; font-size: 0.72rem; padding: 2px 7px; }}
        .progress-bar-bg {{ background: #374151; height: 8px; border-radius: 4px; overflow: hidden; }}
        .progress-bar-fill {{ height: 100%; border-radius: 4px; }}
        .narrative {{ background: #0f172a; border-left: 4px solid var(--primary); padding: 14px; border-radius: 4px; font-size: 0.95rem; margin-top: 14px; }}
        .disclaimer {{ font-size: 0.75rem; color: #64748b; line-height: 1.4; border-top: 1px solid var(--border); padding-top: 16px; margin-top: 24px; }}
        .provenance-text {{ font-family: monospace; font-size: 0.8rem; color: #94a3b8; line-height: 1.6; }}
        .img-card {{ text-align: center; background: #1e293b; padding: 10px; border-radius: 8px; }}
        .img-card img {{ width: 100%; height: auto; border-radius: 4px; display: block; }}
        .img-label {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 8px; font-weight: 600; text-transform: uppercase; }}
        .anomaly-pill {{ background: #1e293b; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; display: flex; gap: 10px; font-size: 0.85rem; }}
        @media(max-width: 768px) {{
            .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
            .verdict-card {{ flex-direction: column; align-items: flex-start; gap: 14px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <div class="logo-title">FORGE-Lite V3 Forensic Dossier</div>
                <div class="logo-sub">Multi-Modal Decision-Support Audit System</div>
            </div>
            <div style="text-align:right;">
                <span class="badge badge-active">FROZEN V3 VERIFIED</span>
                <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">{_esc(p.analysis_timestamp_utc)}</div>
            </div>
        </div>

        <!-- Triage Verdict Banner -->
        <div class="card verdict-card">
            <div>
                <div class="verdict-sub">Operational Triage</div>
                <div class="verdict-badge">{_esc(op_triage)}</div>
                <div style="color:var(--text-muted); font-size:0.85rem; margin-top:4px;">{_esc(verdict_sub)}</div>
                <div class="model-assess-tag">Model Assessment: <span class="model-assess-val">{_esc(model_assess)}</span></div>
                {primary_trigger_badge}
                <div style="margin-top:10px; display:flex; gap:6px; flex-wrap:wrap;">
                    {reason_badges}
                </div>
            </div>
            <div class="grid grid-3" style="min-width: 420px;">
                <div class="metric-box">
                    <span class="metric-label">Calibrated Risk</span>
                    <span class="metric-val">{dossier.calibrated_fake_risk:.2f}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Uncertainty</span>
                    <span class="metric-val" style="font-size:1.1rem;">{_esc(getattr(u, "uncertainty_level", "LOW"))} ({u.composite_uncertainty:.2f})</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Media Quality</span>
                    <span class="metric-val">{_esc(q.quality_tier)}</span>
                </div>
            </div>
        </div>

        {op_highlight}

        <!-- Forensic Narrative -->
        <div class="card">
            <h3>Forensic Decision-Support Commentary</h3>
            <div class="narrative">{_esc(dossier.forensic_narrative)}</div>
        </div>

        <!-- Visual Evidence Attribution Table -->
        <div class="card">
            <h3>Multi-Branch Evidence Attribution</h3>
            <p style="font-size:0.85rem; color:var(--text-muted); margin-top:4px;">
                Directional evidence (C<sub>i</sub>) reflects branch confidence. Counterfactual branch effect (&Delta;z<sub>i</sub>) 
                measures the exact change in the final logit when that branch is ablated (&Delta;z<sub>i</sub> = z<sub>full</sub> &minus; z<sub>without_i</sub>).
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Modality</th>
                        <th>Status</th>
                        <th>Fake Prob</th>
                        <th>Router Wt</th>
                        <th>Directional (C<sub>i</sub>)</th>
                        <th>Counterfactual Branch Effect (&Delta;z<sub>i</sub>)</th>
                        <th>Direction</th>
                        <th>Forensic Role</th>
                    </tr>
                </thead>
                <tbody>
                    {branch_rows}
                </tbody>
            </table>
            
            <!-- Directional Evidence Breakdown -->
            <div style="background:#0b1120; border:1px solid #1e293b; border-radius:8px; padding:16px; margin-top:16px;">
                <div style="font-weight:700; color:#cbd5e1; font-size:0.95rem; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;">Directional Evidence Breakdown</div>
                <div class="grid grid-2" style="font-size:0.88rem; margin-bottom:14px;">
                    <div>
                        <div style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; font-weight:700; margin-bottom:6px;">Supporting Evidence (&Delta;z<sub>i</sub> &gt; 0)</div>
                        {supporting_html}
                    </div>
                    <div>
                        <div style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; font-weight:700; margin-bottom:6px;">Contradictory Evidence (&Delta;z<sub>i</sub> &lt; 0)</div>
                        {contradictory_html}
                    </div>
                </div>
                <div style="border-top:1px solid #1e293b; padding-top:10px; font-size:0.88rem;">
                    <div>
                        <strong>Largest Branch Effect:</strong> {_esc(largest_branch_str)} 
                        <span style="font-family:monospace; font-weight:700;">(|&Delta;z| = {abs(largest_delta_val):.3f}, Signed: {largest_delta_val:+.3f})</span>
                        &nbsp;&bull;&nbsp; Direction: <span style="font-weight:700; color:{largest_dir_color};">{_esc(largest_dir_str)}</span>
                    </div>
                    <div style="font-size:0.84rem; color:#94a3b8; margin-top:4px;">
                        {_esc(largest_interp_str)}
                    </div>
                </div>
            </div>
        </div>

        <!-- Uncertainty Deconstruction & Quality Audit (Side-by-Side) -->
        <div class="grid grid-2">
            <!-- Uncertainty 4-Factor Card -->
            <div class="card">
                <h3>Uncertainty Deconstruction (4-Factor)</h3>
                <div style="margin-top:10px; font-size:0.86rem; border-bottom:1px solid #1e293b; padding-bottom:10px; margin-bottom:12px;">
                    {f'<div style="color:#f59e0b; margin-bottom:4px;"><strong>Primary Review Trigger:</strong> {_esc(dossier.primary_review_trigger)}</div>' if getattr(dossier, "primary_review_trigger", None) else ''}
                    <div><strong>Uncertainty Level:</strong> <span style="color:#38bdf8; font-weight:700;">{_esc(getattr(u, "uncertainty_level", "LOW"))} (U = {u.composite_uncertainty:.4f})</span></div>
                    <div style="color:#94a3b8; margin-top:4px; font-style:italic;">{_esc(getattr(u, "uncertainty_interpretation", ""))}</div>
                </div>
                <div class="grid grid-2">
                    <div class="metric-box">
                        <span class="metric-label">Predictive Entropy</span>
                        <span class="metric-val">{u.predictive_entropy:.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Margin Proximity</span>
                        <span class="metric-val">{u.decision_margin_uncertainty:.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Quality Defect</span>
                        <span class="metric-val">{u.quality_defect:.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Branch Disagree</span>
                        <span class="metric-val">{u.branch_disagreement:.2f}</span>
                    </div>
                </div>
                <div style="margin-top:14px; font-size:0.85rem;">
                    <strong>Primary Uncertainty Driver:</strong><br>
                    <span style="color:var(--primary);">{_esc(u.primary_uncertainty_driver)}</span>
                </div>
            </div>

            <!-- Quality Reliability Card -->
            <div class="card">
                <h3>Media Quality & Reliability Audit</h3>
                <div class="grid grid-2" style="margin-top:12px;">
                    <div class="metric-box">
                        <span class="metric-label">Sharpness</span>
                        <span class="metric-val">{q.blur_sharpness:.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Compression Q</span>
                        <span class="metric-val">{q.compression_artifact_score:.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Illumination</span>
                        <span class="metric-val">{q.illumination_score:.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Face Resolution</span>
                        <span class="metric-val">{q.face_resolution_ratio:.2f}</span>
                    </div>
                </div>
                <div style="margin-top:14px; font-size:0.85rem; color:#94a3b8;">
                    {_esc(q.reliability_assessment)}
                </div>
            </div>
        </div>

        {video_section}

        {images_gallery}

        <!-- Provenance Header Card -->
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3>Cryptographic Audit & Provenance Trail</h3>
                <span class="badge" style="background:#1e293b; color:#38bdf8; border:1px solid #38bdf8;">HTML-escaped against tested injection cases</span>
            </div>
            <div class="provenance-text" style="margin-top:10px;">
                Input SHA-256:          {_esc(p.input_sha256)}<br>
                Checkpoint SHA-256:     {_esc(p.checkpoint_sha256)}<br>
                Model Version:          {_esc(p.model_version)} | Calibration: {_esc(p.calibration_version)} | Uncertainty: {_esc(p.uncertainty_version)}<br>
                Schema Version:         {_esc(p.report_schema_version)} | Execution Device: {_esc(p.execution_device)}<br>
                Analysis Timestamp:     {_esc(p.analysis_timestamp_utc)}
            </div>
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer">
            {_esc(dossier.operational_disclaimer)}
        </div>
    </div>
</body>
</html>
"""
    return html_content


def save_forensic_dossier(
    dossier: ForensicDossier,
    output_prefix: Union[str, Path],
    save_html: bool = True,
    save_md: bool = True,
    save_json: bool = True,
) -> Dict[str, Path]:
    """
    Saves the dossier in JSON, Markdown, and/or HTML formats.
    """
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    if save_json:
        json_path = prefix.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(dossier.to_json(indent=2))
        saved_files["json"] = json_path

    if save_md:
        md_path = prefix.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_forensic_markdown(dossier))
        saved_files["markdown"] = md_path

    if save_html:
        html_path = prefix.with_suffix(".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(generate_forensic_html(dossier))
        saved_files["html"] = html_path

    return saved_files
