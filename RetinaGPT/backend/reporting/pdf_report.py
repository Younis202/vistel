"""
pdf_report.py — Clinical PDF Report Generator for Retina-GPT
=============================================================
Generates professional, print-ready medical reports for ophthalmologists.

Report includes:
    • Patient information header
    • Retinal image + Grad-CAM overlay
    • Quantitative findings table
    • DR grading with severity scale
    • Lesion detection summary
    • Longitudinal progression chart (if available)
    • Clinical recommendations
    • AI confidence indicators
    • Disclaimer

Uses ReportLab for PDF generation (no browser/LaTeX dependency).

Usage:
    generator = ClinicalPDFGenerator()

    report_path = generator.generate(
        output_path="reports/patient_001.pdf",
        patient_info={"id": "P001", "name": "...", "dob": "1965-03-12"},
        analysis_result=model_output,
        original_image=image_rgb,
        explanation_panel=explainer_output,
    )
"""

import os
import io
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────

class ReportColors:
    """Clinical report color scheme."""
    NAVY      = (0.05,  0.10,  0.30)
    BLUE      = (0.15,  0.35,  0.70)
    LIGHT_BLUE= (0.85,  0.91,  0.97)
    GREEN     = (0.12,  0.58,  0.30)
    AMBER     = (0.80,  0.55,  0.05)
    RED       = (0.75,  0.10,  0.10)
    GRAY      = (0.50,  0.50,  0.50)
    LIGHT_GRAY= (0.93,  0.93,  0.93)
    WHITE     = (1.00,  1.00,  1.00)
    BLACK     = (0.05,  0.05,  0.05)

    @staticmethod
    def risk_color(risk_level: str) -> Tuple[float, float, float]:
        return {
            "low":      ReportColors.GREEN,
            "moderate": ReportColors.AMBER,
            "high":     ReportColors.RED,
            "critical": (0.50, 0.00, 0.00),
        }.get(risk_level.lower(), ReportColors.GRAY)

    @staticmethod
    def grade_color(grade: int) -> Tuple[float, float, float]:
        colors = [
            ReportColors.GREEN,         # 0 - No DR
            (0.60, 0.80, 0.20),         # 1 - Mild
            ReportColors.AMBER,         # 2 - Moderate
            (0.90, 0.40, 0.00),         # 3 - Severe
            ReportColors.RED,           # 4 - Proliferative
        ]
        return colors[min(grade, 4)]


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Generator
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalPDFGenerator:
    """
    Professional clinical PDF report generator.

    Falls back gracefully if ReportLab is not installed:
    generates a structured text report instead.

    Usage:
        gen = ClinicalPDFGenerator(
            clinic_name="Cairo University Hospital",
            clinic_logo_path="assets/logo.png",
        )

        path = gen.generate(
            output_path="report.pdf",
            patient_info={"id": "P-001", "age": 55, "sex": "M"},
            analysis_result=model.analyze(image),
            original_image=image_rgb,
        )
    """

    DR_GRADE_LABELS = {
        0: "No Diabetic Retinopathy",
        1: "Mild Non-Proliferative DR",
        2: "Moderate Non-Proliferative DR",
        3: "Severe Non-Proliferative DR",
        4: "Proliferative Diabetic Retinopathy",
    }

    AMD_STAGE_LABELS = {
        0: "No AMD",
        1: "Early AMD",
        2: "Intermediate AMD",
        3: "Late AMD",
    }

    def __init__(
        self,
        clinic_name:     str = "Retina-GPT AI Clinic",
        clinic_logo_path: Optional[str] = None,
        footer_text:     str = "This report is AI-assisted. Clinical correlation required.",
        page_size:       str = "A4",
    ):
        self.clinic_name      = clinic_name
        self.clinic_logo_path = clinic_logo_path
        self.footer_text      = footer_text
        self.page_size_name   = page_size
        self._has_reportlab   = self._check_reportlab()

    def _check_reportlab(self) -> bool:
        try:
            import reportlab
            return True
        except ImportError:
            return False

    def generate(
        self,
        output_path:        str,
        patient_info:       Dict[str, Any],
        analysis_result:    Dict[str, Any],
        original_image:     Optional[np.ndarray] = None,
        explanation_panel:  Optional[np.ndarray] = None,
        progression_report: Optional[Any] = None,
        include_gradcam:    bool = True,
    ) -> str:
        """
        Generate a complete clinical PDF report.

        Args:
            output_path:       where to save the PDF
            patient_info:      dict with id, name, age, sex, referring_physician
            analysis_result:   dict output from RetinaGPTFoundationModel.analyze()
            original_image:    (H,W,3) uint8 RGB fundus image
            explanation_panel: (H,W*4,3) Grad-CAM panel from RetinaExplainer
            progression_report: ProgressionReport from Retina-TIME (optional)
            include_gradcam:   include explainability panel in report

        Returns:
            output_path (str) if successful
        """
        os.makedirs(Path(output_path).parent, exist_ok=True)

        if self._has_reportlab:
            return self._generate_pdf(
                output_path, patient_info, analysis_result,
                original_image, explanation_panel,
                progression_report, include_gradcam,
            )
        else:
            # Fallback to text report
            txt_path = output_path.replace(".pdf", ".txt")
            return self._generate_text(
                txt_path, patient_info, analysis_result, progression_report
            )

    # ── ReportLab PDF ──────────────────────────────────────────────────────

    def _generate_pdf(
        self, output_path, patient_info, analysis_result,
        original_image, explanation_panel, progression_report, include_gradcam,
    ) -> str:
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.units import cm, mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, HRFlowable, KeepTogether,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.pdfgen import canvas

        page_size = A4 if self.page_size_name == "A4" else letter
        W, H = page_size

        doc = SimpleDocTemplate(
            output_path, pagesize=page_size,
            topMargin=1.5*cm, bottomMargin=2*cm,
            leftMargin=1.8*cm, rightMargin=1.8*cm,
        )

        styles = getSampleStyleSheet()
        story  = []

        # ── Define custom styles ───────────────────────────────────────────
        def c(rgb): return colors.Color(*rgb)

        header_style = ParagraphStyle(
            "HeaderStyle", parent=styles["Normal"],
            fontSize=18, fontName="Helvetica-Bold",
            textColor=c(ReportColors.NAVY), spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "SubtitleStyle", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica",
            textColor=c(ReportColors.GRAY),
        )
        section_style = ParagraphStyle(
            "SectionStyle", parent=styles["Normal"],
            fontSize=12, fontName="Helvetica-Bold",
            textColor=c(ReportColors.NAVY), spaceBefore=8, spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "BodyStyle", parent=styles["Normal"],
            fontSize=9, fontName="Helvetica", leading=13,
        )
        finding_style = ParagraphStyle(
            "FindingStyle", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica",
            leftIndent=10, leading=14,
        )
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle", parent=styles["Normal"],
            fontSize=7, fontName="Helvetica-Oblique",
            textColor=c(ReportColors.GRAY), alignment=TA_CENTER,
        )

        # ── Header ─────────────────────────────────────────────────────────
        story.append(Paragraph(self.clinic_name, header_style))
        story.append(Paragraph("AI-Assisted Retinal Analysis Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=c(ReportColors.BLUE)))
        story.append(Spacer(1, 0.3*cm))

        # ── Patient Info Table ─────────────────────────────────────────────
        pid   = patient_info.get("id",   "—")
        pname = patient_info.get("name", "—")
        page  = patient_info.get("age",  "—")
        sex   = patient_info.get("sex",  "—")
        ref   = patient_info.get("referring_physician", "—")
        exam_date = datetime.now().strftime("%B %d, %Y")
        exam_time = datetime.now().strftime("%H:%M")

        info_data = [
            ["Patient ID:", pid,       "Examination Date:", exam_date],
            ["Patient Name:", pname,   "Time:", exam_time],
            ["Age / Sex:", f"{page} / {sex}", "Referring Physician:", ref],
        ]
        info_table = Table(info_data, colWidths=[3.5*cm, 5.5*cm, 4.5*cm, 5.5*cm])
        info_table.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTNAME",    (2,0), (2,-1),  "Helvetica-Bold"),
            ("BACKGROUND",  (0,0), (-1,-1), c(ReportColors.LIGHT_GRAY)),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [c(ReportColors.LIGHT_GRAY), c(ReportColors.WHITE)]),
            ("GRID",        (0,0), (-1,-1), 0.3, c(ReportColors.GRAY)),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Fundus Image + GradCAM ─────────────────────────────────────────
        if original_image is not None or explanation_panel is not None:
            story.append(Paragraph("RETINAL IMAGES", section_style))
            img_cells = []

            if original_image is not None:
                img_buf = self._array_to_image_buffer(original_image, size=(300, 300))
                if img_buf:
                    rl_img = RLImage(img_buf, width=7*cm, height=7*cm)
                    img_cells.append([rl_img, Paragraph("Original Fundus Image", body_style)])

            if explanation_panel is not None and include_gradcam:
                # Use only the first 2 panels (original + gradcam) for space
                h, w = explanation_panel.shape[:2]
                panel_half = explanation_panel[:, :w//2, :]
                buf = self._array_to_image_buffer(panel_half, size=(300, 300))
                if buf:
                    rl_img = RLImage(buf, width=7*cm, height=7*cm)
                    img_cells.append([rl_img, Paragraph("Grad-CAM Visualization\n(highlighted decision regions)", body_style)])

            if img_cells:
                flat = [cell for row in img_cells for cell in row]
                img_table = Table([flat], colWidths=[7.5*cm] * len(img_cells))
                img_table.setStyle(TableStyle([
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("ALIGN",  (0,0), (-1,-1), "CENTER"),
                    ("TOPPADDING", (0,0), (-1,-1), 4),
                ]))
                story.append(img_table)
                story.append(Spacer(1, 0.3*cm))

        # ── AI Analysis Results ────────────────────────────────────────────
        story.append(Paragraph("AI ANALYSIS RESULTS", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c(ReportColors.LIGHT_BLUE)))
        story.append(Spacer(1, 0.2*cm))

        # Quality badge
        quality = analysis_result.get("quality", {})
        q_score = quality.get("score", None)
        q_adequate = quality.get("adequate", True)
        if isinstance(q_score, float):
            q_text = f"{'✓' if q_adequate else '⚠'} Image Quality: {q_score:.0%}"
            q_color = ReportColors.GREEN if q_adequate else ReportColors.AMBER
            story.append(Paragraph(f"<b>{q_text}</b>", ParagraphStyle(
                "QStyle", parent=body_style,
                textColor=c(q_color), fontSize=10,
            )))
            story.append(Spacer(1, 0.2*cm))

        # Main findings table
        findings_data = [["Finding", "Result", "Confidence", "Status"]]

        # DR Grading
        dr = analysis_result.get("dr", {})
        dr_grade = dr.get("grade", None)
        if dr_grade is not None:
            if hasattr(dr_grade, "item"):
                dr_grade = dr_grade.item()
            dr_grade = int(dr_grade)
            dr_label = self.DR_GRADE_LABELS.get(dr_grade, f"Grade {dr_grade}")
            dr_conf  = dr.get("confidence", 0)
            if hasattr(dr_conf, "item"):
                dr_conf = dr_conf.item()
            refer = "REFER" if dr_grade >= 2 else "ROUTINE"
            findings_data.append([
                "Diabetic Retinopathy", dr_label, f"{float(dr_conf):.0%}", refer
            ])

        # AMD
        amd = analysis_result.get("amd", {})
        amd_stage = amd.get("stage", None)
        if amd_stage is not None:
            if hasattr(amd_stage, "item"):
                amd_stage = amd_stage.item()
            amd_stage = int(amd_stage)
            amd_label = self.AMD_STAGE_LABELS.get(amd_stage, f"Stage {amd_stage}")
            amd_conf  = amd.get("confidence", 0)
            if hasattr(amd_conf, "item"):
                amd_conf = amd_conf.item()
            status = "REFER" if amd_stage >= 3 else "MONITOR" if amd_stage > 0 else "NORMAL"
            findings_data.append([
                "Age-Related Macular Degen.", amd_label, f"{float(amd_conf):.0%}", status
            ])

        # Glaucoma
        glau = analysis_result.get("glaucoma", {})
        if glau:
            suspect = glau.get("suspect", False)
            if hasattr(suspect, "item"):
                suspect = suspect.item()
            cdr = glau.get("cup_disc_ratio", 0)
            if hasattr(cdr, "item"):
                cdr = cdr.item()
            conf = glau.get("confidence", 0)
            if hasattr(conf, "item"):
                conf = conf.item()
            findings_data.append([
                "Glaucoma Suspect",
                f"{'Suspect' if bool(suspect) else 'No suspicion'} (CDR={float(cdr):.2f})",
                f"{float(conf):.0%}",
                "REFER" if bool(suspect) else "NORMAL",
            ])

        col_widths = [5.5*cm, 6.0*cm, 3.0*cm, 3.5*cm]
        findings_table = Table(findings_data, colWidths=col_widths)

        def status_color(status):
            if "REFER" in status or "URGENT" in status:
                return c(ReportColors.RED)
            if "MONITOR" in status:
                return c(ReportColors.AMBER)
            return c(ReportColors.GREEN)

        table_style = TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  c(ReportColors.NAVY)),
            ("TEXTCOLOR",     (0,0), (-1,0),  c(ReportColors.WHITE)),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [c(ReportColors.WHITE), c(ReportColors.LIGHT_GRAY)]),
            ("GRID",          (0,0), (-1,-1), 0.3, c(ReportColors.GRAY)),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ])

        # Color status column
        for row_idx in range(1, len(findings_data)):
            status = findings_data[row_idx][-1]
            findings_table.setStyle(TableStyle([
                ("TEXTCOLOR",  (3, row_idx), (3, row_idx), status_color(status)),
                ("FONTNAME",   (3, row_idx), (3, row_idx), "Helvetica-Bold"),
            ]))

        findings_table.setStyle(table_style)
        story.append(findings_table)
        story.append(Spacer(1, 0.3*cm))

        # ── Lesion Summary ─────────────────────────────────────────────────
        lesions = analysis_result.get("lesions", {})
        if lesions:
            story.append(Paragraph("LESION DETECTION", section_style))
            present = [(k.replace("_", " ").title(), v.get("probability", 0))
                       for k, v in lesions.items()
                       if v.get("present", False) or
                          (hasattr(v.get("present", False), "item") and v["present"].item())]
            if present:
                for lesion_name, prob in present:
                    if hasattr(prob, "item"):
                        prob = prob.item()
                    story.append(Paragraph(
                        f"• <b>{lesion_name}</b> detected (confidence: {float(prob):.0%})",
                        finding_style,
                    ))
            else:
                story.append(Paragraph(
                    "• No significant lesions detected.",
                    ParagraphStyle("", parent=finding_style, textColor=c(ReportColors.GREEN))
                ))
            story.append(Spacer(1, 0.3*cm))

        # ── Progression Section (if available) ────────────────────────────
        if progression_report is not None:
            story.append(Paragraph("LONGITUDINAL PROGRESSION", section_style))
            prog_text = (
                f"Visits: {progression_report.num_visits} | "
                f"Period: {progression_report.date_range_days} days | "
                f"Trend: <b>{progression_report.overall_trend.upper()}</b> | "
                f"12-month risk: <b>{progression_report.risk_12m:.0%}</b> "
                f"[{progression_report.risk_level.upper()}]"
            )
            story.append(Paragraph(prog_text, finding_style))
            story.append(Spacer(1, 0.2*cm))

        # ── Clinical Recommendations ───────────────────────────────────────
        story.append(Paragraph("CLINICAL RECOMMENDATIONS", section_style))
        report_data = analysis_result.get("report", {})
        rec_text = report_data.get("recommendation",
                   "Routine screening in 12 months.")
        story.append(Paragraph(rec_text, finding_style))
        story.append(Spacer(1, 0.5*cm))

        # ── Footer / Disclaimer ────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5, color=c(ReportColors.GRAY)))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            f"<b>IMPORTANT DISCLAIMER:</b> {self.footer_text}",
            disclaimer_style,
        ))
        story.append(Paragraph(
            f"Generated by Retina-GPT Foundation Model | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            disclaimer_style,
        ))

        # Build PDF
        doc.build(story)
        return output_path

    # ── Text Fallback ──────────────────────────────────────────────────────

    def _generate_text(
        self, output_path, patient_info, analysis_result, progression_report
    ) -> str:
        """Fallback text report when ReportLab is unavailable."""
        lines = [
            "=" * 65,
            f"  {self.clinic_name.upper()}",
            "  AI-ASSISTED RETINAL ANALYSIS REPORT",
            "=" * 65,
            f"  Patient ID:  {patient_info.get('id', '—')}",
            f"  Patient:     {patient_info.get('name', '—')}",
            f"  Age / Sex:   {patient_info.get('age', '—')} / {patient_info.get('sex', '—')}",
            f"  Date:        {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 65,
            "",
        ]

        # DR result
        dr = analysis_result.get("dr", {})
        dr_grade = dr.get("grade", None)
        if dr_grade is not None:
            if hasattr(dr_grade, "item"):
                dr_grade = dr_grade.item()
            dr_label = self.DR_GRADE_LABELS.get(int(dr_grade), str(dr_grade))
            dr_conf  = dr.get("confidence", 0)
            if hasattr(dr_conf, "item"):
                dr_conf = dr_conf.item()
            lines.append(f"  DIABETIC RETINOPATHY: {dr_label} ({float(dr_conf):.0%})")

        # Lesions
        lesions = analysis_result.get("lesions", {})
        present_lesions = [k for k, v in lesions.items()
                           if v.get("present", False)]
        if present_lesions:
            lines.append(f"  LESIONS DETECTED: {', '.join(present_lesions)}")
        else:
            lines.append("  LESIONS: None detected")

        # Report
        report = analysis_result.get("report", {})
        lines.append("")
        lines.append(report.get("structured_findings", ""))
        lines.append("")
        lines.append(report.get("recommendation", ""))

        if progression_report:
            lines.append("")
            lines.append(progression_report.full_report)

        lines += [
            "",
            "-" * 65,
            f"  {self.footer_text}",
            f"  Generated by Retina-GPT | {datetime.now().isoformat()}",
            "-" * 65,
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    # ── Image Utilities ────────────────────────────────────────────────────

    def _array_to_image_buffer(
        self,
        image: np.ndarray,
        size: Tuple[int, int] = (300, 300),
    ) -> Optional[io.BytesIO]:
        """Convert numpy image array to BytesIO buffer for ReportLab."""
        try:
            from PIL import Image as PILImage
            import cv2

            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)

            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)

            resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
            pil_img = PILImage.fromarray(resized)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            return buf
        except Exception:
            return None
