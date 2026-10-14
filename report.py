
import os
from datetime import datetime

from flask import (
    Blueprint,
    render_template,
    session,
    send_file,
    current_app
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak
)


report = Blueprint(
    "report",
    __name__
)


# ============================================================
# PATH HELPER
# ============================================================

def clean_path(path):

    if not path:
        return None

    return str(path).replace("\\", "/").lstrip("/")


def absolute_file(path):

    path = clean_path(path)

    if not path:
        return None

    if os.path.isabs(path):
        return path

    return os.path.join(
        current_app.root_path,
        path
    )


# ============================================================
# VALUE HELPERS
# ============================================================

def safe_value(value, default="Not Available"):

    if value is None:
        return default

    if isinstance(value, str) and not value.strip():
        return default

    return value


def percentage(value):

    try:
        return f"{float(value):.2f}%"
    except:
        return "0.00%"


# ============================================================
# PROBABILITY NORMALIZER
# ============================================================

def normalize_probabilities(probabilities):

    result = []

    if not probabilities:
        return result

    # Dictionary
    if isinstance(probabilities, dict):

        for label, value in probabilities.items():

            try:
                value = float(value)
            except:
                value = 0

            result.append(
                (
                    str(label),
                    value
                )
            )

    # List
    elif isinstance(probabilities, list):

        for item in probabilities:

            if isinstance(item, dict):

                label = item.get(
                    "label",
                    item.get(
                        "class",
                        "Unknown"
                    )
                )

                value = item.get(
                    "value",
                    item.get(
                        "probability",
                        0
                    )
                )

                try:
                    value = float(value)
                except:
                    value = 0

                result.append(
                    (
                        str(label),
                        value
                    )
                )

    return result


# ============================================================
# PDF STYLES
# ============================================================

styles = getSampleStyleSheet()


styles.add(
    ParagraphStyle(
        name="SGTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
)


styles.add(
    ParagraphStyle(
        name="SGSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10
    )
)


styles.add(
    ParagraphStyle(
        name="SGSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=3,
        spaceAfter=9
    )
)


styles.add(
    ParagraphStyle(
        name="SGBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
)


styles.add(
    ParagraphStyle(
        name="SGCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b")
    )
)


styles.add(
    ParagraphStyle(
        name="SGHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )
)


styles.add(
    ParagraphStyle(
        name="SGTable",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1f2937")
    )
)


styles.add(
    ParagraphStyle(
        name="SGDisclaimerTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#9f1239")
    )
)


styles.add(
    ParagraphStyle(
        name="SGDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#881337")
    )
)


# ============================================================
# TABLE CREATOR
# ============================================================

def create_table(data, widths):

    converted = []

    for row_index, row in enumerate(data):

        converted_row = []

        for value in row:

            if row_index == 0:

                converted_row.append(
                    Paragraph(
                        str(value),
                        styles["SGHeader"]
                    )
                )

            else:

                converted_row.append(
                    Paragraph(
                        str(value),
                        styles["SGTable"]
                    )
                )

        converted.append(converted_row)

    table = Table(
        converted,
        colWidths=widths,
        repeatRows=1
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e3a5f")
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f8fafc")
                ]
            )
        ])
    )

    return table


# ============================================================
# IMAGE CREATOR
# ============================================================

def pdf_image(
    path,
    max_width=130 * mm,
    max_height=80 * mm
):

    full_path = absolute_file(path)

    if not full_path:
        return None

    if not os.path.exists(full_path):
        print(
            "PDF image not found:",
            full_path
        )
        return None

    try:

        image = Image(full_path)

        image._restrictSize(
            max_width,
            max_height
        )

        return image

    except Exception as e:

        print(
            "PDF image error:",
            e
        )

        return None


# ============================================================
# PAGE HEADER / FOOTER
# ============================================================

def header_footer(canvas, doc):

    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(
        colors.HexColor("#cbd5e1")
    )

    canvas.line(
        18 * mm,
        height - 14 * mm,
        width - 18 * mm,
        height - 14 * mm
    )

    canvas.setFont(
        "Helvetica-Bold",
        8
    )

    canvas.setFillColor(
        colors.HexColor("#475569")
    )

    canvas.drawString(
        18 * mm,
        height - 11 * mm,
        "SkinGuard AI"
    )

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.drawRightString(
        width - 18 * mm,
        height - 11 * mm,
        "AI-Based Dermoscopic Skin Analyzer"
    )

    canvas.line(
        18 * mm,
        12 * mm,
        width - 18 * mm,
        12 * mm
    )

    canvas.setFont(
        "Helvetica",
        7.5
    )

    canvas.setFillColor(
        colors.HexColor("#64748b")
    )

    canvas.drawString(
        18 * mm,
        7 * mm,
        "SkinGuard AI | Explainable AI Skin Lesion Analysis"
    )

    canvas.drawRightString(
        width - 18 * mm,
        7 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ============================================================
# REPORT PAGE
# ============================================================

@report.route("/report")
def report_page():

    return render_template(

        "report.html",

        scan_id=session.get(
            "scan_id",
            "SG-" + datetime.now().strftime(
                "%Y%m%d%H%M%S"
            )
        ),

        report_date=datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        ),

        uploaded_image=session.get(
            "uploaded_image"
        ),

        quality_score=session.get(
            "quality_score",
            0
        ),

        blur_status=session.get(
            "blur_status",
            "Not Available"
        ),

        brightness_status=session.get(
            "brightness_status",
            "Not Available"
        ),

        noise_status=session.get(
            "noise_status",
            "Not Available"
        ),

        quality_status=session.get(
            "quality_status",
            "Acceptable"
        ),

        preprocessed_image=session.get(
            "preprocessed_image"
        ),

        segmentation_image=session.get(
            "segmentation_image"
        ),

        segmentation_status=session.get(
            "segmentation_status",
            "Completed"
        ),

        lesion_area=session.get(
            "lesion_area",
            "Detected"
        ),

        disease=session.get(
            "disease",
            "Unknown"
        ),

        confidence=session.get(
            "confidence",
            0
        ),

        probabilities=session.get(
            "probabilities",
            {}
        ),

        gradcam_image=session.get(
            "gradcam_image"
        ),

        important_region=session.get(
            "important_region"
        ),

        severity=session.get(
            "severity",
            "Not Available"
        ),

        severity_score=session.get(
            "severity_score",
            0
        ),

        recommendation=session.get(
            "recommendation",
            "Please consult a qualified dermatologist."
        )
    )


# ============================================================
# PDF DOWNLOAD
# ============================================================

@report.route("/report/download")
def generate_report():

    # ========================================================
    # REPORT FOLDER
    # ========================================================

    reports_folder = os.path.join(
        current_app.root_path,
        "static",
        "reports"
    )

    os.makedirs(
        reports_folder,
        exist_ok=True
    )

    # ========================================================
    # REPORT DATA
    # ========================================================

    scan_id = session.get(
        "scan_id",
        "SG-" + datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )
    )

    report_date = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )

    uploaded_image = session.get(
        "uploaded_image"
    )

    preprocessed_image = session.get(
        "preprocessed_image"
    )

    segmentation_image = session.get(
        "segmentation_image"
    )

    gradcam_image = session.get(
        "gradcam_image"
    )

    important_region = session.get(
        "important_region"
    )

    quality_score = session.get(
        "quality_score",
        0
    )

    blur_status = session.get(
        "blur_status",
        "Not Available"
    )

    brightness_status = session.get(
        "brightness_status",
        "Not Available"
    )

    noise_status = session.get(
        "noise_status",
        "Not Available"
    )

    quality_status = session.get(
        "quality_status",
        "Acceptable"
    )

    disease = safe_value(
        session.get("disease"),
        "Unknown"
    )

    confidence = session.get(
        "confidence",
        0
    )

    probabilities = normalize_probabilities(
        session.get(
            "probabilities",
            {}
        )
    )

    severity = safe_value(
        session.get("severity"),
        "Not Available"
    )

    severity_score = session.get(
        "severity_score",
        0
    )

    recommendation = safe_value(
        session.get("recommendation"),
        "Please consult a qualified dermatologist for final evaluation."
    )

    segmentation_status = session.get(
        "segmentation_status",
        "Completed"
    )

    lesion_area = session.get(
        "lesion_area",
        "Detected"
    )

    # ========================================================
    # PDF PATH
    # ========================================================

    pdf_path = os.path.join(
        reports_folder,
        f"SkinGuard_AI_{scan_id}.pdf"
    )

    # ========================================================
    # DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(

        pdf_path,

        pagesize=A4,

        rightMargin=18 * mm,
        leftMargin=18 * mm,

        topMargin=20 * mm,
        bottomMargin=18 * mm,

        title="SkinGuard AI Analysis Report",
        author="SkinGuard AI"
    )

    story = []

    # ========================================================
    # HEADER
    # ========================================================

    story.append(
        Spacer(1, 4 * mm)
    )

    story.append(
        Paragraph(
            "SKINGUARD AI",
            styles["SGTitle"]
        )
    )

    story.append(
        Paragraph(
            "Explainable AI Based Dermoscopic Skin Analyzer Report",
            styles["SGSubtitle"]
        )
    )

    story.append(
        Paragraph(
            "AI-assisted skin lesion screening and analysis report",
            styles["SGCenter"]
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # ========================================================
    # 1. SCAN INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Scan Information",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "Parameter",
                    "Result"
                ],

                [
                    "Scan ID",
                    scan_id
                ],

                [
                    "Date & Time",
                    report_date
                ],

                [
                    "Predicted Disease",
                    disease
                ],

                [
                    "Confidence",
                    percentage(confidence)
                ],

                [
                    "Severity",
                    severity
                ],

                [
                    "Severity Score",
                    percentage(severity_score)
                ]
            ],

            [
                65 * mm,
                107 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 6 * mm)
    )

    # ========================================================
    # 2. UPLOADED IMAGE
    # ========================================================

    story.append(
        Paragraph(
            "2. Uploaded Dermoscopic Image",
            styles["SGSection"]
        )
    )

    img = pdf_image(
        uploaded_image,
        130 * mm,
        75 * mm
    )

    if img:

        image_table = Table(
            [[img]],
            colWidths=[170 * mm]
        )

        image_table.setStyle(
            TableStyle([
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#cbd5e1")
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#f8fafc")
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        story.append(image_table)

    else:

        story.append(
            Paragraph(
                "Uploaded image is not available.",
                styles["SGCenter"]
            )
        )

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        Paragraph(
            "Original dermoscopic image uploaded for AI-based skin analysis.",
            styles["SGCenter"]
        )
    )

    story.append(PageBreak())

    # ========================================================
    # 3. IMAGE QUALITY CHECK
    # ========================================================

    story.append(
        Paragraph(
            "3. Image Quality Check",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "Parameter",
                    "Result"
                ],

                [
                    "Quality Score",
                    percentage(quality_score)
                ],

                [
                    "Blur",
                    blur_status
                ],

                [
                    "Brightness",
                    brightness_status
                ],

                [
                    "Noise",
                    noise_status
                ],

                [
                    "Overall Status",
                    quality_status
                ]
            ],

            [
                65 * mm,
                107 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    # ========================================================
    # 4. PREPROCESSING
    # ========================================================

    story.append(
        Paragraph(
            "4. Image Pre-processing",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "Parameter",
                    "Result"
                ],

                [
                    "Status",
                    "Completed"
                ],

                [
                    "Resize",
                    "224 × 224 pixels"
                ],

                [
                    "Enhancement",
                    "Applied"
                ],

                [
                    "Normalization",
                    "Applied"
                ],

                [
                    "Pipeline",
                    "Original → Resized → Enhanced → Normalized"
                ]
            ],

            [
                65 * mm,
                107 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    img = pdf_image(
        preprocessed_image,
        105 * mm,
        62 * mm
    )

    if img:

        story.append(
            Table(
                [[img]],
                colWidths=[170 * mm],
                style=TableStyle([
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#cbd5e1")
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )
                ])
            )
        )

    story.append(PageBreak())

    # ========================================================
    # 5. SEGMENTATION
    # ========================================================

    story.append(
        Paragraph(
            "5. Lesion Segmentation",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "Parameter",
                    "Result"
                ],

                [
                    "Segmentation Model",
                    "U-Net"
                ],

                [
                    "Status",
                    segmentation_status
                ],

                [
                    "Lesion Area",
                    lesion_area
                ]
            ],

            [
                65 * mm,
                107 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 6 * mm)
    )

    img = pdf_image(
        segmentation_image,
        125 * mm,
        75 * mm
    )

    if img:

        story.append(
            Table(
                [[img]],
                colWidths=[170 * mm],
                style=TableStyle([
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#cbd5e1")
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )
                ])
            )
        )

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        Paragraph(
            "U-Net based lesion segmentation output.",
            styles["SGCenter"]
        )
    )

    story.append(PageBreak())

    # ========================================================
    # 6. CLASSIFICATION
    # ========================================================

    story.append(
        Paragraph(
            "6. Disease Classification",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "Parameter",
                    "Result"
                ],

                [
                    "Predicted Class",
                    disease
                ],

                [
                    "Confidence Score",
                    percentage(confidence)
                ],

                [
                    "Classification Model",
                    "EfficientNet-B3"
                ],

                [
                    "Number of Classes",
                    "7"
                ]
            ],

            [
                65 * mm,
                107 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    # ========================================================
    # 7. PROBABILITY DISTRIBUTION
    # ========================================================

    story.append(
        Paragraph(
            "7. Seven-Class Probability Distribution",
            styles["SGSection"]
        )
    )

    probability_data = [
        [
            "Disease Class",
            "Probability"
        ]
    ]

    for label, value in probabilities:

        probability_data.append(
            [
                label,
                percentage(value)
            ]
        )

    if len(probability_data) == 1:

        probability_data.append(
            [
                "Probability data",
                "Not Available"
            ]
        )

    story.append(
        create_table(
            probability_data,
            [
                110 * mm,
                62 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    story.append(
        Paragraph(
            "Probability distribution generated by the EfficientNet-B3 classification model.",
            styles["SGBody"]
        )
    )

    story.append(PageBreak())

    # ========================================================
    # 8. GRAD CAM
    # ========================================================

    story.append(
        Paragraph(
            "8. Explainable AI - Grad-CAM",
            styles["SGSection"]
        )
    )

    explanation = Table(

        [[
            Paragraph(
                "<b>Grad-CAM:</b> Grad-CAM highlights the image regions that contribute to the model prediction.",
                styles["SGBody"]
            )
        ]],

        colWidths=[170 * mm]
    )

    explanation.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#eff6ff")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#bfdbfe")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(explanation)

    story.append(
        Spacer(1, 7 * mm)
    )

    original = pdf_image(
        uploaded_image,
        48 * mm,
        48 * mm
    )

    heatmap = pdf_image(
        gradcam_image,
        48 * mm,
        48 * mm
    )

    important = pdf_image(
        important_region,
        48 * mm,
        48 * mm
    )

    gradcam_table = Table(

        [
            [
                original or Paragraph(
                    "Unavailable",
                    styles["SGCenter"]
                ),

                heatmap or Paragraph(
                    "Unavailable",
                    styles["SGCenter"]
                ),

                important or Paragraph(
                    "Unavailable",
                    styles["SGCenter"]
                )
            ],

            [
                Paragraph(
                    "<b>Original Image</b>",
                    styles["SGCenter"]
                ),

                Paragraph(
                    "<b>Grad-CAM Heatmap</b>",
                    styles["SGCenter"]
                ),

                Paragraph(
                    "<b>Important Region</b>",
                    styles["SGCenter"]
                )
            ]
        ],

        colWidths=[
            56 * mm,
            56 * mm,
            56 * mm
        ]
    )

    gradcam_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#f8fafc")
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        gradcam_table
    )

    story.append(PageBreak())

    # ========================================================
    # 9. SEVERITY
    # ========================================================

    story.append(
        Paragraph(
            "9. Severity Prediction",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "Parameter",
                    "Result"
                ],

                [
                    "Severity Level",
                    severity
                ],

                [
                    "Severity Score",
                    percentage(severity_score)
                ],

                [
                    "Algorithm",
                    "Rule-Based Severity Prediction"
                ]
            ],

            [
                65 * mm,
                107 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    # Recommendation

    story.append(
        Paragraph(
            "Personalized Clinical Recommendation",
            styles["SGSection"]
        )
    )

    recommendation_box = Table(

        [[
            Paragraph(
                str(recommendation),
                styles["SGBody"]
            )
        ]],

        colWidths=[170 * mm]
    )

    recommendation_box.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#eff6ff")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#2563eb")
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                10
            )
        ])
    )

    story.append(
        recommendation_box
    )

    story.append(PageBreak())

    # ========================================================
    # 10. AI SYSTEM SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "10. AI System Summary",
            styles["SGSection"]
        )
    )

    story.append(
        create_table(

            [
                [
                    "System Component",
                    "Technology / Method"
                ],

                [
                    "Image Pre-processing",
                    "OpenCV"
                ],

                [
                    "Lesion Segmentation",
                    "U-Net"
                ],

                [
                    "Disease Classification",
                    "EfficientNet-B3"
                ],

                [
                    "Explainable AI",
                    "Grad-CAM"
                ],

                [
                    "Severity Prediction",
                    "Rule-Based Algorithm"
                ],

                [
                    "Application",
                    "Flask"
                ]
            ],

            [
                70 * mm,
                102 * mm
            ]
        )
    )

    story.append(
        Spacer(1, 10 * mm)
    )

    # ========================================================
    # MEDICAL DISCLAIMER
    # ========================================================

    disclaimer = Table(

        [

            [
                Paragraph(
                    "MEDICAL DISCLAIMER",
                    styles["SGDisclaimerTitle"]
                )
            ],

            [
                Paragraph(
                    "This AI-generated report is intended for screening and clinical decision support purposes only. "
                    "It should not be considered as a substitute for professional medical diagnosis. "
                    "Please consult a qualified dermatologist for final evaluation.",
                    styles["SGDisclaimer"]
                )
            ]

        ],

        colWidths=[170 * mm]
    )

    disclaimer.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#ffeef1")
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor("#f7b7c2")
            ),

            (
                "LINEBELOW",
                (0, 0),
                (-1, 0),
                0.4,
                colors.HexColor("#f7b7c2")
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        disclaimer
    )

    story.append(
        Spacer(1, 7 * mm)
    )

    story.append(
        Paragraph(
            f"Report generated by SkinGuard AI on {report_date}",
            styles["SGCenter"]
        )
    )

    # ========================================================
    # BUILD
    # ========================================================

    document.build(

        story,

        onFirstPage=header_footer,

        onLaterPages=header_footer
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    return send_file(

        pdf_path,

        as_attachment=True,

        download_name=f"SkinGuard_AI_{scan_id}.pdf",

        mimetype="application/pdf"
    )
