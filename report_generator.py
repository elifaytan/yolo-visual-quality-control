from __future__ import annotations

from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import font_manager
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PAGE_WIDTH, PAGE_HEIGHT = A4

FONT_REGULAR = "QualityRegular"
FONT_BOLD = "QualityBold"

NAVY = colors.HexColor("#17365D")
DARK_NAVY = colors.HexColor("#102A43")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
BORDER_GRAY = colors.HexColor("#CBD5E1")
TEXT_GRAY = colors.HexColor("#475569")

PASS_GREEN = colors.HexColor("#15803D")
PASS_BACKGROUND = colors.HexColor("#DCFCE7")

FAIL_RED = colors.HexColor("#B91C1C")
FAIL_BACKGROUND = colors.HexColor("#FEE2E2")

WARNING_BACKGROUND = colors.HexColor("#FFF7ED")
WARNING_BORDER = colors.HexColor("#FDBA74")


def register_pdf_fonts() -> None:
    """Türkçe karakter destekli yazı tiplerini kaydeder."""

    regular_path = font_manager.findfont(
        font_manager.FontProperties(
            family="DejaVu Sans",
        )
    )

    bold_path = font_manager.findfont(
        font_manager.FontProperties(
            family="DejaVu Sans",
            weight="bold",
        )
    )

    registered_fonts = pdfmetrics.getRegisteredFontNames()

    if FONT_REGULAR not in registered_fonts:
        pdfmetrics.registerFont(
            TTFont(
                FONT_REGULAR,
                regular_path,
            )
        )

    if FONT_BOLD not in registered_fonts:
        pdfmetrics.registerFont(
            TTFont(
                FONT_BOLD,
                bold_path,
            )
        )


def create_styles():
    """PDF içerisinde kullanılacak metin stillerini oluşturur."""

    register_pdf_fonts()

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="InstitutionName",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=14,
            leading=18,
            alignment=TA_LEFT,
            textColor=colors.white,
        )
    )

    styles.add(
        ParagraphStyle(
            name="InstitutionUnit",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=8.5,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.white,
        )
    )

    styles.add(
        ParagraphStyle(
            name="DocumentCode",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.white,
        )
    )

    styles.add(
        ParagraphStyle(
            name="QualityTitle",
            parent=styles["Title"],
            fontName=FONT_BOLD,
            fontSize=16,
            leading=21,
            alignment=TA_CENTER,
            textColor=DARK_NAVY,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="QualitySubtitle",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=8.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=TEXT_GRAY,
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=14,
            textColor=DARK_NAVY,
            spaceBefore=8,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1F2937"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyBold",
            parent=styles["BodyText"],
            fontName=FONT_BOLD,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1F2937"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7,
            leading=9.5,
            textColor=TEXT_GRAY,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResultPass",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            alignment=TA_CENTER,
            textColor=PASS_GREEN,
            leading=19,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResultFail",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            alignment=TA_CENTER,
            textColor=FAIL_RED,
            leading=19,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SignatureTitle",
            parent=styles["Normal"],
            fontName=FONT_BOLD,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=DARK_NAVY,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SignatureText",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10,
            alignment=TA_CENTER,
            textColor=TEXT_GRAY,
        )
    )

    return styles


def safe_paragraph(
    text: object,
    style,
) -> Paragraph:
    """Metni ReportLab Paragraph için güvenli hâle getirir."""

    safe_text = str(text)
    safe_text = safe_text.replace("&", "&amp;")
    safe_text = safe_text.replace("<", "&lt;")
    safe_text = safe_text.replace(">", "&gt;")

    return Paragraph(
        safe_text,
        style,
    )


def numpy_image_to_buffer(
    image_array: np.ndarray,
) -> BytesIO:
    """NumPy görüntüsünü JPEG veri akışına dönüştürür."""

    image = PILImage.fromarray(
        image_array.astype("uint8")
    ).convert("RGB")

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=92,
    )

    buffer.seek(0)

    return buffer


def pil_image_to_buffer(
    image: PILImage.Image,
) -> BytesIO:
    """Pillow görüntüsünü JPEG veri akışına dönüştürür."""

    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=92,
    )

    buffer.seek(0)

    return buffer


def calculate_pdf_image_size(
    image_width: int,
    image_height: int,
    max_width: float,
    max_height: float,
) -> tuple[float, float]:
    """Görüntünün oranını koruyarak PDF boyutunu hesaplar."""

    width_ratio = max_width / image_width
    height_ratio = max_height / image_height

    scale = min(
        width_ratio,
        height_ratio,
    )

    return (
        image_width * scale,
        image_height * scale,
    )


def get_model_display_name(
    model_path: str,
) -> str:
    """Model dosya yolunu kullanıcı dostu model adına dönüştürür."""

    model_filename = Path(model_path).name.lower()

    if model_filename == "best.pt":
        return "YOLO11 Özel Eğitimli Görsel Kalite Kontrol Modeli"

    if "yolo11" in model_filename:
        return "Ultralytics YOLO11 Nesne Algılama Modeli"

    return f"Özel Eğitimli YOLO Modeli ({Path(model_path).name})"


def get_component_status(
    expected_count: int,
    detected_count: int,
) -> tuple[str, int]:
    """Beklenen ve bulunan adetlere göre açıklayıcı durum oluşturur."""

    difference = detected_count - expected_count

    if difference == 0:
        return "Uygun", difference

    if difference > 0:
        return (
            f"{difference} adet fazla tespit edildi",
            difference,
        )

    return (
        f"{abs(difference)} adet eksik tespit edildi",
        difference,
    )


def calculate_summary(
    expected_objects: dict[str, int],
    detected_counts: Counter | dict[str, int],
) -> dict[str, float | int]:
    """Genel bileşen istatistiklerini hesaplar."""

    total_expected = sum(
        expected_objects.values()
    )

    total_detected = sum(
        int(
            detected_counts.get(
                class_name,
                0,
            )
        )
        for class_name in expected_objects
    )

    suitable_components = sum(
        1
        for class_name, expected_count
        in expected_objects.items()
        if int(
            detected_counts.get(
                class_name,
                0,
            )
        ) == expected_count
    )

    total_component_types = len(
        expected_objects
    )

    suitability_rate = (
        round(
            suitable_components
            / total_component_types
            * 100,
            1,
        )
        if total_component_types
        else 0.0
    )

    return {
        "total_expected": total_expected,
        "total_detected": total_detected,
        "suitable_components": suitable_components,
        "total_component_types": total_component_types,
        "suitability_rate": suitability_rate,
    }


def add_page_header_and_footer(
    canvas,
    document,
) -> None:
    """Sayfalara belge üst bilgisi, alt bilgisi ve numara ekler."""

    canvas.saveState()

    canvas.setStrokeColor(BORDER_GRAY)
    canvas.setLineWidth(0.5)

    canvas.line(
        1.5 * cm,
        PAGE_HEIGHT - 0.85 * cm,
        PAGE_WIDTH - 1.5 * cm,
        PAGE_HEIGHT - 0.85 * cm,
    )

    canvas.setFont(
        FONT_REGULAR,
        6.8,
    )

    canvas.setFillColor(TEXT_GRAY)

    canvas.drawString(
        1.5 * cm,
        0.72 * cm,
        "Yapay Zekâ Destekli Görsel Kalite Kontrol Platformu",
    )

    canvas.drawCentredString(
        PAGE_WIDTH / 2,
        0.72 * cm,
        "Elektronik ortamda otomatik oluşturulmuştur.",
    )

    canvas.drawRightString(
        PAGE_WIDTH - 1.5 * cm,
        0.72 * cm,
        f"Sayfa {document.page}",
    )

    canvas.restoreState()


def create_institution_header(
    styles,
    report_number: str,
    report_date: str,
) -> Table:
    """Kurumsal belge üst başlığını oluşturur."""

    left_content = [
        safe_paragraph(
            "TSE KALİTE KAMPÜSÜ",
            styles["InstitutionName"],
        ),
        safe_paragraph(
            "Yapay Zekâ Destekli Görsel Kalite Kontrol Sistemi",
            styles["InstitutionUnit"],
        ),
    ]

    right_content = [
        safe_paragraph(
            "Belge Kodu: YZ-GKK-01",
            styles["DocumentCode"],
        ),
        safe_paragraph(
            "Revizyon No: 00",
            styles["DocumentCode"],
        ),
        safe_paragraph(
            f"Rapor No: {report_number}",
            styles["DocumentCode"],
        ),
        safe_paragraph(
            f"Tarih: {report_date}",
            styles["DocumentCode"],
        ),
    ]

    header_table = Table(
        [
            [
                left_content,
                right_content,
            ]
        ],
        colWidths=[
            11.5 * cm,
            6.0 * cm,
        ],
    )

    header_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    NAVY,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    DARK_NAVY,
                ),
                (
                    "LINEBEFORE",
                    (1, 0),
                    (1, 0),
                    0.6,
                    colors.white,
                ),
            ]
        )
    )

    return header_table


def create_signature_section(
    styles,
) -> Table:
    """Rapor sonunda kontrol ve onay alanlarını oluşturur."""

    signature_table = Table(
        [
            [
                safe_paragraph(
                    "KONTROLÜ GERÇEKLEŞTİREN",
                    styles["SignatureTitle"],
                ),
                safe_paragraph(
                    "KONTROL EDEN / ONAYLAYAN",
                    styles["SignatureTitle"],
                ),
            ],
            [
                Paragraph(
                    (
                        "Yapay Zekâ Destekli<br/>"
                        "Görsel Kalite Kontrol Platformu"
                    ),
                    styles["SignatureText"],
                ),
                Paragraph(
                    (
                        "<br/><br/>"
                        "Ad Soyad / İmza"
                    ),
                    styles["SignatureText"],
                ),
            ],
        ],
        colWidths=[
            8.75 * cm,
            8.75 * cm,
        ],
        rowHeights=[
            0.75 * cm,
            2.3 * cm,
        ],
    )

    signature_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    LIGHT_BLUE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    BORDER_GRAY,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    return signature_table


def create_quality_report_pdf(
    *,
    profile_name: str,
    standard_name: str,
    description: str,
    is_pass: bool,
    detected_counts: Counter | dict[str, int],
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    display_names: dict[str, str],
    errors: Iterable[str],
    confidence: float,
    original_image: PILImage.Image,
    annotated_image: np.ndarray,
    model_path: str,
) -> tuple[bytes, str]:
    """Kalite kontrol sonucuna ait kurumsal PDF raporu oluşturur."""

    styles = create_styles()

    now = datetime.now()

    report_number = now.strftime(
        "GKK-%Y%m%d-%H%M%S"
    )

    report_date = now.strftime(
        "%d.%m.%Y %H:%M:%S"
    )

    result_text = (
        "UYGUN — PASS"
        if is_pass
        else "UYGUN DEĞİL — FAIL"
    )

    result_style = (
        styles["ResultPass"]
        if is_pass
        else styles["ResultFail"]
    )

    result_background = (
        PASS_BACKGROUND
        if is_pass
        else FAIL_BACKGROUND
    )

    result_border = (
        PASS_GREEN
        if is_pass
        else FAIL_RED
    )

    error_list = list(
        errors
    )

    is_defect_profile = not expected_objects

    summary = calculate_summary(
        expected_objects=expected_objects,
        detected_counts=detected_counts,
    )

    report_buffer = BytesIO()

    document = SimpleDocTemplate(
        report_buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.15 * cm,
        bottomMargin=1.35 * cm,
        title="Görsel Kalite Kontrol ve Uygunluk Analiz Raporu",
        author="Yapay Zekâ Destekli Görsel Kalite Kontrol Platformu",
    )

    story = []

    story.append(
        create_institution_header(
            styles=styles,
            report_number=report_number,
            report_date=report_date,
        )
    )

    story.append(
        Spacer(
            1,
            0.10 * cm,
        )
    )

    story.append(
        Paragraph(
            "GÖRSEL KALİTE KONTROL VE UYGUNLUK ANALİZ RAPORU",
            styles["QualityTitle"],
        )
    )

    story.append(
        Paragraph(
            (
                "Bu rapor; YOLO11 tabanlı nesne algılama modeli, "
                "tanımlı ürün profili ve yazılım tabanlı uygunluk "
                "kuralları kullanılarak otomatik olarak hazırlanmıştır."
            ),
            styles["QualitySubtitle"],
        )
    )

    result_table = Table(
        [
            [
                Paragraph(
                    result_text,
                    result_style,
                )
            ]
        ],
        colWidths=[
            17.5 * cm
        ],
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    result_background,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    1.4,
                    result_border,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(
        result_table
    )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )

    story.append(
        Paragraph(
            "1. Kontrol ve Belge Bilgileri",
            styles["SectionHeading"],
        )
    )

    information_rows = [
        [
            safe_paragraph(
                "Rapor Numarası",
                styles["BodyBold"],
            ),
            safe_paragraph(
                report_number,
                styles["Body"],
            ),
            safe_paragraph(
                "Belge Kodu",
                styles["BodyBold"],
            ),
            safe_paragraph(
                "YZ-GKK-01",
                styles["Body"],
            ),
        ],
        [
            safe_paragraph(
                "Kontrol Tarihi",
                styles["BodyBold"],
            ),
            safe_paragraph(
                report_date,
                styles["Body"],
            ),
            safe_paragraph(
                "Revizyon",
                styles["BodyBold"],
            ),
            safe_paragraph(
                "00",
                styles["Body"],
            ),
        ],
        [
            safe_paragraph(
                "Ürün Profili",
                styles["BodyBold"],
            ),
            safe_paragraph(
                profile_name,
                styles["Body"],
            ),
            safe_paragraph(
                "Güven Eşiği",
                styles["BodyBold"],
            ),
            safe_paragraph(
                f"{confidence:.2f}",
                styles["Body"],
            ),
        ],
        [
            safe_paragraph(
                "Uygulanan Kriter",
                styles["BodyBold"],
            ),
            safe_paragraph(
                standard_name,
                styles["Body"],
            ),
            safe_paragraph(
                "Kullanılan Model",
                styles["BodyBold"],
            ),
            safe_paragraph(
                get_model_display_name(
                    model_path
                ),
                styles["Body"],
            ),
        ],
    ]

    information_table = Table(
        information_rows,
        colWidths=[
            3.0 * cm,
            5.75 * cm,
            3.0 * cm,
            5.75 * cm,
        ],
    )

    information_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    LIGHT_GRAY,
                ),
                (
                    "BACKGROUND",
                    (2, 0),
                    (2, -1),
                    LIGHT_GRAY,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER_GRAY,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        information_table
    )

    story.append(
        Spacer(
            1,
            0.22 * cm,
        )
    )

    story.append(
        safe_paragraph(
            description,
            styles["Body"],
        )
    )

    story.append(
        Paragraph(
            "2. Yönetici Özeti",
            styles["SectionHeading"],
        )
    )


    if is_defect_profile:
        detected_defect_count = sum(
            int(detected_counts.get(class_name, 0))
            for class_name in forbidden_objects
        )

        detected_defect_types = sum(
            1
            for class_name in forbidden_objects
            if int(detected_counts.get(class_name, 0)) > 0
        )

        summary_rows = [
            [
                safe_paragraph(
                    "Kontrol Edilen Kusur Türü",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    len(forbidden_objects),
                    styles["Body"],
                ),
                safe_paragraph(
                    "Tespit Edilen Kusur Sayısı",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    detected_defect_count,
                    styles["Body"],
                ),
            ],
            [
                safe_paragraph(
                    "Tespit Edilen Kusur Türü",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    detected_defect_types,
                    styles["Body"],
                ),
                safe_paragraph(
                    "Model Güven Eşiği",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    f"%{confidence * 100:.0f}",
                    styles["Body"],
                ),
            ],
            [
                safe_paragraph(
                    "Uygunsuzluk Sayısı",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    len(error_list),
                    styles["Body"],
                ),
                safe_paragraph(
                    "Genel Karar",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    result_text,
                    styles["BodyBold"],
                ),
            ],
        ]
    else:
        summary_rows = [
            [
                safe_paragraph(
                    "Beklenen Toplam Bileşen",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    summary["total_expected"],
                    styles["Body"],
                ),
                safe_paragraph(
                    "Bulunan Toplam Bileşen",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    summary["total_detected"],
                    styles["Body"],
                ),
            ],
            [
                safe_paragraph(
                    "Uygun Bileşen Türü",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    (
                        f"{summary['suitable_components']} / "
                        f"{summary['total_component_types']}"
                    ),
                    styles["Body"],
                ),
                safe_paragraph(
                    "Bileşen Uygunluk Oranı",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    f"%{summary['suitability_rate']}",
                    styles["Body"],
                ),
            ],
            [
                safe_paragraph(
                    "Uygunsuzluk Sayısı",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    len(error_list),
                    styles["Body"],
                ),
                safe_paragraph(
                    "Genel Karar",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    result_text,
                    styles["BodyBold"],
                ),
            ],
        ]

    summary_table = Table(
        summary_rows,
        colWidths=[
            4.2 * cm,
            4.55 * cm,
            4.2 * cm,
            4.55 * cm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BLUE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER_GRAY,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        summary_table
    )


    if is_defect_profile:
        story.append(
            Paragraph(
                "3. Kusur Kontrol Sonuçları",
                styles["SectionHeading"],
            )
        )

        defect_rows = [
            [
                safe_paragraph(
                    "Kusur Türü",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Tespit Sayısı",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Durum",
                    styles["BodyBold"],
                ),
            ]
        ]

        defect_statuses = []

        for class_name in forbidden_objects:
            detected_count = int(
                detected_counts.get(class_name, 0)
            )

            is_suitable = detected_count == 0
            defect_statuses.append(is_suitable)

            visible_name = display_names.get(
                class_name,
                class_name,
            )

            defect_rows.append(
                [
                    safe_paragraph(
                        visible_name,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        detected_count,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        (
                            "Kusur tespit edilmedi"
                            if is_suitable
                            else "Kusur tespit edildi"
                        ),
                        styles["Body"],
                    ),
                ]
            )

        defect_table = Table(
            defect_rows,
            colWidths=[
                7.5 * cm,
                3.5 * cm,
                6.5 * cm,
            ],
            repeatRows=1,
        )

        defect_commands = [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                NAVY,
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER_GRAY,
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER",
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ]

        for row_index, is_suitable in enumerate(
            defect_statuses,
            start=1,
        ):
            defect_commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    (
                        PASS_BACKGROUND
                        if is_suitable
                        else FAIL_BACKGROUND
                    ),
                )
            )

        defect_table.setStyle(
            TableStyle(defect_commands)
        )

        story.append(defect_table)
    else:
        story.append(
            Paragraph(
                "3. Bileşen Kontrol Sonuçları",
                styles["SectionHeading"],
            )
        )

        component_rows = [
            [
                safe_paragraph(
                    "Bileşen",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Beklenen",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Bulunan",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Sapma",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Değerlendirme",
                    styles["BodyBold"],
                ),
            ]
        ]

        component_statuses = []

        for class_name, expected_count in expected_objects.items():
            detected_count = int(
                detected_counts.get(class_name, 0)
            )

            status_text, difference = get_component_status(
                expected_count=expected_count,
                detected_count=detected_count,
            )

            visible_name = display_names.get(
                class_name,
                class_name,
            )

            component_statuses.append(
                difference == 0
            )

            difference_text = (
                "0"
                if difference == 0
                else f"{difference:+d}"
            )

            component_rows.append(
                [
                    safe_paragraph(
                        visible_name,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        expected_count,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        detected_count,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        difference_text,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        status_text,
                        styles["Body"],
                    ),
                ]
            )

        component_table = Table(
            component_rows,
            colWidths=[
                5.2 * cm,
                2.2 * cm,
                2.2 * cm,
                1.8 * cm,
                6.1 * cm,
            ],
            repeatRows=1,
        )

        component_commands = [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                NAVY,
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER_GRAY,
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "ALIGN",
                (1, 1),
                (3, -1),
                "CENTER",
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ]

        for row_index, is_suitable in enumerate(
            component_statuses,
            start=1,
        ):
            row_background = (
                PASS_BACKGROUND
                if is_suitable
                else FAIL_BACKGROUND
            )

            component_commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    row_background,
                )
            )

        component_table.setStyle(
            TableStyle(component_commands)
        )

        story.append(component_table)


    story.append(
        Paragraph(
            "4. Uygunsuzluk ve Bulgular",
            styles["SectionHeading"],
        )
    )

    if error_list:
        error_rows = [
            [
                safe_paragraph(
                    "No",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Uygunsuzluk Açıklaması",
                    styles["BodyBold"],
                ),
            ]
        ]

        for index, error in enumerate(
            error_list,
            start=1,
        ):
            error_rows.append(
                [
                    safe_paragraph(
                        index,
                        styles["Body"],
                    ),
                    safe_paragraph(
                        error,
                        styles["Body"],
                    ),
                ]
            )

        errors_table = Table(
            error_rows,
            colWidths=[
                1.0 * cm,
                16.5 * cm,
            ],
            repeatRows=1,
        )

        errors_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        FAIL_RED,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        FAIL_BACKGROUND,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#FCA5A5"),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "ALIGN",
                        (0, 1),
                        (0, -1),
                        "CENTER",
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(
            errors_table
        )

    else:
        suitable_table = Table(
            [
                [
                    safe_paragraph(
                        (
                            (
                                "Analiz edilen seramik yüzey üzerinde "
                                "kenar kırığı, yüzey boşluğu veya çizgisel "
                                "kusur tespit edilmemiştir. Ürün tanımlanan "
                                "görsel kalite kriterlerini sağlamaktadır."
                            )
                            if is_defect_profile
                            else (
                                "Tanımlanan bütün görsel uygunluk "
                                "kriterleri sağlanmıştır. Herhangi "
                                "bir uygunsuzluk tespit edilmemiştir."
                            )
                        ),
                        styles["Body"],
                    )
                ]
            ],
            colWidths=[
                17.5 * cm
            ],
        )

        suitable_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        PASS_BACKGROUND,
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        PASS_GREEN,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                ]
            )
        )

        story.append(
            suitable_table
        )

    story.append(
        Spacer(
            1,
            0.35 * cm,
        )
    )

    story.append(
        create_signature_section(
            styles=styles
        )
    )

    story.append(
        PageBreak()
    )

    story.append(
        create_institution_header(
            styles=styles,
            report_number=report_number,
            report_date=report_date,
        )
    )

    story.append(
        Spacer(
            1,
            0.35 * cm,
        )
    )

    story.append(
        Paragraph(
            "5. Görsel Kanıtlar ve Yapay Zekâ Analizi",
            styles["SectionHeading"],
        )
    )

    original_buffer = pil_image_to_buffer(
        original_image
    )

    annotated_buffer = numpy_image_to_buffer(
        annotated_image
    )

    original_width, original_height = (
        original_image.size
    )

    annotated_height, annotated_width = (
        annotated_image.shape[:2]
    )

    original_pdf_width, original_pdf_height = (
        calculate_pdf_image_size(
            image_width=original_width,
            image_height=original_height,
            max_width=8.1 * cm,
            max_height=10.4 * cm,
        )
    )

    annotated_pdf_width, annotated_pdf_height = (
        calculate_pdf_image_size(
            image_width=annotated_width,
            image_height=annotated_height,
            max_width=8.1 * cm,
            max_height=10.4 * cm,
        )
    )

    images_table = Table(
        [
            [
                safe_paragraph(
                    "Orijinal Kontrol Görüntüsü",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    "Yapay Zekâ Analiz Görüntüsü",
                    styles["BodyBold"],
                ),
            ],
            [
                Image(
                    original_buffer,
                    width=original_pdf_width,
                    height=original_pdf_height,
                ),
                Image(
                    annotated_buffer,
                    width=annotated_pdf_width,
                    height=annotated_pdf_height,
                ),
            ],
        ],
        colWidths=[
            8.75 * cm,
            8.75 * cm,
        ],
    )

    images_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    LIGHT_BLUE,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    BORDER_GRAY,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        images_table
    )

    story.append(
        Spacer(
            1,
            0.45 * cm,
        )
    )

    technical_note = Table(
        [
            [
                safe_paragraph(
                    "TEKNİK AÇIKLAMA",
                    styles["BodyBold"],
                ),
                safe_paragraph(
                    (
                        (
                            "Analiz, özel olarak eğitilmiş YOLO11 nesne "
                            "algılama modeliyle gerçekleştirilmiştir. "
                            "Seramik yüzeyde tespit edilen kusurlar, "
                            "tanımlı kusur türleri ve güven eşiği "
                            "kullanılarak değerlendirilmiştir."
                        )
                        if is_defect_profile
                        else (
                            "Analiz, özel olarak eğitilmiş YOLO11 nesne "
                            "algılama modeliyle gerçekleştirilmiştir. "
                            "Görüntüde tespit edilen nesneler, seçilen "
                            "ürün profiline ait beklenen adet ve konum "
                            "kurallarıyla karşılaştırılmıştır."
                        )
                    ),
                    styles["Body"],
                ),
            ]
        ],
        colWidths=[
            3.2 * cm,
            14.3 * cm,
        ],
    )

    technical_note.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, 0),
                    LIGHT_GRAY,
                ),
                (
                    "BACKGROUND",
                    (1, 0),
                    (1, 0),
                    WARNING_BACKGROUND,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    WARNING_BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        technical_note
    )

    story.append(
        Spacer(
            1,
            0.35 * cm,
        )
    )

    story.append(
        Paragraph(
            (
                "Yasal ve teknik not: Bu belge prototip bir yapay "
                "zekâ uygulaması tarafından üretilmiştir. Rapor, "
                "yetkili laboratuvar veya uzman değerlendirmesinin "
                "yerini tutmaz. Nihai uygunluk kararı, ilgili standart "
                "ve yetkili personel değerlendirmesi doğrultusunda verilmelidir."
            ),
            styles["Small"],
        )
    )

    document.build(
        story,
        onFirstPage=add_page_header_and_footer,
        onLaterPages=add_page_header_and_footer,
    )

    pdf_bytes = report_buffer.getvalue()

    report_buffer.close()

    safe_profile_name = (
        profile_name.lower()
        .replace(" ", "_")
        .replace("ş", "s")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )

    filename = (
        f"{report_number.lower()}_"
        f"{safe_profile_name}.pdf"
    )

    return (
        pdf_bytes,
        filename,
    )