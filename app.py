import hashlib
import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from database_manager import (
    delete_all_inspections,
    get_inspection_history,
    get_summary_statistics,
    initialize_database,
    save_inspection,
)
from detector import (
    detect_objects,
    get_horizontal_region,
    load_yolo_model,
)
from image_manager import (
    save_annotated_image,
    save_original_image,
)
from profiles import (
    POSITION_OPTIONS,
    PRODUCT_PROFILES,
)
from quality_engine import (
    create_expected_result_table,
    create_forbidden_result_table,
    create_position_result_table,
    evaluate_quality,
)
from report_generator import (
    create_quality_report_pdf,
)


@st.cache_resource
def load_cached_model(
    model_path: str,
) -> YOLO:
    """YOLO modelini önbelleğe alarak yükler."""

    return load_yolo_model(
        model_path=model_path
    )


def add_custom_style() -> None:
    """Uygulamaya kurumsal görünüm kazandırır."""

    st.markdown(
        """
        <style>
        :root {
            --navy: #12355b;
            --navy-dark: #0b2744;
            --blue: #1f6fb2;
            --surface: #ffffff;
            --border: #dbe5ef;
            --text: #172b3a;
            --muted: #64748b;
            --success: #15803d;
            --danger: #c62828;
        }

        .stApp {
            background:
                radial-gradient(circle at 90% 0%, rgba(31, 111, 178, 0.08), transparent 22rem),
                linear-gradient(180deg, #fbfdff 0%, #f6f9fc 100%);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
        }

        .hero-card {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 28px 30px;
            margin-bottom: 20px;
            background:
                linear-gradient(135deg, rgba(18, 53, 91, 0.98), rgba(31, 111, 178, 0.92));
            box-shadow: 0 18px 42px rgba(15, 43, 70, 0.14);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 270px;
            height: 270px;
            right: -95px;
            top: -145px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.10);
        }

        .hero-badge {
            display: inline-block;
            padding: 6px 11px;
            margin-bottom: 11px;
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 999px;
            color: #e8f3ff;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .main-title {
            position: relative;
            z-index: 1;
            max-width: 960px;
            color: #ffffff;
            font-size: clamp(2rem, 3vw, 2.7rem);
            line-height: 1.12;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .sub-title {
            position: relative;
            z-index: 1;
            max-width: 880px;
            color: rgba(255, 255, 255, 0.82);
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0;
        }

        .workflow-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 0 0 20px 0;
        }

        .workflow-item {
            display: flex;
            align-items: center;
            gap: 10px;
            min-height: 58px;
            padding: 11px 13px;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 5px 16px rgba(15, 43, 70, 0.05);
        }

        .workflow-number {
            display: grid;
            place-items: center;
            flex: 0 0 30px;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--navy);
            color: white;
            font-size: 0.85rem;
            font-weight: 750;
        }

        .workflow-text {
            color: var(--text);
            font-size: 0.87rem;
            font-weight: 650;
        }

        .info-card {
            min-height: 108px;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 8px 22px rgba(15, 43, 70, 0.055);
        }

        .info-card strong {
            display: inline-block;
            color: var(--navy);
            margin-bottom: 5px;
        }

        .camera-panel {
            max-width: 760px;
            margin: 20px auto 10px auto;
            padding: 15px 18px;
            border: 1px solid #cfe0ef;
            border-radius: 16px;
            background: linear-gradient(180deg, #f8fbfe 0%, #edf5fb 100%);
            text-align: center;
        }

        .camera-panel-title {
            color: var(--navy);
            font-size: 1.1rem;
            font-weight: 780;
            margin-bottom: 4px;
        }

        .camera-panel-text {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        div[data-testid="stCameraInput"] {
            max-width: 680px;
            margin: 0 auto;
        }

        div[data-testid="stCameraInput"] video {
            max-height: 390px;
            object-fit: cover;
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 12px 26px rgba(15, 43, 70, 0.10);
        }

        div[data-testid="stCameraInput"] button {
            min-height: 42px;
            border-radius: 10px;
            font-weight: 700;
        }

        div[data-testid="stFileUploader"] {
            max-width: 780px;
            margin: 0 auto;
        }

        div[data-testid="stMetric"] {
            min-height: 104px;
            padding: 14px 16px;
            border: 1px solid var(--border);
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 7px 20px rgba(15, 43, 70, 0.045);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--navy-dark);
        }

        .status-pass,
        .status-fail {
            min-height: 104px;
            display: grid;
            place-items: center;
            border-radius: 15px;
            padding: 18px;
            text-align: center;
            font-size: 1.28rem;
            font-weight: 800;
        }

        .status-pass {
            border: 2px solid #22a447;
            color: #11682e;
            background: linear-gradient(180deg, #f2fff6, #e8f8ed);
        }

        .status-fail {
            border: 2px solid #e04747;
            color: #a61e1e;
            background: linear-gradient(180deg, #fff7f7, #fdecec);
        }

        .component-pass,
        .component-fail {
            min-height: 104px;
            border-radius: 12px;
            padding: 13px 15px;
            margin-bottom: 8px;
        }

        .component-pass {
            border: 1px solid rgba(21, 128, 61, 0.24);
            border-left: 5px solid var(--success);
            background: #effaf2;
        }

        .component-fail {
            border: 1px solid rgba(198, 40, 40, 0.22);
            border-left: 5px solid var(--danger);
            background: #fff1f1;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f3f7fb 0%, #eef4f9 100%);
            border-right: 1px solid #d9e4ee;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--navy-dark);
        }

        div[data-testid="stTabs"] button {
            padding-top: 0.72rem;
            padding-bottom: 0.72rem;
            font-weight: 700;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 44px;
            border-radius: 10px;
            font-weight: 720;
        }

        .stDownloadButton > button {
            border-color: #b9ccdc;
            color: var(--navy);
            background: #ffffff;
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        div[data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 22px rgba(15, 43, 70, 0.07);
        }

        @media (max-width: 900px) {
            .workflow-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            div[data-testid="stCameraInput"] video {
                max-height: 320px;
            }
        }

        @media (max-width: 620px) {
            .hero-card {
                padding: 22px 20px;
            }

            .workflow-strip {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_display_name(
    class_name: str,
    display_names: dict[str, str],
) -> str:
    """Model sınıfının Türkçe görünen adını döndürür."""

    return display_names.get(
        class_name,
        class_name,
    )


def build_profile_settings(
    selected_profile_name: str,
    selected_profile: dict,
    model: YOLO,
) -> tuple[
    dict[str, int],
    list[str],
    dict[str, str],
    float,
    str,
]:
    """Seçilen ürün profiline ait kontrolleri oluşturur."""

    display_names = selected_profile["display_names"]
    configured_expected = selected_profile["expected_objects"]
    is_defect_profile = not configured_expected

    expected_objects: dict[str, int] = {}

    if configured_expected:
        st.sidebar.subheader("Beklenen Bileşenler")

        for class_name, default_count in configured_expected.items():
            visible_name = get_display_name(
                class_name,
                display_names,
            )

            expected_objects[class_name] = st.sidebar.number_input(
                label=f"{visible_name} adedi",
                min_value=0,
                max_value=20,
                value=default_count,
                step=1,
                key=(
                    f"count_{selected_profile_name}_"
                    f"{class_name}"
                ),
            )

        st.sidebar.subheader("Konum Kuralları")

    position_rules: dict[str, str] = {}

    for class_name in expected_objects:
        default_position = selected_profile[
            "position_rules"
        ].get(
            class_name,
            "Herhangi",
        )

        visible_name = get_display_name(
            class_name,
            display_names,
        )

        position_rules[class_name] = st.sidebar.selectbox(
            label=f"{visible_name} konumu",
            options=POSITION_OPTIONS,
            index=POSITION_OPTIONS.index(
                default_position
            ),
            key=(
                f"position_{selected_profile_name}_"
                f"{class_name}"
            ),
        )

    if is_defect_profile:
        st.sidebar.subheader("Kontrol Edilen Kusur Türleri")
        forbidden_label = (
            "Tespit edilmesi hâlinde uygunsuzluk "
            "oluşturacak yüzey kusurları"
        )
    else:
        st.sidebar.subheader("İzin Verilmeyen Nesneler")
        forbidden_label = (
            "Tespit edilmesi hâlinde uygunsuzluk "
            "oluşturacak nesneler"
        )

    model_class_names = list(model.names.values())

    forbidden_objects = st.sidebar.multiselect(
        forbidden_label,
        options=model_class_names,
        default=selected_profile["forbidden_objects"],
        key=f"forbidden_{selected_profile_name}",
    )

    confidence = st.sidebar.slider(
        "Güven eşiği",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
    )

    input_type = st.sidebar.radio(
        "Görüntü kaynağı",
        options=[
            "Kameradan fotoğraf çek",
            "Bilgisayardan görsel yükle",
        ],
    )

    return (
        expected_objects,
        forbidden_objects,
        position_rules,
        confidence,
        input_type,
    )


def create_inspection_signature(
    image_bytes: bytes,
    profile_name: str,
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    position_rules: dict[str, str],
    confidence: float,
) -> str:
    """Aynı kontrolün iki kez kaydedilmesini önleyen imza üretir."""

    settings_text = json.dumps(
        {
            "profile_name": profile_name,
            "expected_objects": expected_objects,
            "forbidden_objects": forbidden_objects,
            "position_rules": position_rules,
            "confidence": confidence,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    signature_source = (
        image_bytes
        + settings_text.encode("utf-8")
    )

    return hashlib.sha256(
        signature_source
    ).hexdigest()


def show_component_cards(
    expected_objects: dict[str, int],
    detected_counts,
    display_names: dict[str, str],
) -> None:
    """Beklenen bileşenleri uygunluk kartlarıyla gösterir."""

    st.subheader(
        "Bileşen Uygunluk Özeti"
    )

    columns = st.columns(
        max(
            1,
            len(expected_objects),
        )
    )

    for column, (
        class_name,
        expected_count,
    ) in zip(
        columns,
        expected_objects.items(),
    ):
        detected_count = detected_counts.get(
            class_name,
            0,
        )

        is_suitable = (
            detected_count == expected_count
        )

        visible_name = get_display_name(
            class_name,
            display_names,
        )

        card_class = (
            "component-pass"
            if is_suitable
            else "component-fail"
        )

        icon = (
            "✓"
            if is_suitable
            else "✕"
        )

        with column:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <strong>{icon} {visible_name}</strong><br>
                    Beklenen: {expected_count}<br>
                    Bulunan: {detected_count}
                </div>
                """,
                unsafe_allow_html=True,
            )



def show_defect_cards(
    forbidden_objects: list[str],
    detected_counts,
    display_names: dict[str, str],
) -> None:
    """Kontrol edilen kusur türlerini sonuç kartlarıyla gösterir."""

    st.subheader("Seramik Kusur Kontrol Özeti")

    if not forbidden_objects:
        st.info("Kontrol edilecek kusur türü tanımlanmadı.")
        return

    columns = st.columns(len(forbidden_objects))

    for column, class_name in zip(
        columns,
        forbidden_objects,
    ):
        detected_count = detected_counts.get(
            class_name,
            0,
        )

        is_suitable = detected_count == 0
        visible_name = get_display_name(
            class_name,
            display_names,
        )

        card_class = (
            "component-pass"
            if is_suitable
            else "component-fail"
        )

        icon = "✓" if is_suitable else "✕"
        status_text = (
            "Kusur tespit edilmedi"
            if is_suitable
            else f"{detected_count} adet kusur tespit edildi"
        )

        with column:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <strong>{icon} {visible_name}</strong><br>
                    {status_text}
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_detection_details(
    detections,
    image_width: int,
    display_names: dict[str, str],
    show_position: bool,
) -> None:
    """Algılanan nesnelerin teknik ayrıntılarını gösterir."""

    with st.expander(
        "Algılanan nesnelerin teknik ayrıntıları"
    ):
        if not detections:
            st.write(
                "Görüntüde herhangi bir nesne algılanmadı."
            )
            return

        rows = []

        for detection in detections:
            row = {
                "Nesne": get_display_name(
                    detection.class_name,
                    display_names,
                ),
                "Model Sınıfı": detection.class_name,
                "Güven Oranı": round(
                    detection.confidence,
                    3,
                ),
            }

            if show_position:
                row["Yatay Konum"] = (
                    get_horizontal_region(
                        center_x=detection.center_x,
                        image_width=image_width,
                    )
                )

            rows.append(row)

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


def show_inspection_history() -> None:
    """Kontrol geçmişi ve istatistik ekranını gösterir."""

    st.subheader(
        "Kontrol Geçmişi ve İstatistikler"
    )

    statistics = get_summary_statistics()

    (
        total_column,
        pass_column,
        fail_column,
        rate_column,
    ) = st.columns(4)

    with total_column:
        st.metric(
            "Toplam Kontrol",
            statistics["total_count"],
        )

    with pass_column:
        st.metric(
            "Uygun",
            statistics["pass_count"],
        )

    with fail_column:
        st.metric(
            "Uygun Değil",
            statistics["fail_count"],
        )

    with rate_column:
        st.metric(
            "Uygunluk Oranı",
            f"%{statistics['pass_rate']}",
        )

    history = get_inspection_history()

    if history.empty:
        st.info(
            (
                "Henüz kaydedilmiş kalite kontrol "
                "sonucu bulunmuyor."
            )
        )
        return

    (
        filter_column,
        result_column,
    ) = st.columns(2)

    profile_options = [
        "Tümü",
        *sorted(
            history[
                "Ürün Profili"
            ].unique().tolist()
        ),
    ]

    with filter_column:
        selected_profile_filter = st.selectbox(
            "Ürün profiline göre filtrele",
            options=profile_options,
            key="history_profile_filter",
        )

    with result_column:
        selected_result_filter = st.selectbox(
            "Sonuca göre filtrele",
            options=[
                "Tümü",
                "PASS",
                "FAIL",
            ],
            key="history_result_filter",
        )

    filtered_history = history.copy()

    if selected_profile_filter != "Tümü":
        filtered_history = filtered_history[
            filtered_history["Ürün Profili"]
            == selected_profile_filter
        ]

    if selected_result_filter != "Tümü":
        filtered_history = filtered_history[
            filtered_history["Sonuç"]
            == selected_result_filter
        ]

    st.dataframe(
        filtered_history,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_history.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label=(
            "Kontrol geçmişini CSV olarak indir"
        ),
        data=csv_data,
        file_name=(
            "kalite_kontrol_gecmisi.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander(
        "Kontrol geçmişini temizle"
    ):
        st.warning(
            (
                "Bu işlem bütün kontrol kayıtlarını "
                "kalıcı olarak siler."
            )
        )

        confirm_delete = st.checkbox(
            (
                "Bütün kayıtların silinmesini "
                "onaylıyorum."
            ),
            key="confirm_delete_history",
        )

        if st.button(
            "Bütün Kontrol Geçmişini Sil",
            type="primary",
            disabled=not confirm_delete,
        ):
            delete_all_inspections()

            st.session_state.pop(
                "last_saved_signature",
                None,
            )

            st.success(
                "Kontrol geçmişi silindi."
            )

            st.rerun()


def show_new_inspection(
    selected_profile_name: str,
    selected_profile: dict,
    model: YOLO,
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    position_rules: dict[str, str],
    confidence: float,
    input_type: str,
) -> None:
    """Yeni kalite kontrol ekranını gösterir."""

    display_names = selected_profile["display_names"]
    is_defect_profile = not expected_objects

    control_time = datetime.now().strftime(
        "%d.%m.%Y %H:%M:%S"
    )

    st.subheader(
        f"Ürün Profili: {selected_profile_name}"
    )

    information_column, standard_column = st.columns(2)

    with information_column:
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Kontrol Açıklaması</strong><br>
                {selected_profile["description"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with standard_column:
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Uygulanan Kriter</strong><br>
                {selected_profile["standard_name"]}<br>
                <small>Kontrol zamanı: {control_time}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if expected_objects:
        summary_columns = st.columns(
            len(expected_objects)
        )

        for column, (
            class_name,
            expected_count,
        ) in zip(
            summary_columns,
            expected_objects.items(),
        ):
            with column:
                visible_name = get_display_name(
                    class_name,
                    display_names,
                )

                st.metric(
                    label=visible_name,
                    value=f"{expected_count} adet",
                    delta=(
                        "Beklenen konum: "
                        f"{position_rules[class_name]}"
                    ),
                    delta_color="off",
                )
    else:
        st.info(
            (
                "Bu profilde bileşen ve konum kontrolü yerine "
                "seramik yüzey kusuru tespiti yapılmaktadır."
            )
        )

    image_file = None

    if is_defect_profile:
        camera_instruction = (
            "Seramik yüzeyi tamamen kadraja yerleştirin. "
            "Yüzeyin net, yeterince aydınlık ve mümkün olduğunca "
            "gölgesiz görünmesini sağlayın."
        )
    else:
        camera_instruction = (
            "Ürünü kadrajın orta bölümüne yerleştirin ve şişe "
            "gövdesi, kapak ile etiketin net biçimde görünmesini sağlayın."
        )

    st.markdown(
        f"""
        <div class="camera-panel">
            <div class="camera-panel-title">Görüntü Alma Alanı</div>
            <div class="camera-panel-text">
                {camera_instruction}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if input_type == "Kameradan fotoğraf çek":
        (
            camera_left_column,
            camera_column,
            camera_right_column,
        ) = st.columns([1.45, 2.10, 1.45])

        with camera_column:
            image_file = st.camera_input(
                "Ürün fotoğrafını çekin",
                key="quality_camera_input",
                label_visibility="collapsed",
            )
    else:
        (
            uploader_left_column,
            uploader_column,
            uploader_right_column,
        ) = st.columns([1.20, 2.60, 1.20])

        with uploader_column:
            image_file = st.file_uploader(
                "Analiz edilecek görseli seçin",
                type=["jpg", "jpeg", "png"],
                key="quality_file_uploader",
            )

    if image_file is None:
        st.warning(
            (
                "Kontrolü başlatmak için fotoğraf çekin "
                "veya bilgisayarınızdan bir görsel yükleyin."
            )
        )
        return

    image_bytes = image_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    with st.spinner(
        "Görüntü yapay zekâ modeliyle analiz ediliyor..."
    ):
        (
            annotated_image,
            detections,
            detected_counts,
            image_width,
        ) = detect_objects(
            image=image,
            model=model,
            confidence=confidence,
            show_regions=bool(expected_objects),
        )

        (
            is_pass,
            errors,
            position_counts,
        ) = evaluate_quality(
            detections=detections,
            detected_counts=detected_counts,
            expected_objects=expected_objects,
            forbidden_objects=forbidden_objects,
            position_rules=position_rules,
            image_width=image_width,
        )

    original_column, detection_column = st.columns(2)

    with original_column:
        st.subheader("Orijinal Kontrol Görüntüsü")
        st.image(
            image,
            use_container_width=True,
        )

    with detection_column:
        st.subheader("Yapay Zekâ Analiz Görüntüsü")
        st.image(
            annotated_image,
            use_container_width=True,
        )

    st.divider()

    status_column, object_column, error_column = st.columns(3)

    with status_column:
        if is_pass:
            st.markdown(
                """
                <div class="status-pass">
                    ✓ UYGUN — PASS
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-fail">
                    ✕ UYGUN DEĞİL — FAIL
                </div>
                """,
                unsafe_allow_html=True,
            )

    relevant_classes = (
        set(expected_objects)
        | set(forbidden_objects)
    )

    relevant_detected_count = sum(
        detected_counts.get(class_name, 0)
        for class_name in relevant_classes
    )

    with object_column:
        st.metric(
            (
                "Tespit Edilen Kusur"
                if is_defect_profile
                else "Kontrol Edilen Nesne"
            ),
            relevant_detected_count,
        )

    with error_column:
        st.metric(
            "Uygunsuzluk Sayısı",
            len(errors),
        )

    if is_defect_profile:
        show_defect_cards(
            forbidden_objects=forbidden_objects,
            detected_counts=detected_counts,
            display_names=display_names,
        )
    else:
        show_component_cards(
            expected_objects=expected_objects,
            detected_counts=detected_counts,
            display_names=display_names,
        )

    if errors:
        st.subheader("Belirlenen Uygunsuzluklar")

        for error in errors:
            st.error(error)
    else:
        if is_defect_profile:
            st.success(
                (
                    "Analiz edilen seramik yüzey üzerinde "
                    "tanımlı kusur türlerinden hiçbiri tespit edilmedi."
                )
            )
        else:
            st.success(
                (
                    "Tanımlanan bütün görsel uygunluk "
                    "kriterleri sağlandı."
                )
            )

    inspection_signature = create_inspection_signature(
        image_bytes=image_bytes,
        profile_name=selected_profile_name,
        expected_objects=expected_objects,
        forbidden_objects=forbidden_objects,
        position_rules=position_rules,
        confidence=confidence,
    )

    pdf_bytes, pdf_filename = create_quality_report_pdf(
        profile_name=selected_profile_name,
        standard_name=selected_profile["standard_name"],
        description=selected_profile["description"],
        is_pass=is_pass,
        detected_counts=detected_counts,
        expected_objects=expected_objects,
        forbidden_objects=forbidden_objects,
        display_names=display_names,
        errors=errors,
        confidence=confidence,
        original_image=image,
        annotated_image=annotated_image,
        model_path=selected_profile["model_path"],
    )

    st.download_button(
        label="PDF Uygunluk Raporunu İndir",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
    )

    if st.button(
        "Kontrol Sonucunu ve Görüntüleri Kaydet",
        type="primary",
        use_container_width=True,
    ):
        last_saved_signature = st.session_state.get(
            "last_saved_signature"
        )

        if last_saved_signature == inspection_signature:
            st.warning(
                "Bu kontrol sonucu daha önce kaydedildi."
            )
        else:
            original_path = save_original_image(
                image=image,
                profile_name=selected_profile_name,
                is_pass=is_pass,
            )

            annotated_path = save_annotated_image(
                image=annotated_image,
                profile_name=selected_profile_name,
                is_pass=is_pass,
            )

            record_id = save_inspection(
                profile_name=selected_profile_name,
                is_pass=is_pass,
                detected_objects=dict(detected_counts),
                expected_objects=expected_objects,
                forbidden_objects=forbidden_objects,
                position_rules=position_rules,
                errors=errors,
                confidence=confidence,
            )

            st.session_state[
                "last_saved_signature"
            ] = inspection_signature

            st.success(
                (
                    "Kontrol kaydedildi. "
                    f"Kontrol numarası: {record_id}"
                )
            )

            st.caption(
                f"Orijinal görüntü: {original_path}"
            )
            st.caption(
                f"İşlenmiş görüntü: {annotated_path}"
            )

    if expected_objects:
        st.subheader(
            "Beklenen Bileşen ve Adet Kontrolleri"
        )

        expected_table = create_expected_result_table(
            expected_objects=expected_objects,
            detected_counts=detected_counts,
        )

        st.dataframe(
            expected_table,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Konum Kontrolleri")

        position_table = create_position_result_table(
            expected_objects=expected_objects,
            position_rules=position_rules,
            position_counts=position_counts,
            detected_counts=detected_counts,
        )

        st.dataframe(
            position_table,
            use_container_width=True,
            hide_index=True,
        )

    if forbidden_objects:
        st.subheader(
            (
                "Kontrol Edilen Kusur Türleri"
                if is_defect_profile
                else "İzin Verilmeyen Nesne Kontrolleri"
            )
        )

        if is_defect_profile:
            defect_rows = []

            for class_name in forbidden_objects:
                detected_count = detected_counts.get(
                    class_name,
                    0,
                )

                defect_rows.append(
                    {
                        "Kusur Türü": get_display_name(
                            class_name,
                            display_names,
                        ),
                        "Tespit Sayısı": detected_count,
                        "Durum": (
                            "Uygun"
                            if detected_count == 0
                            else "Uygun Değil"
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(defect_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            forbidden_table = create_forbidden_result_table(
                forbidden_objects=forbidden_objects,
                detected_counts=detected_counts,
            )

            st.dataframe(
                forbidden_table,
                use_container_width=True,
                hide_index=True,
            )
    elif not is_defect_profile:
        st.info(
            "İzin verilmeyen nesne tanımlanmadı."
        )

    show_detection_details(
        detections=detections,
        image_width=image_width,
        display_names=display_names,
        show_position=bool(expected_objects),
    )


def main() -> None:
    st.set_page_config(
        page_title=(
            "Görsel Kalite Kontrol Platformu"
        ),
        page_icon="🔍",
        layout="wide",
    )

    initialize_database()
    add_custom_style()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">TSE Kalite Kampüsü • Akıllı Kontrol Sistemi</div>
            <div class="main-title">
                Yapay Zekâ Destekli Görsel Kalite Kontrol Platformu
            </div>
            <div class="sub-title">
                Ürün bileşenlerini varlık, adet, konum ve görsel uygunluk
                kriterlerine göre değerlendirir; sonuçları kayıt altına alır
                ve PDF raporu olarak sunar.
            </div>
        </div>

        <div class="workflow-strip">
            <div class="workflow-item">
                <div class="workflow-number">1</div>
                <div class="workflow-text">Kontrol kriterlerini belirle</div>
            </div>
            <div class="workflow-item">
                <div class="workflow-number">2</div>
                <div class="workflow-text">Kamera veya dosyadan görüntü al</div>
            </div>
            <div class="workflow-item">
                <div class="workflow-number">3</div>
                <div class="workflow-text">Yapay zekâ analizini gerçekleştir</div>
            </div>
            <div class="workflow-item">
                <div class="workflow-number">4</div>
                <div class="workflow-text">Sonucu kaydet ve raporla</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header(
        "Kontrol Ayarları"
    )

    selected_profile_name = (
        st.sidebar.selectbox(
            "Ürün profili",
            options=list(
                PRODUCT_PROFILES.keys()
            ),
        )
    )

    selected_profile = PRODUCT_PROFILES[
        selected_profile_name
    ]

    model = load_cached_model(
        model_path=selected_profile[
            "model_path"
        ]
    )

    (
        expected_objects,
        forbidden_objects,
        position_rules,
        confidence,
        input_type,
    ) = build_profile_settings(
        selected_profile_name=(
            selected_profile_name
        ),
        selected_profile=selected_profile,
        model=model,
    )

    if expected_objects:
        st.sidebar.info(
            (
                "Konum kontrolü için görüntü yatay "
                "olarak sol, orta ve sağ bölgelerine "
                "ayrılır."
            )
        )
    else:
        st.sidebar.info(
            (
                "Seramik yüzeyde kenar kırığı, yüzey boşluğu "
                "ve çizgisel kusur tespiti yapılır."
            )
        )

    (
        inspection_tab,
        history_tab,
    ) = st.tabs(
        [
            "Yeni Kalite Kontrolü",
            "Kontrol Geçmişi",
        ]
    )

    with inspection_tab:
        show_new_inspection(
            selected_profile_name=(
                selected_profile_name
            ),
            selected_profile=selected_profile,
            model=model,
            expected_objects=expected_objects,
            forbidden_objects=forbidden_objects,
            position_rules=position_rules,
            confidence=confidence,
            input_type=input_type,
        )

    with history_tab:
        show_inspection_history()


if __name__ == "__main__":
    main()