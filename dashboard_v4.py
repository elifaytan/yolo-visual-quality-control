from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


DEFAULT_PROFILES = {
    "Şişe ve Telefon Kontrolü": {
        "expected_objects": {
            "bottle": 1,
            "cell phone": 1,
        },
        "forbidden_objects": ["cup"],
        "position_rules": {
            "bottle": "Sağ",
            "cell phone": "Sol",
        },
    },
    "Masa Ekipmanı Kontrolü": {
        "expected_objects": {
            "laptop": 1,
            "mouse": 1,
        },
        "forbidden_objects": ["cell phone"],
        "position_rules": {
            "laptop": "Orta",
            "mouse": "Sağ",
        },
    },
    "Kişisel Eşya Kontrolü": {
        "expected_objects": {
            "backpack": 1,
            "cell phone": 1,
        },
        "forbidden_objects": ["handbag"],
        "position_rules": {
            "backpack": "Sol",
            "cell phone": "Sağ",
        },
    },
}

POSITION_OPTIONS = [
    "Herhangi",
    "Sol",
    "Orta",
    "Sağ",
]


@dataclass
class Detection:
    class_name: str
    confidence: float
    center_x: float
    center_y: float


@st.cache_resource
def load_model() -> YOLO:
    """YOLO modelini yalnızca bir kez yükler."""
    return YOLO("yolo11n.pt")


def get_horizontal_region(
    center_x: float,
    image_width: int,
) -> str:
    """Nesnenin merkezine göre sol, orta veya sağ bölgeyi belirler."""

    first_boundary = image_width / 3
    second_boundary = (image_width * 2) / 3

    if center_x < first_boundary:
        return "Sol"

    if center_x < second_boundary:
        return "Orta"

    return "Sağ"


def extract_detections(
    result,
    model: YOLO,
) -> list[Detection]:
    """YOLO sonucundaki sınıf, güven ve kutu merkezi bilgilerini çıkarır."""

    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    class_ids = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    boxes = result.boxes.xyxy.tolist()

    for class_id, confidence, box in zip(
        class_ids,
        confidences,
        boxes,
    ):
        x1, y1, x2, y2 = box

        detections.append(
            Detection(
                class_name=model.names[int(class_id)],
                confidence=float(confidence),
                center_x=(x1 + x2) / 2,
                center_y=(y1 + y2) / 2,
            )
        )

    return detections


def count_detected_objects(
    detections: list[Detection],
) -> Counter:
    """Algılanan bütün nesneleri sınıflarına göre sayar."""

    return Counter(
        detection.class_name
        for detection in detections
    )


def count_objects_in_position(
    detections: list[Detection],
    class_name: str,
    required_position: str,
    image_width: int,
) -> int:
    """Belirtilen sınıfın istenen bölgede kaç kez bulunduğunu hesaplar."""

    if required_position == "Herhangi":
        return sum(
            detection.class_name == class_name
            for detection in detections
        )

    count = 0

    for detection in detections:
        if detection.class_name != class_name:
            continue

        detected_region = get_horizontal_region(
            center_x=detection.center_x,
            image_width=image_width,
        )

        if detected_region == required_position:
            count += 1

    return count


def calculate_quality_result(
    detections: list[Detection],
    detected_counts: Counter,
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    position_rules: dict[str, str],
    image_width: int,
) -> tuple[bool, list[str], dict[str, int]]:
    """
    Eksik, fazla, yanlış nesne ve konum kurallarını kontrol eder.
    """

    errors: list[str] = []
    position_counts: dict[str, int] = {}

    # Beklenen adet kontrolü
    for class_name, expected_count in expected_objects.items():
        detected_count = detected_counts.get(class_name, 0)

        if detected_count < expected_count:
            missing_count = expected_count - detected_count

            errors.append(
                f"Eksik nesne — {class_name}: "
                f"{missing_count} adet eksik"
            )

        elif detected_count > expected_count:
            extra_count = detected_count - expected_count

            errors.append(
                f"Fazla nesne — {class_name}: "
                f"{extra_count} adet fazla"
            )

    # İzin verilmeyen nesne kontrolü
    for class_name in forbidden_objects:
        detected_count = detected_counts.get(class_name, 0)

        if detected_count > 0:
            errors.append(
                f"Yanlış nesne — {class_name}: "
                f"{detected_count} adet tespit edildi"
            )

    # Konum kontrolü
    for class_name, required_position in position_rules.items():
        expected_count = expected_objects.get(class_name, 0)

        objects_in_correct_position = count_objects_in_position(
            detections=detections,
            class_name=class_name,
            required_position=required_position,
            image_width=image_width,
        )

        position_counts[class_name] = objects_in_correct_position

        if required_position == "Herhangi":
            continue

        total_detected = detected_counts.get(class_name, 0)

        # Nesne hiç bulunmadıysa eksik nesne hatası zaten oluşturulmuştur.
        if total_detected == 0:
            continue

        if objects_in_correct_position < expected_count:
            errors.append(
                f"Konum hatası — {class_name}: "
                f"{required_position} bölgede bulunmalı"
            )

    return len(errors) == 0, errors, position_counts


def draw_region_guides(
    image: np.ndarray,
) -> np.ndarray:
    """Sol, orta ve sağ bölgeleri görüntü üzerinde gösterir."""

    output = image.copy()
    height, width = output.shape[:2]

    first_x = width // 3
    second_x = (width * 2) // 3

    cv2.line(
        output,
        (first_x, 0),
        (first_x, height),
        (255, 165, 0),
        2,
    )

    cv2.line(
        output,
        (second_x, 0),
        (second_x, height),
        (255, 165, 0),
        2,
    )

    cv2.putText(
        output,
        "SOL",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 165, 0),
        2,
    )

    cv2.putText(
        output,
        "ORTA",
        (first_x + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 165, 0),
        2,
    )

    cv2.putText(
        output,
        "SAG",
        (second_x + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 165, 0),
        2,
    )

    return output


def analyze_image(
    image: Image.Image,
    model: YOLO,
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    position_rules: dict[str, str],
    confidence: float,
):
    """Görüntüyü analiz eder ve bütün kalite sonuçlarını üretir."""

    rgb_image = np.array(image.convert("RGB"))
    image_height, image_width = rgb_image.shape[:2]

    results = model.predict(
        source=rgb_image,
        conf=confidence,
        verbose=False,
    )

    result = results[0]

    detections = extract_detections(
        result=result,
        model=model,
    )

    detected_counts = count_detected_objects(
        detections=detections,
    )

    (
        is_pass,
        errors,
        position_counts,
    ) = calculate_quality_result(
        detections=detections,
        detected_counts=detected_counts,
        expected_objects=expected_objects,
        forbidden_objects=forbidden_objects,
        position_rules=position_rules,
        image_width=image_width,
    )

    annotated_bgr = result.plot()

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

    annotated_rgb = draw_region_guides(
        image=annotated_rgb,
    )

    return (
        annotated_rgb,
        detections,
        detected_counts,
        position_counts,
        is_pass,
        errors,
    )


def create_expected_result_table(
    expected_objects: dict[str, int],
    detected_counts: Counter,
) -> pd.DataFrame:
    """Beklenen nesne adet sonuçlarını tabloya dönüştürür."""

    rows = []

    for class_name, expected_count in expected_objects.items():
        detected_count = detected_counts.get(class_name, 0)

        if detected_count == expected_count:
            status = "Uygun"
        elif detected_count < expected_count:
            status = "Eksik"
        else:
            status = "Fazla"

        rows.append(
            {
                "Nesne": class_name,
                "Beklenen": expected_count,
                "Bulunan": detected_count,
                "Durum": status,
            }
        )

    return pd.DataFrame(rows)


def create_position_result_table(
    expected_objects: dict[str, int],
    position_rules: dict[str, str],
    position_counts: dict[str, int],
    detected_counts: Counter,
) -> pd.DataFrame:
    """Konum kontrolü sonuçlarını tabloya dönüştürür."""

    rows = []

    for class_name, required_position in position_rules.items():
        expected_count = expected_objects.get(class_name, 0)
        correct_position_count = position_counts.get(class_name, 0)
        total_detected = detected_counts.get(class_name, 0)

        if required_position == "Herhangi":
            status = "Kontrol Dışı"
        elif (
            total_detected >= expected_count
            and correct_position_count >= expected_count
        ):
            status = "Uygun"
        elif total_detected == 0:
            status = "Nesne Bulunamadı"
        else:
            status = "Yanlış Konum"

        rows.append(
            {
                "Nesne": class_name,
                "Beklenen Bölge": required_position,
                "Doğru Bölgede Bulunan": correct_position_count,
                "Toplam Bulunan": total_detected,
                "Durum": status,
            }
        )

    return pd.DataFrame(rows)


def create_forbidden_result_table(
    forbidden_objects: list[str],
    detected_counts: Counter,
) -> pd.DataFrame:
    """İzin verilmeyen nesne sonuçlarını tabloya dönüştürür."""

    rows = []

    for class_name in forbidden_objects:
        detected_count = detected_counts.get(class_name, 0)

        rows.append(
            {
                "Nesne": class_name,
                "İzin Verilen": 0,
                "Bulunan": detected_count,
                "Durum": (
                    "Uygun"
                    if detected_count == 0
                    else "Yanlış Nesne"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_profile_settings(
    model: YOLO,
) -> tuple[
    str,
    dict[str, int],
    list[str],
    dict[str, str],
]:
    """Profil, nesne adedi, yasaklı nesne ve konum ayarlarını oluşturur."""

    st.sidebar.header("Kontrol Ayarları")

    selected_profile_name = st.sidebar.selectbox(
        "Hazır ürün profili",
        options=list(DEFAULT_PROFILES.keys()),
    )

    selected_profile = DEFAULT_PROFILES[selected_profile_name]

    st.sidebar.subheader("Beklenen Nesneler")

    expected_objects: dict[str, int] = {}

    for class_name, default_count in selected_profile[
        "expected_objects"
    ].items():
        expected_objects[class_name] = st.sidebar.number_input(
            label=f"{class_name} adedi",
            min_value=0,
            max_value=20,
            value=default_count,
            step=1,
            key=f"count_{selected_profile_name}_{class_name}",
        )

    st.sidebar.subheader("Konum Kuralları")

    position_rules: dict[str, str] = {}

    for class_name in expected_objects:
        default_position = selected_profile[
            "position_rules"
        ].get(class_name, "Herhangi")

        default_index = POSITION_OPTIONS.index(
            default_position
        )

        position_rules[class_name] = st.sidebar.selectbox(
            label=f"{class_name} konumu",
            options=POSITION_OPTIONS,
            index=default_index,
            key=f"position_{selected_profile_name}_{class_name}",
        )

    st.sidebar.subheader("İzin Verilmeyen Nesneler")

    model_class_names = list(model.names.values())

    forbidden_objects = st.sidebar.multiselect(
        "Görülmesi hâlinde FAIL oluşturacak nesneler",
        options=model_class_names,
        default=selected_profile["forbidden_objects"],
        key=f"forbidden_{selected_profile_name}",
    )

    return (
        selected_profile_name,
        expected_objects,
        forbidden_objects,
        position_rules,
    )


def main() -> None:
    st.set_page_config(
        page_title="YOLO Kalite Kontrol",
        page_icon="🔍",
        layout="wide",
    )

    st.title(
        "YOLO Tabanlı Modüler Görsel Kalite Kontrol Platformu"
    )

    st.caption(
        "Nesne varlığı, adet, yanlış nesne ve konum uygunluğu "
        "kontrollerini gerçekleştiren yapay zekâ destekli prototip"
    )

    model = load_model()

    (
        selected_profile_name,
        expected_objects,
        forbidden_objects,
        position_rules,
    ) = build_profile_settings(model)

    with st.sidebar:
        confidence = st.slider(
            "Güven eşiği",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
        )

        input_type = st.radio(
            "Görüntü kaynağı",
            options=[
                "Kameradan fotoğraf çek",
                "Bilgisayardan görsel yükle",
            ],
        )

        st.info(
            "Konum kontrolünde görüntü yatay olarak "
            "sol, orta ve sağ olmak üzere üç bölgeye ayrılır."
        )

    st.subheader(f"Seçilen profil: {selected_profile_name}")

    summary_columns = st.columns(len(expected_objects))

    for column, (class_name, expected_count) in zip(
        summary_columns,
        expected_objects.items(),
    ):
        with column:
            required_position = position_rules[class_name]

            st.metric(
                label=class_name,
                value=f"{expected_count} adet",
                delta=f"Konum: {required_position}",
                delta_color="off",
            )

    image_file = None

    if input_type == "Kameradan fotoğraf çek":
        image_file = st.camera_input(
            "Kontrol edilecek nesneleri kameraya gösterin"
        )
    else:
        image_file = st.file_uploader(
            "Analiz edilecek görseli seçin",
            type=["jpg", "jpeg", "png"],
        )

    if image_file is None:
        st.warning(
            "Kontrolü başlatmak için fotoğraf çekin "
            "veya bir görüntü yükleyin."
        )
        return

    image = Image.open(image_file)

    with st.spinner("Görüntü analiz ediliyor..."):
        (
            annotated_image,
            detections,
            detected_counts,
            position_counts,
            is_pass,
            errors,
        ) = analyze_image(
            image=image,
            model=model,
            expected_objects=expected_objects,
            forbidden_objects=forbidden_objects,
            position_rules=position_rules,
            confidence=confidence,
        )

    original_column, detection_column = st.columns(2)

    with original_column:
        st.subheader("Orijinal Görüntü")

        st.image(
            image,
            use_container_width=True,
        )

    with detection_column:
        st.subheader("YOLO Tespit ve Konum Sonucu")

        st.image(
            annotated_image,
            use_container_width=True,
        )

    st.divider()

    status_column, detected_column, error_column = st.columns(3)

    with status_column:
        if is_pass:
            st.success("PASS — Ürün uygun")
        else:
            st.error("FAIL — Ürün uygun değil")

    with detected_column:
        relevant_classes = (
            set(expected_objects)
            | set(forbidden_objects)
        )

        relevant_detected_count = sum(
            detected_counts.get(class_name, 0)
            for class_name in relevant_classes
        )

        st.metric(
            "Kontrol edilen nesne sayısı",
            relevant_detected_count,
        )

    with error_column:
        st.metric(
            "Uygunsuzluk sayısı",
            len(errors),
        )

    st.subheader("Beklenen Nesne ve Adet Kontrolleri")

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

    st.subheader("Yanlış Nesne Kontrolleri")

    if forbidden_objects:
        forbidden_table = create_forbidden_result_table(
            forbidden_objects=forbidden_objects,
            detected_counts=detected_counts,
        )

        st.dataframe(
            forbidden_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("İzin verilmeyen nesne seçilmedi.")

    if errors:
        st.subheader("Belirlenen Uygunsuzluklar")

        for error in errors:
            st.error(error)
    else:
        st.success(
            "Tanımlanan bütün kalite kuralları sağlandı."
        )

    with st.expander("Algılanan bütün nesnelerin ayrıntıları"):
        if detections:
            detection_table = pd.DataFrame(
                [
                    {
                        "Nesne": detection.class_name,
                        "Güven": round(
                            detection.confidence,
                            3,
                        ),
                        "Yatay Konum": get_horizontal_region(
                            center_x=detection.center_x,
                            image_width=image.width,
                        ),
                    }
                    for detection in detections
                ]
            )

            st.dataframe(
                detection_table,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("Görüntüde herhangi bir nesne algılanmadı.")


if __name__ == "__main__":
    main()