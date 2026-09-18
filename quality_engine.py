from collections import Counter

import pandas as pd

from detector import Detection, get_horizontal_region


def count_objects_in_position(
    detections: list[Detection],
    class_name: str,
    required_position: str,
    image_width: int,
) -> int:
    """Belirli nesnenin doğru bölgede bulunan adedini hesaplar."""

    if required_position == "Herhangi":
        return sum(
            detection.class_name == class_name
            for detection in detections
        )

    correct_position_count = 0

    for detection in detections:
        if detection.class_name != class_name:
            continue

        detected_region = get_horizontal_region(
            center_x=detection.center_x,
            image_width=image_width,
        )

        if detected_region == required_position:
            correct_position_count += 1

    return correct_position_count


def evaluate_quality(
    detections: list[Detection],
    detected_counts: Counter,
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    position_rules: dict[str, str],
    image_width: int,
) -> tuple[bool, list[str], dict[str, int]]:
    """
    Bütün kalite kontrol kurallarını değerlendirir.

    Kontroller:
    - Eksik nesne
    - Fazla nesne
    - Yanlış nesne
    - Konum uygunluğu
    """

    errors: list[str] = []
    position_counts: dict[str, int] = {}

    # Eksik ve fazla nesne kontrolü
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

    # Yanlış veya izin verilmeyen nesne kontrolü
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

        correct_position_count = count_objects_in_position(
            detections=detections,
            class_name=class_name,
            required_position=required_position,
            image_width=image_width,
        )

        position_counts[class_name] = correct_position_count

        if required_position == "Herhangi":
            continue

        total_detected = detected_counts.get(class_name, 0)

        # Nesne bulunmadığında eksik nesne hatası zaten eklenmiştir.
        if total_detected == 0:
            continue

        if correct_position_count < expected_count:
            errors.append(
                f"Konum hatası — {class_name}: "
                f"{required_position} bölgede bulunmalı"
            )

    is_pass = len(errors) == 0

    return is_pass, errors, position_counts


def create_expected_result_table(
    expected_objects: dict[str, int],
    detected_counts: Counter,
) -> pd.DataFrame:
    """Beklenen nesne ve adet kontrol tablosunu oluşturur."""

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
    """Konum doğrulama sonuçlarını tablo hâline getirir."""

    rows = []

    for class_name, required_position in position_rules.items():
        expected_count = expected_objects.get(class_name, 0)
        correct_position_count = position_counts.get(class_name, 0)
        total_detected = detected_counts.get(class_name, 0)

        if required_position == "Herhangi":
            status = "Kontrol Dışı"

        elif total_detected == 0:
            status = "Nesne Bulunamadı"

        elif correct_position_count >= expected_count:
            status = "Uygun"

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
    """İzin verilmeyen nesne kontrol tablosunu oluşturur."""

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