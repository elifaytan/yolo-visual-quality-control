from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    """YOLO tarafından tespit edilen bir nesnenin bilgileri."""

    class_name: str
    confidence: float
    center_x: float
    center_y: float


def load_yolo_model(model_path: str) -> YOLO:
    """Belirtilen dosya yolundaki YOLO modelini yükler."""

    return YOLO(model_path)


def get_horizontal_region(
    center_x: float,
    image_width: int,
) -> str:
    """
    Nesnenin yatay merkezine göre bulunduğu bölgeyi belirler.

    Görüntü üç eşit bölüme ayrılır:
    - Sol
    - Orta
    - Sağ
    """

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
    """YOLO sonucunu Detection nesnelerine dönüştürür."""

    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    class_ids = result.boxes.cls.tolist()
    confidences = result.boxes.conf.tolist()
    bounding_boxes = result.boxes.xyxy.tolist()

    for class_id, confidence, box in zip(
        class_ids,
        confidences,
        bounding_boxes,
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
    """Tespit edilen bütün nesneleri sınıflarına göre sayar."""

    return Counter(
        detection.class_name
        for detection in detections
    )


def draw_region_guides(
    image: np.ndarray,
) -> np.ndarray:
    """Sol, orta ve sağ bölgeleri görüntü üzerinde gösterir."""

    output = image.copy()

    height, width = output.shape[:2]

    first_x = width // 3
    second_x = (width * 2) // 3

    guide_color = (255, 165, 0)

    cv2.line(
        output,
        (first_x, 0),
        (first_x, height),
        guide_color,
        2,
    )

    cv2.line(
        output,
        (second_x, 0),
        (second_x, height),
        guide_color,
        2,
    )

    cv2.putText(
        output,
        "SOL",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        guide_color,
        2,
    )

    cv2.putText(
        output,
        "ORTA",
        (first_x + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        guide_color,
        2,
    )

    cv2.putText(
        output,
        "SAG",
        (second_x + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        guide_color,
        2,
    )

    return output


def detect_objects(
    image: Image.Image,
    model: YOLO,
    confidence: float,
    show_regions: bool = True,
) -> tuple[
    np.ndarray,
    list[Detection],
    Counter,
    int,
]:
    """
    Görüntü üzerinde YOLO tahmini gerçekleştirir.

    Döndürülen değerler:
    - kutular çizilmiş görüntü,
    - tespit ayrıntıları,
    - sınıf bazında nesne sayıları,
    - görüntü genişliği.
    """

    rgb_image = np.array(
        image.convert("RGB")
    )

    image_width = rgb_image.shape[1]

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

    annotated_bgr = result.plot()

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

    if show_regions:
        annotated_rgb = draw_region_guides(
            image=annotated_rgb,
    )
    return (
        annotated_rgb,
        detections,
        detected_counts,
        image_width,
    )