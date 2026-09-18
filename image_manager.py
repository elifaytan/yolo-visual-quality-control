from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image


OUTPUT_ROOT = Path("outputs")
PASS_IMAGE_DIRECTORY = OUTPUT_ROOT / "pass"
FAIL_IMAGE_DIRECTORY = OUTPUT_ROOT / "fail"


def initialize_image_directories() -> None:
    """PASS ve FAIL görüntü klasörlerini oluşturur."""

    PASS_IMAGE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAIL_IMAGE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_image_filename(
    profile_name: str,
    result: str,
) -> str:
    """Tarih, profil ve sonuç bilgisini içeren benzersiz dosya adı üretir."""

    safe_profile_name = (
        profile_name
        .lower()
        .replace(" ", "_")
        .replace("ş", "s")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    unique_id = uuid4().hex[:8]

    return (
        f"{timestamp}_"
        f"{safe_profile_name}_"
        f"{result.lower()}_"
        f"{unique_id}.jpg"
    )


def save_annotated_image(
    image: np.ndarray,
    profile_name: str,
    is_pass: bool,
) -> Path:
    """
    YOLO kutularının çizildiği RGB görüntüyü diske kaydeder.

    PASS görüntüleri outputs/pass,
    FAIL görüntüleri outputs/fail klasöründe saklanır.
    """

    initialize_image_directories()

    result = "PASS" if is_pass else "FAIL"

    target_directory = (
        PASS_IMAGE_DIRECTORY
        if is_pass
        else FAIL_IMAGE_DIRECTORY
    )

    filename = create_image_filename(
        profile_name=profile_name,
        result=result,
    )

    file_path = target_directory / filename

    # Streamlit ve Pillow tarafında görüntüler RGB,
    # OpenCV kayıt sırasında BGR bekler.
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(file_path),
        image_bgr,
    )

    if not success:
        raise RuntimeError(
            "Analiz görüntüsü kaydedilemedi."
        )

    return file_path


def save_original_image(
    image: Image.Image,
    profile_name: str,
    is_pass: bool,
) -> Path:
    """Kameradan veya dosyadan alınan orijinal görüntüyü kaydeder."""

    initialize_image_directories()

    result = "PASS" if is_pass else "FAIL"

    target_directory = (
        PASS_IMAGE_DIRECTORY
        if is_pass
        else FAIL_IMAGE_DIRECTORY
    )

    filename = create_image_filename(
        profile_name=f"{profile_name}_original",
        result=result,
    )

    file_path = target_directory / filename

    image.convert("RGB").save(
        file_path,
        format="JPEG",
        quality=95,
    )

    return file_path