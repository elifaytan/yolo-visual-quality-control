from datetime import datetime
from pathlib import Path

import cv2


DATASET_ROOT = Path("dataset/dalin_v2/raw")

CATEGORIES = {
    ord("1"): "front",
    ord("2"): "back",
    ord("3"): "no_cap",
}


def count_images(category: str) -> int:
    """Seçilen klasördeki mevcut fotoğraf sayısını döndürür."""

    category_path = DATASET_ROOT / category
    return len(list(category_path.glob("*.jpg")))


def save_frame(frame, category: str) -> Path:
    """Kamera karesini seçilen kategoriye kaydeder."""

    category_path = DATASET_ROOT / category
    category_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = category_path / f"{category}_{timestamp}.jpg"

    success = cv2.imwrite(str(file_path), frame)

    if not success:
        raise RuntimeError("Fotoğraf kaydedilemedi.")

    return file_path


def draw_information_panel(
    frame,
    selected_category: str,
    saved_count: int,
):
    """Kullanım bilgilerini kamera görüntüsüne yazar."""

    output = frame.copy()

    lines = [
        f"Secili kategori: {selected_category}",
        f"Kaydedilen fotograf: {saved_count}",
        "1: On yuz",
        "2: Arka yuz",
        "3: Kapaksiz",
        "SPACE: Fotograf kaydet",
        "Q veya ESC: Kapat",
    ]

    y_position = 30

    for line in lines:
        cv2.putText(
            output,
            line,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        y_position += 30

    return output


def main() -> None:
    for category in CATEGORIES.values():
        (DATASET_ROOT / category).mkdir(
            parents=True,
            exist_ok=True,
        )

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Kamera açılamadı. Başka bir uygulamanın "
            "kamerayı kullanmadığını kontrol edin."
        )

    selected_category = "front"
    window_name = "Veri Seti Fotograf Toplama"

    print("Fotoğraf toplama programı başlatıldı.")
    print("1: Ön yüz")
    print("2: Arka yüz")
    print("3: Kapaksız")
    print("SPACE: Fotoğraf kaydet")
    print("Q veya ESC: Programı kapat")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Kameradan görüntü alınamadı.")
                break

            current_count = count_images(
                selected_category
            )

            display_frame = draw_information_panel(
                frame=frame,
                selected_category=selected_category,
                saved_count=current_count,
            )

            cv2.imshow(
                window_name,
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in CATEGORIES:
                selected_category = CATEGORIES[key]

                print(
                    f"Kategori değiştirildi: "
                    f"{selected_category}"
                )

            elif key == 32:
                saved_path = save_frame(
                    frame=frame,
                    category=selected_category,
                )

                print(
                    f"Fotoğraf kaydedildi: "
                    f"{saved_path}"
                )

            elif key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Fotoğraf toplama programı kapatıldı.")


if __name__ == "__main__":
    main()