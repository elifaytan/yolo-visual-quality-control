from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/detect/runs/ceramic_model/weights/best.pt"
)

TEST_IMAGES_PATH = Path(
    "dataset/ceramic/test/images"
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH.resolve()}"
        )

    if not TEST_IMAGES_PATH.exists():
        raise FileNotFoundError(
            f"Test görselleri bulunamadı: {TEST_IMAGES_PATH.resolve()}"
        )

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(TEST_IMAGES_PATH),
        conf=0.25,
        imgsz=640,
        save=True,
        project="runs/ceramic_tests",
        name="first_test",
        exist_ok=True
    )

    print("\nTest tamamlandı.")
    print("Sonuç klasörü: runs/ceramic_tests/first_test")


if __name__ == "__main__":
    main()