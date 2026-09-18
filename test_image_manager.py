import numpy as np
from PIL import Image

from image_manager import (
    initialize_image_directories,
    save_annotated_image,
    save_original_image,
)


def main() -> None:
    initialize_image_directories()

    # Test için yapay, gri bir görüntü oluşturulur.
    test_array = np.full(
        shape=(480, 640, 3),
        fill_value=180,
        dtype=np.uint8,
    )

    test_pil_image = Image.fromarray(
        test_array
    )

    original_path = save_original_image(
        image=test_pil_image,
        profile_name="Test Ürünü",
        is_pass=False,
    )

    annotated_path = save_annotated_image(
        image=test_array,
        profile_name="Test Ürünü",
        is_pass=False,
    )

    print("Görüntü kayıt testi başarılı.")
    print(f"Orijinal görüntü: {original_path}")
    print(f"İşlenmiş görüntü: {annotated_path}")


if __name__ == "__main__":
    main()