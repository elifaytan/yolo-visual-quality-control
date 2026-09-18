"""
Dalin şişesi için görsel kalite kontrol profilleri.

Model sınıfları:
- bottle
- cap
- front_label
- back_label
"""

POSITION_OPTIONS = [
    "Herhangi",
    "Sol",
    "Orta",
    "Sağ",
]


PRODUCT_PROFILES = {
    "Dalin Kontrolü": {
    "description": (
        "Dalin şişesinde şişe gövdesi, kapak, ön etiket "
        "ve arka etiket alanlarını görsel olarak kontrol eder."
    ),
    "standard_name": "Dalin Şişesi Görsel Uygunluk Kriteri",
    "model_path": "runs/detect/dalin_v2-2/weights/best.pt",
    "expected_objects": {
        "bottle": 1,
        "cap": 1,
    },
    "display_names": {
        "bottle": "Şişe Gövdesi",
        "cap": "Kapak",
        "front_label": "Ön Etiket",
        "back_label": "Arka Etiket ve Barkod Alanı",
    },
    "forbidden_objects": [],
    "position_rules": {
        "bottle": "Orta",
        "cap": "Orta",
    },
},
    
        "Seramik Yüzey Kusur Kontrolü": {
        "description": (
            "Seramik yüzey üzerinde kenar kırığı, yüzey boşluğu "
            "ve çizgisel kusur bulunup bulunmadığını kontrol eder."
        ),
        "standard_name": (
            "Seramik Yüzey Görsel Muayene Kriteri"
        ),
        "model_path": "ceramic/best.pt",
        "expected_objects": {},
        "display_names": {
            "edge-chipping": "Kenar Kırığı",
            "hole": "Yüzey Boşluğu / Delik",
            "line": "Çizgisel Kusur",
        },
        "forbidden_objects": [
            "edge-chipping",
            "hole",
            "line",
        ],
        "position_rules": {},
    },
}