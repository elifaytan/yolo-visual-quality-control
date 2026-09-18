from ultralytics import YOLO

# Önceden eğitilmiş YOLO11 nano modeli
model = YOLO("yolo11n.pt")

# Modeli eğit
model.train(
    data="dataset/ceramic/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs",
    name="ceramic_model"
)