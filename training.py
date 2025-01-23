from ultralytics import YOLO

model = YOLO("yolov8s.pt")  


model.train(
    data="data.yaml",
    epochs=300,
    patience=30, 
    batch= 64,
    )
