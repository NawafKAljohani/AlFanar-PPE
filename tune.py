from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # Load the pre-trained model


model.tune(
    data="data.yaml",
    iterations=300 ,
    epochs=30,
    patience=30, 
    batch= 60,
    )
