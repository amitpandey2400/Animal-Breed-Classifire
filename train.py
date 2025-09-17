from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 nano model

results = model.train(data='Dataset/data.yaml', epochs=25, imgsz=640)
