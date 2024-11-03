from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("weights/yolov8x-worldv2.pt")

# Train the model on the COCO8 dataset for 100 epochs
results = model.train(data="cfg/traffic.yaml", epochs=1, imgsz=640)