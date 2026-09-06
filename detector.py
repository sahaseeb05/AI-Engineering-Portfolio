import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Step 1: Internet se YOLOv8 Nano model download/load hota hai
        self.model = YOLO(model_path)

    def process_frame(self, frame, confidence=0.5):
        # Step 2: Image ko model ke paas bheja jata hai
        # conf parameter bata-ta hai ke kitni surety par object show karna hai
        results = self.model(frame, conf=confidence)[0]
        
        # Step 3: Pata lagaya jata hai ke kitni cheezein mili
        detected_classes = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            detected_classes.append(class_name)

        # Step 4: Box aur text wali final image tayar hoti hai
        annotated_frame = results.plot()
        
        return annotated_frame, detected_classes