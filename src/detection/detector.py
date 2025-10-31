from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = ["car"]

    def detect(self, frame):
        # Resultats d’inferència
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]

        detections = []
        for box in results.boxes:
            cls_name = self.model.names[int(box.cls)]
            if cls_name in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2 - x1, y2 - y1])

        return detections
