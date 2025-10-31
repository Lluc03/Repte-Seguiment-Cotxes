import cv2
import time
from .detection.detector import YoloDetector
from .tracking.tracker import CentroidTrackerHOG
from .tracking.trackerv1 import CentroidTrackerV1
from .counting.counter import VehicleCounter
from .video_io.video_reader import VideoReader
from .video_io.video_writer import VideoWriter

def run_app(video_source):
    detector = YoloDetector(model_path="models/yolov8n.pt", conf_threshold=0.5)
    tracker = CentroidTrackerHOG(
        max_disappeared=50,
        max_distance=250,
        direction_threshold=20,
        alpha=0.8,
        beta=0.2
    )
    # tracker = CentroidTrackerV1()
    reader = VideoReader(video_source)
    writer = VideoWriter("vehicle_counts.txt")
    counter = VehicleCounter(line_y=800)

    while True:
        ret, frame = reader.read()
        if not ret:
            break

        start_total = time.time()

        # --- Detector ---
        start_detector = time.time()
        detections = detector.detect(frame)
        end_detector = time.time()

        # --- Tracker ---
        start_tracker = time.time()
        objects = tracker.update(detections, frame)
        end_tracker = time.time()

        # --- Comptador ---
        start_counter = time.time()
        counts = counter.update_counts(objects)
        end_counter = time.time()

        end_total = time.time()

        # Calcula durades (ms)
        detector_time = (end_detector - start_detector) * 1000
        tracker_time = (end_tracker - start_tracker) * 1000
        counter_time = (end_counter - start_counter) * 1000
        total_time = (end_total - start_total) * 1000

        # print(f"Detector: {detector_time:.2f} ms | Tracker: {tracker_time:.2f} ms | "
        #       f"Counter: {counter_time:.2f} ms | Total: {total_time:.2f} ms")

        # Define el porcentaje del ancho que ocupará la línea
        line_fraction = 0.4  # 40% del ancho total

        # Calcula los extremos de la línea centrada
        frame_width = frame.shape[1]
        line_length = int(frame_width * line_fraction)
        x_start = (frame_width - line_length) // 2
        x_end = x_start + line_length

        # Dibuja la línea centrada
        cv2.line(frame, (x_start, counter.line_y), (x_end, counter.line_y), (0, 255, 255), 2)

        # dibuja bounding boxes y centroides
        for objectID, centroid in objects.items():
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {objectID}", (centroid[0]-10, centroid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"NORD: {counts['north']}  SUD: {counts['south']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Vehicle Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_detected = tracker.nextObjectID
    writer.write_final_counts(counts, total_detected)
    reader.release()
    cv2.destroyAllWindows()
