import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

video_path = os.path.join('.', 'videos', '2954065-hd_1920_1080_30fps.mp4')
video_out_path = os.path.join('.', 'output', 'output.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

model = YOLO("model/yolo26n.pt")
tracker = Tracker()

colors = {}
detection_threshold = 0.5

while ret:
    results = model(frame)

    detections = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)

            # sadece person
            if class_id != 0:
                continue

            if score > detection_threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), score])

    tracker.update(frame, detections)

    for track in tracker.tracks:
        x1, y1, x2, y2 = track.bbox
        track_id = track.track_id

        if track_id not in colors:
            colors[track_id] = (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255)
            )

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            colors[track_id],
            3
        )

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()