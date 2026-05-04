import os
import cv2
from ultralytics import YOLO
from trackers.person_tracker import PersonTracker
from trackers.car_tracker import CarTracker

video_path = os.path.join('.', 'videos', '2954065-hd_1920_1080_30fps.mp4')
video_out_path = os.path.join('.', 'output', 'output_trans_id.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

model = YOLO("model/yolo26n.pt")

person_tracker = PersonTracker()
car_tracker = CarTracker()

colors = {}
threshold = 0.5


def draw_trail(frame, track, color):
    # son 10 frame trail
    for i in range(1, len(track.history)):
        if i >= 10:
            break

        x1, y1, x2, y2 = track.history[-i]
        px1, py1, px2, py2 = track.history[-i-1]

        cx1 = int((x1 + x2) / 2)
        cy1 = int((y1 + y2) / 2)
        cx2 = int((px1 + px2) / 2)
        cy2 = int((py1 + py2) / 2)

        cv2.line(frame, (cx1, cy1), (cx2, cy2), color, 2)


while ret:

    results = model(frame)

    person_dets = []
    car_dets = []

    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            cls = int(cls)

            if score < threshold:
                continue

            # PERSON
            if cls == 0:
                person_dets.append([x1, y1, x2, y2, score])

            # CAR
            elif cls == 2:
                car_dets.append([x1, y1, x2, y2, score])

    # UPDATE
    person_tracker.update(frame, person_dets)
    car_tracker.update(frame, car_dets)

    # DRAW PERSON
    for t in person_tracker.tracks:
        x1, y1, x2, y2 = t.bbox
        tid = t.track_id

        if tid not in colors:
            colors[tid] = (0, 255, 0)

        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      colors[tid], 2)

        cv2.putText(frame, f"P-{tid}",
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors[tid], 2)

    # DRAW CAR + TRAIL
    for t in car_tracker.tracks:
        x1, y1, x2, y2 = t.bbox
        tid = t.track_id

        if tid not in colors:
            colors[tid] = (255, 0, 0)

        # rectangle
        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      colors[tid], 2)

        cv2.putText(frame, f"C-{tid}",
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors[tid], 2)

        # TRAIL EFFECT
        draw_trail(frame, t, colors[tid])

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()