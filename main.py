import os
import numpy as np
import cv2
from ultralytics import YOLO
from trackers.person_tracker import PersonTracker
from trackers.car_tracker import CarTracker

# ── folder paths ──────────────────────────────────────────────────────────────
video_path    = os.path.join('.', 'videos', 'urban-traffic-video.mp4')
video_out_path = os.path.join('.', 'output', 'urban-traffic.mp4')

# ──opening video ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    raise RuntimeError(f"Video not found: {video_path}")

cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

# ── model and trackers ───────────────────────────────────────────────────────
model          = YOLO("model/yolov8s.pt")
person_tracker = PersonTracker(
    checkpoint_path="checkpoints/vit_transreid_market.pth"
)
car_tracker    = CarTracker()

# ── color definitions for tracks ─────────────────────────────────────────────
person_colors: dict[int, tuple] = {}
car_colors:    dict[int, tuple] = {}

THRESHOLD = 0.5

# const colors
PERSON_COLOR = (0,   255,  0)    # green
CAR_COLOR    = (255,  0,   0)    # blue


# ── helper functions ─────────────────────────────────────────────────────
def get_color(colors: dict, tid: int, default: tuple) -> tuple:
    """Returns the color for a given track ID, assigning a default if not already set.
    Parameters:
        colors  : track_id → color dict
        tid     : track ID
        default : default color to assign if tid not in colors
    Returns:
        Color tuple for the given track ID.
    """
    if tid not in colors:
        colors[tid] = default
    return colors[tid]

def draw_trail(frame: np.ndarray, history: list, color: tuple, max_len: int = 30) -> None:
    """Track traces its history as a line.
    Parameters:
        frame   : image to draw on
        history : list of past bounding boxes (xyxy format)
        color   : line color
        max_len : maximum length of the trail (number of past frames)
    Returns:     None (draws directly on the frame)
    """
    # draw backwards from the last position
    tail = history[-max_len:]   # last max_len frames

    for i in range(1, len(tail)):
        x1b, y1b, x2b, y2b = tail[i]
        x1a, y1a, x2a, y2a = tail[i - 1]

        cx_curr = int((x1b + x2b) / 2)
        cy_curr = int((y1b + y2b) / 2)
        cx_prev = int((x1a + x2a) / 2)
        cy_prev = int((y1a + y2a) / 2)

        # the trail fades as it progresses (older positions are more transparent)
        alpha = i / len(tail)
        faded = tuple(int(c * alpha) for c in color)

        cv2.line(frame, (cx_prev, cy_prev), (cx_curr, cy_curr), faded, 2)


def draw_box(frame, bbox, label: str, color: tuple) -> None:
    """Draw bounding box and label.
    Parameters:
        frame : image to draw on
        bbox  : bounding box in xyxy format (x1, y1, x2, y2)
        label : text label to display above the box
        color : color for the box and text
    Returns:     None (draws directly on the frame)"""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

# ── Main loop ─────────────────────────────────────────────────────────────────
while ret:
    results = model(frame)

    person_dets: list[list] = []
    car_dets:    list[list] = []

    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            cls = int(cls)

            if score < THRESHOLD:
                continue

            if cls == 0:                          # person
                person_dets.append([x1, y1, x2, y2, score])
            elif cls == 2:                        # car
                car_dets.append([x1, y1, x2, y2, score])

    # ── Tracker update ───────────────────────────────────────────────────
    person_tracker.update(frame, person_dets)
    car_tracker.update(frame, car_dets)

    # ── Draw persons ──────────────────────────────────────────────────────
    for t in person_tracker.tracks:
        color = get_color(person_colors, t.track_id, PERSON_COLOR)
        draw_box(frame, t.bbox, f"P-{t.track_id}", color)
        if len(t.history) > 1:
            draw_trail(frame, t.history, color)

    # ── Draw cars with trails ─────────────────────────────────────────────────────────
    for t in car_tracker.tracks:
        color = get_color(car_colors, t.track_id, CAR_COLOR)
        draw_box(frame, t.bbox, f"C-{t.track_id}", color)
        if len(t.history) > 1:
            draw_trail(frame, t.history, color)

    cap_out.write(frame)
    ret, frame = cap.read()

# ── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cap_out.release()
cv2.destroyAllWindows()
print("Completed →", video_out_path)