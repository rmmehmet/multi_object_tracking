"""
main.py

FIX-5 : person_colors ve car_colors ayrı dict → track_id çakışması yok.
FIX-4 : draw_trail xyxy formatını açıkça belgeliyor (CarTrack ve PersonTrack
         history'si artık xyxy → tutarlı).
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO
from trackers.person_tracker import PersonTracker
from trackers.car_tracker import CarTracker

# ── Dosya yolları ──────────────────────────────────────────────────────────────
video_path    = os.path.join('.', 'videos', '2954065-hd_1920_1080_30fps.mp4')
video_out_path = os.path.join('.', 'output', 'outv2.mp4')

# ── Video aç ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    raise RuntimeError(f"Video açılamadı: {video_path}")

cap_out = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

# ── Model & Tracker'lar ───────────────────────────────────────────────────────
model          = YOLO("model/yolo26n.pt")
person_tracker = PersonTracker(
    checkpoint_path="checkpoints/vit_transreid_market.pth"
)
car_tracker    = CarTracker()

# FIX-5: Ayrı renk dict'leri — person ve car track_id'leri çakışmaz
person_colors: dict[int, tuple] = {}
car_colors:    dict[int, tuple] = {}

THRESHOLD = 0.5

# Sabit renkler (opsiyonel: random da yapılabilir)
PERSON_COLOR = (0,   255,  0)    # yeşil
CAR_COLOR    = (255,  0,   0)    # mavi


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def get_color(colors: dict, tid: int, default: tuple) -> tuple:
    """Track ID'ye göre renk döndürür, yoksa default atar."""
    if tid not in colors:
        colors[tid] = default
    return colors[tid]


def draw_trail(frame: np.ndarray, history: list, color: tuple, max_len: int = 30) -> None:
    """
    Track geçmişini çizgi olarak çizer.

    Args:
        history : [[x1,y1,x2,y2], ...]  — xyxy formatı (FIX-4: tutarlı)
        max_len : kaç frame geriye bakılacağı
    """
    # En son pozisyondan geriye doğru çiz
    tail = history[-max_len:]   # son max_len frame

    for i in range(1, len(tail)):
        x1b, y1b, x2b, y2b = tail[i]
        x1a, y1a, x2a, y2a = tail[i - 1]

        cx_curr = int((x1b + x2b) / 2)
        cy_curr = int((y1b + y2b) / 2)
        cx_prev = int((x1a + x2a) / 2)
        cy_prev = int((y1a + y2a) / 2)

        # İz ilerledikçe solar
        alpha = i / len(tail)
        faded = tuple(int(c * alpha) for c in color)

        cv2.line(frame, (cx_prev, cy_prev), (cx_curr, cy_curr), faded, 2)


def draw_box(frame, bbox, label: str, color: tuple) -> None:
    """Bounding box + label çizer."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )


# ── Ana döngü ─────────────────────────────────────────────────────────────────
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

    # ── Tracker güncelle ───────────────────────────────────────────────────
    person_tracker.update(frame, person_dets)
    car_tracker.update(frame, car_dets)

    # ── Kişileri çiz ──────────────────────────────────────────────────────
    for t in person_tracker.tracks:
        color = get_color(person_colors, t.track_id, PERSON_COLOR)
        draw_box(frame, t.bbox, f"P-{t.track_id}", color)
        if len(t.history) > 1:
            draw_trail(frame, t.history, color)

    # ── Araçları çiz + iz ─────────────────────────────────────────────────
    for t in car_tracker.tracks:
        color = get_color(car_colors, t.track_id, CAR_COLOR)
        draw_box(frame, t.bbox, f"C-{t.track_id}", color)
        if len(t.history) > 1:
            draw_trail(frame, t.history, color)

    cap_out.write(frame)
    ret, frame = cap.read()

# ── Temizlik ──────────────────────────────────────────────────────────────────
cap.release()
cap_out.release()
cv2.destroyAllWindows()
print("Tamamlandı →", video_out_path)