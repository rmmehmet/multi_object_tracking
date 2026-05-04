class CarTrack:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.history = []   # trail için


class CarTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2-x1) * max(0, y2-y1)

        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])

        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, frame, detections):
        updated = []

        for det in detections:
            best = None
            best_iou = 0

            for t in self.tracks:
                i = self.iou(t.bbox, det[:4])
                if i > best_iou:
                    best_iou = i
                    best = t

            if best_iou > 0.4:
                best.bbox = det[:4]
                best.history.append(det[:4])
                updated.append(best)
            else:
                new = CarTrack(self.next_id, det[:4])
                new.history.append(det[:4])
                self.next_id += 1
                updated.append(new)

        self.tracks = updated