from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
from models.osnet_encoder import OSNetEncoder


class PersonTracker:
    def __init__(self):
        max_cosine_distance = 0.3
        nn_budget = None

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            max_cosine_distance,
            nn_budget
        )

        self.tracker = DeepSortTracker(metric)
        self.encoder = OSNetEncoder()
        self.tracks = []

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for i, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[i], features[i]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []

        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue

            if track.time_since_update > 10:
                continue

            bbox = track.to_tlbr()
            tracks.append(Track(track.track_id, bbox))

        self.tracks = tracks


class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox