"""
PersonTracker — Deep SORT + TransReID

Replaces OSNet encoder with TransReID (ViT-B/16) for stronger
Re-ID in crowded / cross-camera scenarios.

Key change: OSNetEncoder → TransReIDEncoder
Everything else (Deep SORT metric, predict/update loop) stays the same.
"""

import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from models.transreid_encoder import TransReIDEncoder


class PersonTracker:
    def __init__(
        self,
        max_cosine_distance: float = 0.3,
        nn_budget: int | None = None,
        max_age: int = 30,
        checkpoint_path: str = None,
    ):
        """
        Args:
            max_cosine_distance : Cosine threshold for feature matching.
                                  TransReID produces tighter clusters than
                                  OSNet → 0.3 is a safe default; lower to
                                  0.2 if you see ID-switches.
            nn_budget           : Max stored features per identity (None = unlimited).
            max_age             : Frames to keep a lost track alive before deletion.
            checkpoint_path     : Optional path to TransReID .pth checkpoint.
        """
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            max_cosine_distance,
            nn_budget,
        )
        self.tracker = DeepSortTracker(metric, max_age=max_age)
        self.encoder = TransReIDEncoder(checkpoint_path=checkpoint_path)
        self.tracks: list[Track] = []

    # ─────────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list[list]) -> None:
        """
        Args:
            frame      : BGR uint8 numpy array
            detections : list of [x1, y1, x2, y2, score]
        """
        self.tracker.predict()

        if len(detections) == 0:
            self.tracker.update([])
            self._sync_tracks()
            return

        # Deep SORT expects xywh, not xyxy
        bboxes = np.array([d[:4] for d in detections], dtype=np.float32)
        bboxes_xywh = bboxes.copy()
        bboxes_xywh[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]   # w = x2-x1, h = y2-y1

        scores = [d[4] for d in detections]

        # ── TransReID feature extraction ──────────────────────────────────
        features = self.encoder(frame, bboxes_xywh)   # (N, 768)

        dets = [
            Detection(bbox_xywh, score, feat)
            for bbox_xywh, score, feat in zip(bboxes_xywh, scores, features)
        ]

        self.tracker.update(dets)
        self._sync_tracks()

    # ─────────────────────────────────────────────────────────────────────
    def _sync_tracks(self) -> None:
        """Convert Deep SORT internal tracks to our public Track objects."""
        self.tracks = [
            Track(t.track_id, t.to_tlbr())
            for t in self.tracker.tracks
            if t.is_confirmed() and t.time_since_update <= 10
        ]


# ─────────────────────────────────────────────────────────────────────────
class Track:
    """Lightweight public track object exposed to main.py."""
    __slots__ = ("track_id", "bbox")

    def __init__(self, track_id: int, bbox):
        self.track_id = track_id
        self.bbox = bbox          # [x1, y1, x2, y2]