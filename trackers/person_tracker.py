import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from models.transreid_encoder import TransReIDEncoder


class PersonTracker:
    """This class implements person tracking using
    Parameters:
    max_cosine_distance : Cosine distance threshold for matching. TransReID's tight clusters mean 0.3 is a safe default. If you see ID switches, try lowering to 0.2.
    nn_budget           : Maximum number of features to store per track. None for unlimited.
    max_age             : Maximum number of frames to keep a track without updates. Default is 30.
    checkpoint_path     : Path to TransReID .pth checkpoint (optional). None → uses ImageNet ViT-B/16 only. For better Re-ID:
    Attributes:
    tracker : Deep SORT tracker instance
    encoder : TransReIDEncoder instance for feature extraction
    _histories : dictionary mapping track IDs to their history of bounding boxes for trail visualization
    tracks : list of active Track objects representing the current state of tracked persons
    """
    def __init__(
        self,
        max_cosine_distance: float = 0.3,
        nn_budget: int | None = None,
        max_age: int = 30,
        checkpoint_path: str = None,
    ):
        if checkpoint_path is None:
            print(
                "[PersonTracker] Warning: TransReID checkpoint not provided.\n"
                "  Using ImageNet-pretrained ViT-B/16.\n"
                "  For better Re-ID: https://github.com/damo-cv/TransReID\n"
                "  Provide the checkpoint path as a parameter."
            )

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            max_cosine_distance,
            nn_budget,
        )
        self.tracker = DeepSortTracker(metric, max_age=max_age)
        self.encoder = TransReIDEncoder(checkpoint_path=checkpoint_path)

        # to save track history: track_id → history list
        self._histories: dict[int, list] = {}

        self.tracks: list[Track] = []

    # ──────────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list[list]) -> None:
        """Update the tracker with new detections for the current frame, extracting features and maintaining track histories.
        Parameters:
        frame      : BGR uint8 numpy array
        detections : [[x1, y1, x2, y2, score], ...]  (xyxy + score)
        Returns:     None (updates internal state)
        """
        self.tracker.predict()

        if len(detections) == 0:
            self.tracker.update([])
            self._sync_tracks()
            return

        bboxes_xyxy = np.array([d[:4] for d in detections], dtype=np.float32)

        # xyxy → xywh  
        bboxes_xywh = bboxes_xyxy.copy()
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]   # w = x2 - x1
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]   # h = y2 - y1

        scores = [d[4] for d in detections]

        # TransReID encoder xywh 
        features = self.encoder(frame, bboxes_xywh)   # (N, 768)

        dets = [
            Detection(bbox_xywh, score, feat)
            for bbox_xywh, score, feat in zip(bboxes_xywh, scores, features)
        ]

        self.tracker.update(dets)
        self._sync_tracks()

    # ──────────────────────────────────────────────────────────────────────
    def _sync_tracks(self) -> None:
        """Convert Deep SORT's internal tracks to public Track objects, maintaining history for trail visualization and cleaning up stale histories.
         Parameters: None
         Returns:     None (updates internal state)
         Note: This method should be called after tracker.update() to refresh the public track list."""
        active = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 10:
                continue

            bbox_xyxy = t.to_tlbr()   # [x1, y1, x2, y2]
            tid = t.track_id

            # update history for trail visualization
            if tid not in self._histories:
                self._histories[tid] = []
            self._histories[tid].append(list(bbox_xyxy))

            track = Track(tid, bbox_xyxy)
            track.history = self._histories[tid]
            active.append(track)

        # Stale history cleanup: remove histories of tracks that are no longer active to save memory
        active_ids = {t.track_id for t in active}
        for tid in [k for k in self._histories if k not in active_ids]:
            del self._histories[tid]

        self.tracks = active

# ──────────────────────────────────────────────────────────────────────────────
class Track:
    """A simple class to represent an active track with its ID, current bounding box, and history of bounding boxes for trail visualization.
    Parameters:
    track_id : Unique identifier for the track (assigned by Deep SORT)
    bbox     : Current bounding box in xyxy format [x1, y1, x2, y2]
    history  : List of past bounding boxes for this track (used for drawing trails)
    Attributes:
    track_id : Unique identifier for the track
    bbox     : Current bounding box in xyxy format [x1, y1, x2, y2]
    history  : List of past bounding boxes for this track (used for drawing trails)
    """
    __slots__ = ("track_id", "bbox", "history")

    def __init__(self, track_id: int, bbox):
        self.track_id = track_id
        self.bbox     = bbox
        self.history  = []