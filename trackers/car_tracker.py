import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection


# ──────────────────────────────────────────────────────────────────────────────
# Vehicle Re-ID Encoder  
# ──────────────────────────────────────────────────────────────────────────────
class CarEncoder:
    """This class extracts L2-normalized visual features from bounding box crops.
    Parameters:
    device : "cuda" or "cpu" for model inference
    Attributes:
    INPUT_SIZE : (w, h) tuple for resizing crops (standard for vehicle Re-ID
    EMBED_DIM  : dimensionality of the output feature vector (512 for ResNet18)
    MEAN, STD  : normalization parameters for ImageNet pre-trained models
    model       : ResNet18 backbone with final FC layer removed (512-dim output)
    transform   : torchvision transforms for preprocessing crops
    __call__     : method to process a frame and bounding boxes, returning features
    """

    INPUT_SIZE = (128, 256)   # common size for vehicle Re-ID (w, h)
    EMBED_DIM  = 512

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, device: str = None):
        """Initialize the CarEncoder with a ResNet18 backbone and set up the device and transforms."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.model = backbone
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def _crop(self, frame: np.ndarray, box: np.ndarray):
        """Crop the image patch defined by the bounding box and convert it to RGB."""
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = box.astype(int)
        x = max(0, x);  y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        if w <= 0 or h <= 0:
            return None
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def __call__(self, frame: np.ndarray, bboxes_xywh: np.ndarray) -> np.ndarray:
        """Extract L2-normalized features for each bounding box in the frame."""
        crops, valid_idx = [], []
        for i, box in enumerate(bboxes_xywh):
            rgb = self._crop(frame, box)
            if rgb is None:
                continue
            crops.append(self.transform(rgb))
            valid_idx.append(i)

        output = np.zeros((len(bboxes_xywh), self.EMBED_DIM), dtype=np.float32)
        if not crops:
            return output

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            feats = self.model(batch).cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-6, None)

        for out_idx, feat in zip(valid_idx, feats):
            output[out_idx] = feat

        return output


# ──────────────────────────────────────────────────────────────────────────────
# Public track object and utility functions
# ──────────────────────────────────────────────────────────────────────────────
class CarTrack:
    """Public track object for cars, containing track ID, current bounding box, and history of past bounding boxes.
    Parameters:
    track_id : unique identifier for the track
    bbox     : current bounding box in xyxy format (x1, y1, x2, y2)
    history  : list of past bounding boxes (xyxy format) for trail visualization
    Attributes:
    __slots__ : defines the attributes for memory efficiency
    __init__   : initializes the CarTrack with track_id, bbox, and starts history with the initial bbox
    """
    __slots__ = ("track_id", "bbox", "history")

    def __init__(self, track_id: int, bbox):
        """Initialize a CarTrack with a track ID and bounding box, and start the history with the initial bounding box."""
        self.track_id = track_id
        self.bbox     = bbox          # xyxy
        self.history  = [list(bbox)]  # xyxy listesi


# ──────────────────────────────────────────────────────────────────────────────
# CarTracker
# ──────────────────────────────────────────────────────────────────────────────
class CarTracker:
    """This class manages the tracking of cars using Deep SORT, including feature extraction with CarEncoder and maintaining track histories for visualization.
    Parameters:
    max_cosine_distance : threshold for feature matching in Deep SORT (default 0.4 for vehicles)
    nn_budget           : maximum number of features to store for each track (default None for unlimited
    max_age             : maximum number of frames to keep a track without updates (default 30)
    Attributes:
    tracker : Deep SORT tracker instance
    encoder : CarEncoder instance for feature extraction
    _histories : dictionary mapping track IDs to their history of bounding boxes for trail visualization
    tracks : list of active CarTrack objects representing the current state of tracked cars
    """
    def __init__(
        self,
        max_cosine_distance: float = 0.4,   # slightly more tolerant for vehicles
        nn_budget: int | None = None,
        max_age: int = 30,
    ):
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = DeepSortTracker(metric, max_age=max_age)
        self.encoder = CarEncoder()

        self._histories: dict[int, list] = {}

        self.tracks: list[CarTrack] = []

    # ──────────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list[list]) -> None:
        """Update the tracker with new detections for the current frame, extracting features and maintaining track histories.
        """
        self.tracker.predict()

        if len(detections) == 0:
            self.tracker.update([])
            self._sync_tracks()
            return

        bboxes_xyxy = np.array([d[:4] for d in detections], dtype=np.float32)

        # xyxy → xywh
        bboxes_xywh = bboxes_xyxy.copy()
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]   # w = x2-x1
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]   # h = y2-y1

        scores   = [d[4] for d in detections]
        features = self.encoder(frame, bboxes_xywh)   # (N, 512)

        dets = [
            Detection(bbox_xywh, score, feat)
            for bbox_xywh, score, feat in zip(bboxes_xywh, scores, features)
        ]
        self.tracker.update(dets)
        self._sync_tracks()

    # ──────────────────────────────────────────────────────────────────────
    def _sync_tracks(self) -> None:
        """Convert Deep SORT's internal tracks to public CarTrack objects, maintaining history for trail visualization and cleaning up stale histories."""
        active = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 10:
                continue

            bbox_xyxy = t.to_tlbr()   # [x1, y1, x2, y2]
            tid = t.track_id

            # Save history for trail visualization
            if tid not in self._histories:
                self._histories[tid] = []
            self._histories[tid].append(list(bbox_xyxy))

            track      = CarTrack(tid, bbox_xyxy)
            track.history = self._histories[tid]
            active.append(track)

        # Remove histories of tracks that are no longer active
        active_ids = {t.track_id for t in active}
        stale = [tid for tid in self._histories if tid not in active_ids]
        for tid in stale:
            del self._histories[tid]

        self.tracks = active