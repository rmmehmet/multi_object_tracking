"""
CarTracker — Deep SORT tabanlı araç takipçisi.

Önceki sürümde sadece IoU eşleştirmesi vardı → oklüzyon = ID switch.
Bu sürümde:
  - Deep SORT + görsel özellik (ResNet18 encoder) kullanılıyor
  - History xyxy formatında tutulmakta (draw_trail ile tutarlı)
  - Person tracker ile aynı mimaride → tutarlılık sağlandı
"""

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
# Araç Re-ID Encoder  (ResNet18 — araçlar için ViT overkill, bu yeterli)
# ──────────────────────────────────────────────────────────────────────────────
class CarEncoder:
    """
    Bounding box crop'larından L2-normalize edilmiş görsel özellik çıkarır.
    Input : BGR frame + bboxes (N x 4, xywh formatı)
    Output: feature matrix (N x 512), L2-normalized
    """

    INPUT_SIZE = (128, 256)   # w x h  — araç Re-ID standart boyutu
    EMBED_DIM  = 512

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # FC katmanını identity yap → 512-dim embedding al
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
        """xywh formatındaki box'u frame'den kırp (BGR→RGB)."""
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
# Public track objesi
# ──────────────────────────────────────────────────────────────────────────────
class CarTrack:
    """
    Dışarıya açılan track objesi.
    bbox  : [x1, y1, x2, y2]  (xyxy formatı)
    history: [[x1,y1,x2,y2], ...]  — draw_trail ile tutarlı (xyxy)
    """
    __slots__ = ("track_id", "bbox", "history")

    def __init__(self, track_id: int, bbox):
        self.track_id = track_id
        self.bbox     = bbox          # xyxy
        self.history  = [list(bbox)]  # xyxy listesi


# ──────────────────────────────────────────────────────────────────────────────
# CarTracker
# ──────────────────────────────────────────────────────────────────────────────
class CarTracker:
    """
    Deep SORT tabanlı araç takipçisi.

    FIX-1 : Artık IoU-only değil, görsel özellik (ResNet18) de kullanılıyor.
    FIX-4 : history xyxy formatında tutuluyor → draw_trail ile tutarlı.
    """

    def __init__(
        self,
        max_cosine_distance: float = 0.4,   # araçlar için biraz daha toleranslı
        nn_budget: int | None = None,
        max_age: int = 30,
    ):
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = DeepSortTracker(metric, max_age=max_age)
        self.encoder = CarEncoder()

        # Track geçmişini saklamak için: track_id → history listesi
        self._histories: dict[int, list] = {}

        self.tracks: list[CarTrack] = []

    # ──────────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list[list]) -> None:
        """
        Args:
            frame      : BGR uint8 numpy array
            detections : [[x1, y1, x2, y2, score], ...]  (xyxy + score)
        """
        self.tracker.predict()

        if len(detections) == 0:
            self.tracker.update([])
            self._sync_tracks()
            return

        bboxes_xyxy = np.array([d[:4] for d in detections], dtype=np.float32)

        # xyxy → xywh  (Deep SORT ve encoder xywh bekliyor)
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
        """Deep SORT iç track'lerini public CarTrack listesine çevirir."""
        active = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 10:
                continue

            bbox_xyxy = t.to_tlbr()   # [x1, y1, x2, y2]
            tid = t.track_id

            # History'yi koru
            if tid not in self._histories:
                self._histories[tid] = []
            self._histories[tid].append(list(bbox_xyxy))

            track      = CarTrack(tid, bbox_xyxy)
            track.history = self._histories[tid]
            active.append(track)

        # Artık aktif olmayan track'lerin history'sini temizle (bellek)
        active_ids = {t.track_id for t in active}
        stale = [tid for tid in self._histories if tid not in active_ids]
        for tid in stale:
            del self._histories[tid]

        self.tracks = active