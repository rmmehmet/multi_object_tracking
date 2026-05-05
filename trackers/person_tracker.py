"""
PersonTracker — Deep SORT + TransReID

FIX-2 : bboxes_xywh dönüşümü açıkça belgelendi ve doğrulandı.
         Encoder xywh alır → _crop içinde x,y,w,h olarak kullanır. ✓
         Deep SORT Detection da xywh bekler. ✓

FIX-3 : checkpoint_path=None durumunda kullanıcı uyarılıyor.
         ImageNet-pretrained ViT ile çalışmaya devam eder ama
         TransReID fine-tune checkpoint varsa performans artar.
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
            max_cosine_distance : Cosine eşik. TransReID kümeleri sıkı olduğu
                                  için 0.3 güvenli default. ID switch görürsen
                                  0.2'ye indir.
            nn_budget           : Identity başına max saklanan özellik sayısı.
            max_age             : Kayıp track'in silinmeden önceki frame sayısı.
            checkpoint_path     : TransReID .pth checkpoint yolu (opsiyonel).
                                  None → sadece ImageNet ViT kullanılır.
                                  Checkpoint için: https://github.com/damo-cv/TransReID
        """
        # FIX-3: Checkpoint yoksa uyar, yine de çalış
        if checkpoint_path is None:
            print(
                "[PersonTracker] UYARI: TransReID checkpoint belirtilmedi.\n"
                "  ImageNet-pretrained ViT-B/16 kullanılıyor.\n"
                "  Daha iyi Re-ID için: https://github.com/damo-cv/TransReID\n"
                "  checkpoint_path parametresine .pth dosya yolunu ver."
            )

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            max_cosine_distance,
            nn_budget,
        )
        self.tracker = DeepSortTracker(metric, max_age=max_age)
        self.encoder = TransReIDEncoder(checkpoint_path=checkpoint_path)

        # History: track_id → [[x1,y1,x2,y2], ...]  (xyxy formatı)
        self._histories: dict[int, list] = {}

        self.tracks: list[Track] = []

    # ──────────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list[list]) -> None:
        """
        Args:
            frame      : BGR uint8 numpy array
            detections : [[x1, y1, x2, y2, score], ...]   (xyxy + score)

        Koordinat dönüşümü:
            Gelen format  : xyxy  (x1, y1, x2, y2)
            Deep SORT/Enc : xywh  (x1, y1, width, height)
            FIX-2: Bu dönüşüm aşağıda açıkça yapılıyor.
        """
        self.tracker.predict()

        if len(detections) == 0:
            self.tracker.update([])
            self._sync_tracks()
            return

        bboxes_xyxy = np.array([d[:4] for d in detections], dtype=np.float32)

        # xyxy → xywh  ← FIX-2: dönüşüm burada, net ve açık
        bboxes_xywh = bboxes_xyxy.copy()
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]   # w = x2 - x1
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]   # h = y2 - y1

        scores = [d[4] for d in detections]

        # TransReID encoder xywh alır → _crop içinde x,y,w,h olarak kullanır ✓
        features = self.encoder(frame, bboxes_xywh)   # (N, 768)

        dets = [
            Detection(bbox_xywh, score, feat)
            for bbox_xywh, score, feat in zip(bboxes_xywh, scores, features)
        ]

        self.tracker.update(dets)
        self._sync_tracks()

    # ──────────────────────────────────────────────────────────────────────
    def _sync_tracks(self) -> None:
        """Deep SORT iç track'lerini public Track listesine çevirir."""
        active = []
        for t in self.tracker.tracks:
            if not t.is_confirmed() or t.time_since_update > 10:
                continue

            bbox_xyxy = t.to_tlbr()   # [x1, y1, x2, y2]
            tid = t.track_id

            # History güncelle
            if tid not in self._histories:
                self._histories[tid] = []
            self._histories[tid].append(list(bbox_xyxy))

            track = Track(tid, bbox_xyxy)
            track.history = self._histories[tid]
            active.append(track)

        # Stale history temizle
        active_ids = {t.track_id for t in active}
        for tid in [k for k in self._histories if k not in active_ids]:
            del self._histories[tid]

        self.tracks = active


# ──────────────────────────────────────────────────────────────────────────────
class Track:
    """
    Dışarıya açılan track objesi.
    bbox    : [x1, y1, x2, y2]  (xyxy)
    history : [[x1,y1,x2,y2], ...]  — draw_trail ile tutarlı
    """
    __slots__ = ("track_id", "bbox", "history")

    def __init__(self, track_id: int, bbox):
        self.track_id = track_id
        self.bbox     = bbox
        self.history  = []