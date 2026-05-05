import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights


class TransReIDEncoder:
    """This class extracts L2-normalized visual features from bounding box crops using a ViT-B/16 backbone, optionally fine-tuned for person Re-ID.
    Parameters:
    checkpoint_path : Path to a TransReID .pth checkpoint (optional). If None, uses ImageNet-pretrained ViT-B/16 without fine-tuning.
    device          : "cuda" or "cpu" for model inference. Default is auto-detect.
    Attributes:
    INPUT_SIZE : (w, h) tuple for resizing crops (224x224 for ViT-B/16)
    EMBED_DIM  : dimensionality of the output feature vector (768 for ViT-B
    MEAN, STD  : normalization parameters for ImageNet pre-trained models
    model       : ViT-B/16 backbone with final head removed (outputs CLS token features
    transform   : torchvision transforms for preprocessing crops
    __call__     : method to process a frame and bounding boxes, returning features
    """

    INPUT_SIZE = (224, 224)   # ViT-B/16 expects exactly 224x224
    EMBED_DIM  = 768          # ViT-B/16 output dim

    # ImageNet normalization (same stats used in TransReID training)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # ── Build backbone ────────────────────────────────────────────────
        # ViT-B/16 pretrained on ImageNet-21k → fine-tune head for Re-ID.
        # Replace the classification head with an identity projection so
        # we get raw patch-averaged embeddings (CLS token).
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        backbone = vit_b_16(weights=weights)

        # Strip the final MLP head; we want the CLS token features directly
        backbone.heads = nn.Identity()
        self.model = backbone
        self.model.eval()
        self.model.to(self.device)

        # ── Load TransReID checkpoint (optional) ─────────────────────────
        # Official checkpoints from:
        #   https://github.com/damo-cv/TransReID
        # Expected key format: model state_dict with 'base.' prefix stripped
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # ── Preprocessing pipeline ────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                     # numpy → PIL
            transforms.Resize(self.INPUT_SIZE),          # 224x224 — ViT-B/16 hard requirement
            transforms.ToTensor(),                       # PIL → [0,1] float tensor
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    # ─────────────────────────────────────────────────────────────────────
    def _load_checkpoint(self, path: str):
        """Load a TransReID .pth checkpoint (best-effort key remapping)."""
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('model', ckpt)          # handle wrapped checkpoints

        # Remove common prefixes used in TransReID repo
        cleaned = {}
        for k, v in state.items():
            for prefix in ('base.', 'module.', 'backbone.'):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(f"[TransReID] Checkpoint loaded. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # ─────────────────────────────────────────────────────────────────────
    def _crop(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Crop and validate a single bounding box from a BGR frame."""
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

        # BGR → RGB for torchvision
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # ─────────────────────────────────────────────────────────────────────
    def __call__(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract Re-ID features for a batch of bounding boxes.

        Args:
            frame  : BGR uint8 numpy array (H, W, 3)
            bboxes : numpy array of shape (N, 4), each row [x, y, w, h]

        Returns:
            features : numpy array of shape (N, EMBED_DIM), L2-normalized.
                       Rows for invalid/empty crops are zero-vectors.
        """
        crops, valid_idx = [], []

        for i, box in enumerate(bboxes):
            rgb = self._crop(frame, box)
            if rgb is None:
                continue
            tensor = self.transform(rgb)          # (3, 256, 128)
            crops.append(tensor)
            valid_idx.append(i)

        output = np.zeros((len(bboxes), self.EMBED_DIM), dtype=np.float32)

        if not crops:
            return output

        batch = torch.stack(crops).to(self.device)   # (B, 3, 256, 128)

        with torch.no_grad():
            feats = self.model(batch).cpu().numpy()  # (B, 768)

        # L2 normalization — cosine similarity in downstream matching
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-6, None)

        for out_idx, feat in zip(valid_idx, feats):
            output[out_idx] = feat

        return output