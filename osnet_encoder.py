import cv2
import numpy as np
import torch
import torchreid


class OSNetEncoder:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device

        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )

        self.model.eval()
        self.model.to(self.device)

    def __call__(self, frame, bboxes):
        crops = []
        valid_indices = []

        h_frame, w_frame = frame.shape[:2]

        for i, box in enumerate(bboxes):
            x, y, w, h = box.astype(int)

            # boundary fix
            x = max(0, x)
            y = max(0, y)
            w = max(0, min(w, w_frame - x))
            h = max(0, min(h, h_frame - y))

            crop = frame[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (128, 256))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            crop = (crop - mean) / std
            crop = np.transpose(crop, (2, 0, 1))

            crops.append(crop)
            valid_indices.append(i)

        if len(crops) == 0:
            return np.zeros((len(bboxes), 512), dtype=np.float32)

        batch = torch.tensor(np.array(crops), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            feats = self.model(batch).cpu().numpy()

        # normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-6, None)

        # map back
        output = np.zeros((len(bboxes), feats.shape[1]), dtype=np.float32)
        for idx, feat in zip(valid_indices, feats):
            output[idx] = feat

        return output