from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ReIDExtractor:
    """Lightweight appearance feature extractor using MobileNetV3-Small.

    Produces L2-normalized 576-dim embeddings from player crops.
    Uses torchvision (no extra dependencies) and runs efficiently
    on GPU with batch inference. Supports FP16 for ~2x speedup on GPU.
    """

    EMBED_DIM = 576

    def __init__(self, device: str = "cpu", fp16: bool = False):
        self.device = torch.device(device)
        self.fp16 = fp16 and device != "cpu"
        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        self.model = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        ).to(self.device).eval()

        if self.fp16:
            self.model = self.model.half()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Extract L2-normalized embeddings for each bbox crop.

        Args:
            frame: BGR/RGB image array (H, W, 3).
            bboxes: (N, 4+) array of [x1, y1, x2, y2, ...].

        Returns:
            (N, 576) float32 array of unit-norm embeddings.
        """
        n = len(bboxes)
        if n == 0:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)

        h_frame, w_frame = frame.shape[:2]
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            crops.append(self.transform(crop))

        batch = torch.stack(crops).to(self.device)
        if self.fp16:
            batch = batch.half()
        features = self.model(batch)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
