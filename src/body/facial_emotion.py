"""Step 8: Facial emotion recognition via EfficientNet-B0.

Fine-tuned on FER2013 (Kaggle). Classifies 48x48 grayscale face crops
(upsampled to 224x224 RGB) into 7 emotion classes.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# FER2013 uses 7 classes (no "contempt" in standard FER)
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTION_CLASSES)

DEFAULT_MODEL_PATH = "models/facial_emotion/best_model.pt"

# Preprocessing: match training pipeline
INFERENCE_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FacialEmotionNet(nn.Module):
    """EfficientNet-B0 with custom classification head for facial emotion."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        import torchvision.models as models

        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class FacialEmotionClassifier:
    """Classify facial emotions from face crops or full frames.

    Uses a face detector (MediaPipe or passed-in bboxes) to crop faces,
    then runs EfficientNet-B0 for emotion classification.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
    ):
        self.device = device
        self.model = None
        self.classes = list(EMOTION_CLASSES)

        if Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning(
                "Facial emotion model not found at %s — "
                "classify_frame will return Neutral fallback",
                model_path,
            )

    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)

        num_classes = ckpt.get("num_classes", NUM_CLASSES)
        self.classes = ckpt.get("classes", list(EMOTION_CLASSES))

        self.model = FacialEmotionNet(num_classes=num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Loaded facial emotion model (val_acc=%.4f, classes=%d)",
            ckpt.get("val_acc", 0.0), num_classes,
        )

    def classify_crop(self, face_crop: np.ndarray) -> dict:
        """Classify emotion from a face crop.

        Args:
            face_crop: BGR face image (any size).

        Returns:
            Dict with 'emotion', 'emotion_idx', 'confidence', 'probabilities'.
        """
        if self.model is None:
            return self._fallback()

        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # If grayscale-like (all channels same), keep as is
        # Transform expects (H, W, 3) uint8
        tensor = INFERENCE_TRANSFORM(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = probs.argmax().item()

        return {
            "emotion": self.classes[pred],
            "emotion_idx": pred,
            "confidence": float(probs[pred]),
            "probabilities": {
                c: float(probs[i]) for i, c in enumerate(self.classes)
            },
        }

    def classify_frame(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> dict:
        """Classify emotion from a full frame with optional face bbox.

        Args:
            frame: BGR image.
            face_bbox: Optional (x1, y1, x2, y2) face bounding box.

        Returns:
            Emotion classification dict.
        """
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            h, w = frame.shape[:2]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return self._fallback()
            return self.classify_crop(crop)

        # No bbox — use center crop as rough face estimate
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 3  # faces tend to be in upper-center
        size = min(h, w) // 3
        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(w, cx + size)
        y2 = min(h, cy + size)
        crop = frame[y1:y2, x1:x2]
        return self.classify_crop(crop)

    def track_video(
        self,
        frame_paths: list[str],
        face_bboxes: dict[int, tuple] | None = None,
    ) -> list[dict]:
        """Classify emotions across all frames.

        Args:
            frame_paths: Ordered list of frame image paths.
            face_bboxes: Optional dict mapping frame_idx to (x1,y1,x2,y2).

        Returns:
            List of per-frame emotion classification dicts.
        """
        results = []

        for idx, fpath in enumerate(frame_paths):
            frame = cv2.imread(fpath)
            if frame is None:
                logger.warning("Could not read frame: %s", fpath)
                result = self._fallback()
                result["frame_idx"] = idx
                results.append(result)
                continue

            bbox = face_bboxes.get(idx) if face_bboxes else None
            result = self.classify_frame(frame, face_bbox=bbox)
            result["frame_idx"] = idx
            results.append(result)

            if (idx + 1) % 100 == 0:
                logger.info(
                    "Facial emotion: %d / %d frames", idx + 1, len(frame_paths)
                )

        # Log summary
        from collections import Counter
        emotion_counts = Counter(r["emotion"] for r in results)
        logger.info(
            "Facial emotion complete: %d frames, distribution: %s",
            len(results), dict(emotion_counts),
        )

        return results

    def _fallback(self) -> dict:
        """Default response when no model or invalid input."""
        return {
            "emotion": "Neutral",
            "emotion_idx": self.classes.index("Neutral") if "Neutral" in self.classes else 0,
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in self.classes},
        }
