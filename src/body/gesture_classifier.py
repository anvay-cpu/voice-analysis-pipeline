"""Step 5: Gesture type classification — 3-class with feature normalization."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# 3-class taxonomy (collapsed from original 5-class)
GESTURE_CLASSES = ["Active Gesture", "Adaptor", "Rest"]
NUM_CLASSES = len(GESTURE_CLASSES)

DEFAULT_MODEL_PATH = "models/gesture_transformer/best_model.pt"

# Sliding window config (must match training)
WINDOW_FRAMES = 15   # 3 seconds at 5fps
HOP_FRAMES = 8       # ~1.6 second hop
NUM_KEYPOINTS = 33

# Body landmark indices
L_WRIST, R_WRIST = 15, 16
L_SHOULDER, R_SHOULDER = 11, 12
NOSE = 0
L_HIP, R_HIP = 23, 24


# ═══════════════════ Feature Processing ═══════════════════

def _drop_z(kps_frame: np.ndarray) -> np.ndarray:
    """(33, 4) -> (33, 3): keep x, y, visibility; drop z."""
    return kps_frame[:, [0, 1, 3]]


def _normalize_to_body(kp: np.ndarray) -> np.ndarray:
    """Normalize single frame (33, 3) to body-relative coordinates."""
    kp = kp.copy()
    ls, rs = kp[L_SHOULDER, :2], kp[R_SHOULDER, :2]
    lh, rh = kp[L_HIP, :2], kp[R_HIP, :2]
    center = (ls + rs + lh + rh) / 4.0
    body_h = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2)
    if body_h < 0.01:
        return kp
    kp[:, 0] = (kp[:, 0] - center[0]) / body_h
    kp[:, 1] = (kp[:, 1] - center[1]) / body_h
    return kp


def process_window(kps_raw: np.ndarray) -> np.ndarray:
    """Full feature pipeline for a single window.

    Args:
        kps_raw: (15, 33, 4) raw keypoints.

    Returns:
        (15, 99) processed features: z dropped, body-normalized, smoothed.
    """
    frames = []
    for t in range(kps_raw.shape[0]):
        f = _drop_z(kps_raw[t])          # (33, 3)
        f = _normalize_to_body(f)         # (33, 3) body-relative
        frames.append(f.flatten())        # (99,)
    window = np.array(frames, dtype=np.float32)  # (15, 99)
    return uniform_filter1d(window, size=3, axis=0).astype(np.float32)


# ═══════════════════ Model Architectures ═══════════════════

class GestureTransformer(nn.Module):
    """Original large Transformer (~3.8M params)."""

    def __init__(self, input_dim: int = 99, num_classes: int = NUM_CLASSES):
        super().__init__()
        d = 256
        self.proj = nn.Linear(input_dim, d)
        self.pe = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 512, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 4)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.pe[:, :x.size(1)]
        x = self.norm(self.enc(x)).mean(dim=1)
        return self.head(x)


class TinyTransformer(nn.Module):
    """Small Transformer (~350K params)."""

    def __init__(self, input_dim: int = 99, num_classes: int = NUM_CLASSES):
        super().__init__()
        d = 128
        self.proj = nn.Linear(input_dim, d)
        self.pe = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 256, 0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.pe[:, :x.size(1)]
        x = self.enc(x).mean(dim=1)
        return self.head(x)


class GestureBiLSTM(nn.Module):
    """BiLSTM (~200K params)."""

    def __init__(self, input_dim: int = 99, hidden: int = 64,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))


# Map model names from checkpoint to classes
_MODEL_REGISTRY = {
    "GestureTransformer": GestureTransformer,
    "TinyTransformer": TinyTransformer,
    "BiLSTM": GestureBiLSTM,
}


# ═══════════════════ Classifier ═══════════════════

class GestureClassifier:
    """Classify gesture types from sequences of pose keypoints.

    Supports multiple model architectures (auto-detected from checkpoint)
    and falls back to an improved body-normalized heuristic when no model
    is available.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        window_frames: int = WINDOW_FRAMES,
        hop_frames: int = HOP_FRAMES,
        device: str = "cpu",
    ):
        self.window_frames = window_frames
        self.hop_frames = hop_frames
        self.device = device
        self.model = None
        self.input_dim = 99  # default: processed features
        self.feature_pipeline = "drop_z+body_norm+smooth"
        self.classes = list(GESTURE_CLASSES)

        if Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning(
                "Gesture model not found at %s — using heuristic fallback",
                model_path,
            )

    def _load_model(self, model_path: str):
        """Load trained model (auto-detects architecture from checkpoint)."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)

        model_name = ckpt.get("model_name", "GestureTransformer")
        self.input_dim = ckpt.get("input_dim", 99)
        num_classes = ckpt.get("num_classes", NUM_CLASSES)
        self.feature_pipeline = ckpt.get("feature_pipeline", "drop_z+body_norm+smooth")
        self.classes = ckpt.get("classes", list(GESTURE_CLASSES))

        model_cls = _MODEL_REGISTRY.get(model_name, GestureTransformer)
        self.model = model_cls(input_dim=self.input_dim, num_classes=num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Loaded gesture model: %s (input_dim=%d, classes=%d, val_f1=%.4f, pipeline=%s)",
            model_name, self.input_dim, num_classes,
            ckpt.get("val_f1", 0.0), self.feature_pipeline,
        )

    def create_windows(
        self, poses: dict[int, np.ndarray | None]
    ) -> list[dict]:
        """Create sliding windows with feature processing.

        Args:
            poses: Dict mapping frame_idx to (33, 4) keypoints or None.

        Returns:
            List of dicts with 'start_frame', 'end_frame', 'keypoints'.
        """
        indices = sorted(poses.keys())
        if not indices:
            return []

        max_idx = indices[-1]
        windows = []

        for start in range(indices[0], max_idx - self.window_frames + 2, self.hop_frames):
            end = start + self.window_frames
            raw_kps = []
            valid = 0

            for i in range(start, end):
                kp = poses.get(i)
                if kp is not None:
                    raw_kps.append(kp)  # (33, 4)
                    valid += 1
                else:
                    raw_kps.append(np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32))

            if valid >= 0.8 * self.window_frames:
                raw_array = np.array(raw_kps, dtype=np.float32)  # (15, 33, 4)

                if "drop_z" in self.feature_pipeline:
                    keypoints = process_window(raw_array)  # (15, 99)
                else:
                    keypoints = raw_array.reshape(self.window_frames, -1)  # (15, 132)

                windows.append({
                    "start_frame": start,
                    "end_frame": end - 1,
                    "keypoints": keypoints,
                    "_raw_kps": raw_array,  # kept for heuristic fallback
                })

        return windows

    def classify_windows(self, windows: list[dict]) -> list[dict]:
        """Classify gesture type for each window."""
        if not windows:
            return []

        if self.model is not None:
            return self._classify_model(windows)
        else:
            return self._classify_heuristic(windows)

    def _classify_model(self, windows: list[dict]) -> list[dict]:
        """Classify using the trained model."""
        batch = np.stack([w["keypoints"] for w in windows])
        tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        results = []
        for i, w in enumerate(windows):
            results.append({
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],
                "gesture_class": int(preds[i]),
                "gesture_label": self.classes[int(preds[i])],
                "confidence": float(probs[i, preds[i]]),
            })
        return results

    def _classify_heuristic(self, windows: list[dict]) -> list[dict]:
        """Improved heuristic fallback using body-normalized features."""
        results = []
        for w in windows:
            raw_kps = w.get("_raw_kps")
            if raw_kps is None:
                raw_kps = w["keypoints"].reshape(self.window_frames, NUM_KEYPOINTS, 4)
            label, conf = self._heuristic_3class(raw_kps)
            results.append({
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],
                "gesture_class": GESTURE_CLASSES.index(label),
                "gesture_label": label,
                "confidence": conf,
            })
        return results

    @staticmethod
    def _heuristic_3class(kps: np.ndarray) -> tuple[str, float]:
        """Body-normalized 3-class heuristic.

        Args:
            kps: (15, 33, 4) raw keypoints.

        Returns:
            (gesture_label, confidence) tuple.
        """
        # Body-normalize for camera-invariant thresholds
        norm_frames = []
        for t in range(kps.shape[0]):
            f = kps[t][:, [0, 1, 3]]  # drop z → (33, 3)
            f = _normalize_to_body(f)
            norm_frames.append(f)
        nkps = np.array(norm_frames)  # (15, 33, 3)

        l_wrist = nkps[:, L_WRIST, :2]
        r_wrist = nkps[:, R_WRIST, :2]
        nose = nkps[:, NOSE, :2]
        hip_mid = (nkps[:, L_HIP, :2] + nkps[:, R_HIP, :2]) / 2

        # Total wrist movement
        l_delta = np.sqrt(np.sum(np.diff(l_wrist, axis=0) ** 2, axis=1))
        r_delta = np.sqrt(np.sum(np.diff(r_wrist, axis=0) ** 2, axis=1))
        total_movement = np.sum(l_delta) + np.sum(r_delta)

        # Hand-to-face distance
        l_face_dist = np.mean(np.sqrt(np.sum((l_wrist - nose) ** 2, axis=1)))
        r_face_dist = np.mean(np.sqrt(np.sum((r_wrist - nose) ** 2, axis=1)))
        min_face_dist = min(l_face_dist, r_face_dist)

        # Hands below waist
        l_below = np.mean(l_wrist[:, 1] > hip_mid[:, 1])
        r_below = np.mean(r_wrist[:, 1] > hip_mid[:, 1])

        # Thresholds (tuned for body-normalized coordinates)
        if min_face_dist < 1.5 and total_movement < 3.0:
            return "Adaptor", 0.75

        if total_movement < 1.0 and max(l_below, r_below) > 0.6:
            return "Rest", 0.80

        if total_movement < 0.8:
            return "Rest", 0.70

        return "Active Gesture", 0.70

    def classify_video(
        self, poses: dict[int, np.ndarray | None], fps: int = 5
    ) -> list[dict]:
        """Full pipeline: create windows, classify, add timestamps.

        Args:
            poses: Dict mapping frame_idx to (33, 4) keypoints.
            fps: Video frame rate for timestamp conversion.

        Returns:
            List of gesture results with timestamps.
        """
        windows = self.create_windows(poses)
        results = self.classify_windows(windows)

        for r in results:
            r["start_sec"] = r["start_frame"] / fps
            r["end_sec"] = r["end_frame"] / fps

        logger.info(
            "Gesture classification: %d windows classified from %d frames",
            len(results),
            len(poses),
        )
        return results
