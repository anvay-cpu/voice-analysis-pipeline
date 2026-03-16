"""Step 5: Gesture type classification via Transformer encoder."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

GESTURE_CLASSES = ["Illustrator", "Emblem", "Beat", "Adaptor", "Rest"]
NUM_CLASSES = len(GESTURE_CLASSES)

# Default model path
DEFAULT_MODEL_PATH = "models/gesture_transformer/best_model.pt"

# Sliding window config (must match training)
WINDOW_FRAMES = 15   # 3 seconds at 5fps
HOP_FRAMES = 8       # ~1.6 second hop
KEYPOINT_DIM = 132    # 33 keypoints × 4 (x, y, z, visibility)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class GestureTransformer(nn.Module):
    """Transformer encoder for gesture classification from pose sequences.

    Input:  (batch, 15, 132)  — 15 frames × 132-dim flattened keypoints
    Output: (batch, 5)        — logits for 5 gesture classes
    """

    def __init__(
        self,
        input_dim: int = KEYPOINT_DIM,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, 132) pose keypoint sequences.

        Returns:
            (batch, num_classes) logits.
        """
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.pos_encoding(x)          # add positional encoding
        x = self.encoder(x)               # (B, T, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)                 # mean pool over time → (B, d_model)
        return self.classifier(x)          # (B, num_classes)


class GestureClassifier:
    """Classify gesture types from sequences of pose keypoints."""

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

        if Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning("Gesture model not found at %s — using heuristic fallback", model_path)

    def _load_model(self, model_path: str):
        """Load trained Transformer model."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = GestureTransformer()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Loaded gesture model (epoch %d, val_f1=%.4f)",
            ckpt.get("epoch", -1),
            ckpt.get("val_f1", 0.0),
        )

    def create_windows(
        self, poses: dict[int, np.ndarray | None]
    ) -> list[dict]:
        """Create sliding windows of pose keypoints.

        Args:
            poses: Dict mapping frame_idx to (33, 4) keypoints or None.

        Returns:
            List of dicts with keys: 'start_frame', 'end_frame', 'keypoints' (15, 132).
        """
        indices = sorted(poses.keys())
        if not indices:
            return []

        max_idx = indices[-1]
        windows = []

        for start in range(indices[0], max_idx - self.window_frames + 2, self.hop_frames):
            end = start + self.window_frames
            window_kps = []
            valid = 0

            for i in range(start, end):
                kp = poses.get(i)
                if kp is not None:
                    window_kps.append(kp.flatten())  # (132,)
                    valid += 1
                else:
                    window_kps.append(np.zeros(KEYPOINT_DIM, dtype=np.float32))

            # Require at least 80% valid frames
            if valid >= 0.8 * self.window_frames:
                windows.append({
                    "start_frame": start,
                    "end_frame": end - 1,
                    "keypoints": np.array(window_kps, dtype=np.float32),  # (15, 132)
                })

        return windows

    def classify_windows(self, windows: list[dict]) -> list[dict]:
        """Classify gesture type for each window.

        Args:
            windows: Output of create_windows().

        Returns:
            List of dicts with added 'gesture_class', 'gesture_label', 'confidence'.
        """
        if not windows:
            return []

        if self.model is not None:
            return self._classify_transformer(windows)
        else:
            return self._classify_heuristic(windows)

    def _classify_transformer(self, windows: list[dict]) -> list[dict]:
        """Classify using trained Transformer model."""
        batch = np.stack([w["keypoints"] for w in windows])  # (N, 15, 132)
        tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)  # (N, 5)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        results = []
        for i, w in enumerate(windows):
            results.append({
                **w,
                "gesture_class": int(preds[i]),
                "gesture_label": GESTURE_CLASSES[int(preds[i])],
                "confidence": float(probs[i, preds[i]]),
            })

        return results

    def _classify_heuristic(self, windows: list[dict]) -> list[dict]:
        """Heuristic fallback when no model is available."""
        results = []
        for w in windows:
            kps = w["keypoints"].reshape(self.window_frames, 33, 4)
            label, conf = self._heuristic_single(kps)
            results.append({
                **w,
                "gesture_class": GESTURE_CLASSES.index(label),
                "gesture_label": label,
                "confidence": conf,
            })
        return results

    @staticmethod
    def _heuristic_single(kps: np.ndarray) -> tuple[str, float]:
        """Rule-based gesture classification for a single window.

        Args:
            kps: (15, 33, 4) keypoints for the window.

        Returns:
            (gesture_label, confidence) tuple.
        """
        # Key indices
        L_WRIST, R_WRIST = 15, 16
        L_SHOULDER, R_SHOULDER = 11, 12
        NOSE = 0
        L_EAR, R_EAR = 7, 8
        L_HIP, R_HIP = 23, 24

        # Extract wrist trajectories (x, y only)
        l_wrist = kps[:, L_WRIST, :2]
        r_wrist = kps[:, R_WRIST, :2]

        # Shoulder midpoint for reference
        shoulder_mid = (kps[:, L_SHOULDER, :2] + kps[:, R_SHOULDER, :2]) / 2
        hip_mid = (kps[:, L_HIP, :2] + kps[:, R_HIP, :2]) / 2

        # Movement magnitude (sum of frame-to-frame deltas)
        l_delta = np.sqrt(np.sum(np.diff(l_wrist, axis=0) ** 2, axis=1))
        r_delta = np.sqrt(np.sum(np.diff(r_wrist, axis=0) ** 2, axis=1))
        total_movement = np.sum(l_delta) + np.sum(r_delta)

        # Hands near face (adaptor check)
        nose = kps[:, NOSE, :2]
        l_ear = kps[:, L_EAR, :2]
        r_ear = kps[:, R_EAR, :2]
        face_center = nose
        l_face_dist = np.mean(np.sqrt(np.sum((l_wrist - face_center) ** 2, axis=1)))
        r_face_dist = np.mean(np.sqrt(np.sum((r_wrist - face_center) ** 2, axis=1)))

        # Hands below waist (rest check)
        l_below = np.mean(l_wrist[:, 1] > hip_mid[:, 1])
        r_below = np.mean(r_wrist[:, 1] > hip_mid[:, 1])

        # Rhythmic/beat detection (autocorrelation of movement signal)
        combined = l_delta + r_delta
        if len(combined) > 3 and np.std(combined) > 0:
            normed = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)
            autocorr = np.correlate(normed, normed, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            if len(autocorr) > 3:
                autocorr = autocorr / (autocorr[0] + 1e-8)
                beat_score = np.max(autocorr[2:min(7, len(autocorr))])
            else:
                beat_score = 0.0
        else:
            beat_score = 0.0

        # Classification rules
        face_threshold = 0.08
        rest_movement_threshold = 0.15
        beat_threshold = 0.4

        # Adaptor: hands near face
        if l_face_dist < face_threshold or r_face_dist < face_threshold:
            return "Adaptor", 0.7

        # Rest: low movement + hands below waist
        if total_movement < rest_movement_threshold and (l_below > 0.7 or r_below > 0.7):
            return "Rest", 0.8

        # Beat: rhythmic movement
        if beat_score > beat_threshold and total_movement > rest_movement_threshold:
            return "Beat", 0.6

        # Illustrator vs Emblem (most common active gesture)
        if total_movement > rest_movement_threshold:
            return "Illustrator", 0.5

        return "Rest", 0.6

    def classify_video(
        self, poses: dict[int, np.ndarray | None], fps: int = 5
    ) -> list[dict]:
        """Full pipeline: create windows and classify.

        Args:
            poses: Dict mapping frame_idx to (33, 4) keypoints.
            fps: Video frame rate for timestamp conversion.

        Returns:
            List of gesture results with timestamps.
        """
        windows = self.create_windows(poses)
        results = self.classify_windows(windows)

        # Add timestamps
        for r in results:
            r["start_sec"] = r["start_frame"] / fps
            r["end_sec"] = r["end_frame"] / fps

        logger.info(
            "Gesture classification: %d windows classified from %d frames",
            len(results),
            len(poses),
        )

        return results
