"""ECAPA-TDNN based vocal emotion classification."""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

# Patch torchaudio for SpeechBrain compatibility (torchaudio>=2.1 removed list_audio_backends)
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["default"]


EMOTION_CLASSES = ["Neutral", "Enthusiastic", "Nervous", "Angry", "Sad"]


@dataclass
class EmotionResult:
    """Single emotion prediction result."""
    start_sec: float
    end_sec: float
    label: str
    confidence: float
    distribution: dict[str, float]


class EmotionHead(nn.Module):
    """MLP classification head for emotion recognition.

    Takes 192-dim ECAPA-TDNN embeddings → 5 emotion classes.
    Architecture: 192 → 256 → 128 → 64 → 5
    """

    def __init__(self, input_dim: int = 192, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class VocalEmotionClassifier:
    """Vocal emotion classifier using ECAPA-TDNN encoder + MLP head.

    Usage:
        classifier = VocalEmotionClassifier("models/vocal_emotion/best_model.pt", device="mps")
        result = classifier.predict(audio_array, sr=16000)
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.classes = EMOTION_CLASSES

        # Load ECAPA-TDNN encoder from SpeechBrain
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )

        # Load trained emotion head
        self.head = EmotionHead(input_dim=192, num_classes=len(self.classes))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.head.load_state_dict(checkpoint["head_state_dict"])
        self.head.to(self.device)
        self.head.eval()

    def _extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Extract 192-dim ECAPA-TDNN embedding from audio."""
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        embedding = self.encoder.encode_batch(waveform)
        return embedding.squeeze(0).squeeze(0)  # (192,)

    def predict(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Predict emotion from audio.

        Args:
            audio: Audio signal as float32 numpy array.
            sr: Sample rate.

        Returns:
            Dict with label, confidence, distribution.
        """
        embedding = self._extract_embedding(audio, sr)
        embedding = embedding.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.head(embedding)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        idx = probs.argmax().item()
        distribution = {cls: round(probs[i].item(), 4) for i, cls in enumerate(self.classes)}

        return {
            "label": self.classes[idx],
            "confidence": round(probs[idx].item(), 4),
            "distribution": distribution,
        }

    def predict_windowed(self, audio: np.ndarray, sr: int = 16000,
                         window_sec: float = 2.0, hop_sec: float = 1.0) -> list[EmotionResult]:
        """Predict emotion in sliding windows over audio.

        Args:
            audio: Audio signal.
            sr: Sample rate.
            window_sec: Window size in seconds.
            hop_sec: Hop size in seconds.

        Returns:
            List of EmotionResult per window.
        """
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        results = []

        pos = 0
        while pos + window_samples <= len(audio):
            chunk = audio[pos:pos + window_samples]
            start_sec = round(pos / sr, 3)
            end_sec = round((pos + window_samples) / sr, 3)

            pred = self.predict(chunk, sr)
            results.append(EmotionResult(
                start_sec=start_sec,
                end_sec=end_sec,
                label=pred["label"],
                confidence=pred["confidence"],
                distribution=pred["distribution"],
            ))
            pos += hop_samples

        return results


def load_emotion_model(model_path: str, device: str = "cpu") -> VocalEmotionClassifier:
    """Load a trained vocal emotion classifier.

    Args:
        model_path: Path to best_model.pt checkpoint.
        device: Device string ("cpu", "mps", "cuda").

    Returns:
        VocalEmotionClassifier ready for inference.
    """
    return VocalEmotionClassifier(model_path, device)
