"""Wav2Vec2-based disfluency (stuttering) detection model — multi-label."""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


DISFLUENCY_CLASSES = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]


@dataclass
class DisfluencyEvent:
    """A detected disfluency event."""
    start_sec: float
    end_sec: float
    type: str
    confidence: float


class DisfluencyDetector(nn.Module):
    """Wav2Vec2-base encoder with classifier head for multi-label disfluency detection.

    5 classes (multi-label): Prolongation, Block, SoundRep, WordRep, Interjection.
    Classifier: Linear(768,256) → LayerNorm(256) → GELU → Dropout → Linear(256,5).
    """

    def __init__(self, num_classes: int = 5, freeze_layers: int = 3):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Freeze first N transformer layers + feature extractor
        self.encoder.feature_extractor._freeze_parameters()
        for i in range(freeze_layers):
            for param in self.encoder.encoder.layers[i].parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_values: Raw audio waveform tensor (batch, samples) at 16kHz.

        Returns:
            Logits tensor of shape (batch, num_classes). Use sigmoid for probabilities.
        """
        outputs = self.encoder(input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)


def load_disfluency_model(model_path: str, device: str = "cpu"):
    """Load a trained DisfluencyDetector from checkpoint.

    Args:
        model_path: Path to best_model.pt.
        device: Device to load onto.

    Returns:
        Tuple of (model, processor).
    """
    from transformers import Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    model = DisfluencyDetector()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Remap keys: checkpoint may use "wav2vec2.*" prefix from training,
    # but DisfluencyDetector wraps it as "encoder.*"
    state_dict = checkpoint["model_state_dict"]
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("wav2vec2."):
            remapped["encoder." + k[len("wav2vec2."):]] = v
        else:
            remapped[k] = v

    model.load_state_dict(remapped)
    model.to(device)
    model.eval()

    return model, processor


def predict_disfluency(model: DisfluencyDetector, processor, audio: np.ndarray,
                       sr: int = 16000, device: str = "cpu",
                       window_sec: float = 3.0, hop_sec: float = 1.5,
                       threshold: float = 0.5) -> list[DisfluencyEvent]:
    """Run multi-label disfluency detection on audio using a sliding window.

    Args:
        model: Trained DisfluencyDetector.
        processor: Wav2Vec2Processor.
        audio: Audio waveform (float32, 16kHz).
        sr: Sample rate.
        device: Inference device.
        window_sec: Window size in seconds.
        hop_sec: Hop size in seconds.
        threshold: Sigmoid probability threshold for each class.

    Returns:
        List of DisfluencyEvent objects (any class above threshold).
    """
    model.eval()
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    events = []
    pos = 0

    while pos + window_samples <= len(audio):
        chunk = audio[pos:pos + window_samples]
        start_sec = round(pos / sr, 3)
        end_sec = round((pos + window_samples) / sr, 3)

        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values)
            probs = torch.sigmoid(logits).squeeze(0)

        # Multi-label: report each class above threshold
        for i, cls_name in enumerate(DISFLUENCY_CLASSES):
            if probs[i].item() > threshold:
                events.append(DisfluencyEvent(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    type=cls_name,
                    confidence=round(probs[i].item(), 4),
                ))

        pos += hop_samples

    return events
