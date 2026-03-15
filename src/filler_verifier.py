"""MLP-based filler word verification using Whisper encoder embeddings (Stage 2)."""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VerifiedFiller:
    """A filler candidate that passed acoustic verification."""
    word: str
    start_sec: float
    end_sec: float
    confidence: float
    category: str
    verification_score: float


class FillerVerifierMLP(nn.Module):
    """Binary classifier: is this audio clip a filler word?

    Architecture: 768 → 256 → 64 → 1 with BatchNorm + Dropout.
    Input: Mean-pooled Whisper encoder embedding (768-dim).
    Output: Logit for P(filler).
    """

    def __init__(self, input_dim: int = 768, small: bool = False):
        super().__init__()
        if small:
            # Lightweight arch for small datasets (<200 samples)
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_filler_verifier(model_path: str, device: str = "cpu") -> FillerVerifierMLP:
    """Load a trained FillerVerifierMLP from checkpoint.

    Args:
        model_path: Path to best_model.pt.
        device: Device to load onto.

    Returns:
        Model in eval mode.
    """
    model = FillerVerifierMLP()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _extract_clip_embedding(audio_path: str, start_sec: float, end_sec: float,
                            whisper_model, whisper_processor,
                            sr: int = 16000, device: str = "cpu") -> torch.Tensor:
    """Extract Whisper encoder embedding for an audio clip.

    Loads the clip with context padding, runs through Whisper encoder,
    and returns mean-pooled hidden states (768-dim).
    """
    # Load audio with 0.5s padding on each side
    pad = 0.5
    offset = max(0, start_sec - pad)
    duration = (end_sec - start_sec) + 2 * pad
    audio, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=duration)

    if len(audio) == 0:
        return torch.zeros(768)

    # Process through Whisper encoder
    inputs = whisper_processor(
        audio, sampling_rate=sr, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        encoder_output = whisper_model.get_encoder()(input_features)
        # Mean-pool over time dimension → (768,)
        embedding = encoder_output.last_hidden_state.mean(dim=1).squeeze(0)

    return embedding.cpu()


def verify_fillers(model: FillerVerifierMLP, candidates: list,
                   whisper_model, whisper_processor,
                   threshold: float = 0.5,
                   device: str = "cpu",
                   audio_path: str = None) -> list[VerifiedFiller]:
    """Verify filler candidates using the acoustic MLP.

    Candidates with P(filler) > threshold pass verification.

    Args:
        model: Trained FillerVerifierMLP.
        candidates: List of FillerDetection objects from Stage 1.
        whisper_model: Whisper model (for encoder embeddings).
        whisper_processor: Whisper feature extractor/processor.
        threshold: Probability threshold for accepting a filler.
        device: Device for inference.
        audio_path: Path to the audio file (used for all candidates).

    Returns:
        List of VerifiedFiller objects that passed verification.
    """
    if not candidates:
        return []

    model.eval()
    verified = []

    for candidate in candidates:
        clip_audio_path = audio_path or getattr(candidate, "audio_path", None)
        if clip_audio_path is None:
            continue

        embedding = _extract_clip_embedding(
            clip_audio_path, candidate.start_sec, candidate.end_sec,
            whisper_model, whisper_processor, device=device,
        )

        with torch.no_grad():
            logit = model(embedding.unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()

        if prob > threshold:
            verified.append(VerifiedFiller(
                word=candidate.word,
                start_sec=candidate.start_sec,
                end_sec=candidate.end_sec,
                confidence=candidate.confidence,
                category=candidate.category,
                verification_score=round(prob, 4),
            ))

    return verified
