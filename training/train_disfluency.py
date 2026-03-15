"""Training script for Wav2Vec2 disfluency detection model.

Usage (M1 Mac — slow, ~36 hours):
  python training/train_disfluency.py --platform m1 \
      --audio_dir data/SEP-28k/clips \
      --labels_csv data/SEP-28k/SEP-28k_episodes.csv

Usage (Colab — recommended, ~4 hours):
  Use training/colab_disfluency.ipynb instead.
"""

import argparse
import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.disfluency_model import DisfluencyDetector, DISFLUENCY_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


CLASS_WEIGHTS = [1.0, 6.5, 8.0, 12.0, 5.5]


class DisfluencyDataset(Dataset):
    """Dataset for SEP-28k stuttering events."""

    def __init__(self, samples: list, audio_dir: str, sr: int = 16000,
                 max_len_sec: float = 3.0, augment: bool = False):
        self.samples = samples
        self.audio_dir = audio_dir
        self.sr = sr
        self.max_len = int(max_len_sec * sr)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import librosa

        sample = self.samples[idx]
        audio_path = os.path.join(self.audio_dir, sample["file"])

        try:
            audio, _ = librosa.load(audio_path, sr=self.sr)
        except Exception:
            audio = np.zeros(self.max_len, dtype=np.float32)

        if self.augment:
            audio = self._augment(audio)

        # Pad or truncate to fixed length
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        elif len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)))

        return torch.tensor(audio, dtype=torch.float32), sample["label"]

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        import librosa

        # Speed perturbation (0.9x - 1.1x)
        if random.random() < 0.3:
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        # Additive noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.005, len(audio)).astype(np.float32)
            audio = audio + noise

        # Pitch shift
        if random.random() < 0.2:
            steps = random.uniform(-1, 1)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)

        return audio.astype(np.float32)


def parse_sep28k_csv(labels_csv: str) -> list[dict]:
    """Parse SEP-28k episodes CSV into sample dicts."""
    samples = []
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Determine label from annotation columns
            # SEP-28k has columns: Repetition, Prolongation, Block, Interjection, etc.
            label = 0  # Fluent by default
            max_votes = 0

            for cls_idx, cls_name in enumerate(DISFLUENCY_CLASSES[1:], 1):
                votes = int(row.get(cls_name, 0))
                if votes > max_votes:
                    max_votes = votes
                    label = cls_idx

            # Only use samples with clear majority (>=2 annotators agree)
            if max_votes < 2 and label != 0:
                continue

            # Build clip filename from Show, EpId, ClipId
            show = row.get("Show", "")
            ep_id = row.get("EpId", "")
            clip_id = row.get("ClipId", "")
            filename = f"{show}_{ep_id}_{clip_id}.wav"

            samples.append({"file": filename, "label": label})

    return samples


def train(args):
    """Main training loop."""
    device = "mps" if args.platform == "m1" else "cuda"
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    if not torch.backends.mps.is_available() and device == "mps":
        device = "cpu"

    logger.info(f"Device: {device}")

    # Parse dataset
    samples = parse_sep28k_csv(args.labels_csv)
    logger.info(f"Total samples: {len(samples)}")

    # Split 80/10/10
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    train_samples = samples[:int(0.8 * n)]
    val_samples = samples[int(0.8 * n):int(0.9 * n)]
    test_samples = samples[int(0.9 * n):]

    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_ds = DisfluencyDataset(train_samples, args.audio_dir, augment=True)
    val_ds = DisfluencyDataset(val_samples, args.audio_dir)
    test_ds = DisfluencyDataset(test_samples, args.audio_dir)

    num_workers = 0 if args.platform == "m1" else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=num_workers)

    # Model
    model = DisfluencyDetector()
    model.to(device)

    # Weighted loss
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Differential learning rates
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 2e-5},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-2)

    grad_accum = 4 if args.platform == "m1" else 2
    use_amp = device == "cuda"

    os.makedirs(args.model_dir, exist_ok=True)
    best_f1 = 0.0
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (audio, labels) in enumerate(train_loader):
            audio, labels = audio.to(device), labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(audio)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            train_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for audio, labels in val_loader:
                audio = audio.to(device)
                logits = model(audio)
                preds = logits.argmax(dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_true.extend(labels.tolist())

        macro_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        logger.info(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Macro F1: {macro_f1:.4f}")

        # Save checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "macro_f1": macro_f1,
        }, os.path.join(args.model_dir, f"checkpoint_epoch{epoch}.pt"))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "macro_f1": macro_f1,
            }, os.path.join(args.model_dir, "best_model.pt"))
            logger.info(f"  ✓ Saved best model (Macro F1={macro_f1:.4f})")

    # Test
    logger.info("\nTEST SET EVALUATION")
    checkpoint = torch.load(os.path.join(args.model_dir, "best_model.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for audio, labels in test_loader:
            audio = audio.to(device)
            logits = model(audio)
            test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            test_true.extend(labels.tolist())

    test_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    logger.info(f"Test Macro F1: {test_f1:.4f}")
    logger.info(classification_report(test_true, test_preds,
                                      target_names=DISFLUENCY_CLASSES, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["m1", "colab"], default="m1")
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dir", default="models/disfluency")
    args = parser.parse_args()
    train(args)
