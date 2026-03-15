"""Training script for filler word verification MLP.

Usage:
  # Step 1: Pre-extract Whisper embeddings from labeled audio clips
  python training/train_filler_verifier.py --preextract \
      --data data/filler_labels.json \
      --embeddings_dir data/filler_embeddings \
      --device mps

  # Step 2: Train the MLP on pre-extracted embeddings
  python training/train_filler_verifier.py \
      --embeddings_dir data/filler_embeddings \
      --device mps --epochs 50 --batch_size 16
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.filler_verifier import FillerVerifierMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class FillerEmbeddingDataset(Dataset):
    """Dataset of pre-extracted Whisper encoder embeddings."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════
# PRE-EXTRACTION
# ═══════════════════════════════════════════════════════════════

def preextract_embeddings(data_path: str, embeddings_dir: str, device: str = "mps"):
    """Extract Whisper encoder embeddings for all labeled samples.

    Loads each audio clip, runs through Whisper's encoder, mean-pools
    the hidden states, and saves the 768-dim embedding.

    Args:
        data_path: Path to filler_labels.json.
        embeddings_dir: Directory to save embeddings and labels.
        device: Device for Whisper inference.
    """
    import librosa
    from transformers import WhisperModel, WhisperProcessor

    logger.info("Loading Whisper model for embedding extraction...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small")

    # Use CPU for extraction (MPS has issues with some Whisper ops)
    extract_device = "cpu"
    whisper_model = whisper_model.to(extract_device)
    whisper_model.eval()
    encoder = whisper_model.get_encoder()

    with open(data_path) as f:
        data = json.load(f)

    samples = data["samples"]
    logger.info(f"Processing {len(samples)} samples...")

    embeddings = []
    labels = []
    failed = 0

    for i, sample in enumerate(samples):
        audio_path = sample["audio_path"]

        # Resolve relative paths from project root
        if not os.path.isabs(audio_path):
            audio_path = str(Path(__file__).resolve().parent.parent / audio_path)

        start_sec = sample["start_sec"]
        end_sec = sample["end_sec"]
        is_filler = sample["is_filler"]

        try:
            # Load clip with 0.5s padding
            pad = 0.5
            offset = max(0, start_sec - pad)
            duration = (end_sec - start_sec) + 2 * pad
            audio, sr = librosa.load(audio_path, sr=16000, offset=offset, duration=duration)

            if len(audio) < 800:  # < 50ms at 16kHz
                failed += 1
                continue

            # Run through Whisper encoder
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(extract_device)

            with torch.no_grad():
                out = encoder(input_features)
                emb = out.last_hidden_state.mean(dim=1).squeeze(0).numpy()

            embeddings.append(emb)
            labels.append(is_filler)

        except Exception as e:
            logger.warning(f"  Failed sample {i} ({audio_path}): {e}")
            failed += 1
            continue

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(samples)}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    logger.info(f"\nExtraction complete: {len(embeddings)} embeddings, {failed} failed")
    logger.info(f"  Fillers: {int(labels.sum())}, Non-fillers: {int((1 - labels).sum())}")

    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(embeddings_dir, "labels.npy"), labels)
    logger.info(f"Saved to {embeddings_dir}/")


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def augment_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                       noise_std: float = 0.02, n_copies: int = 3) -> tuple:
    """Augment embeddings by adding small Gaussian noise.

    Creates n_copies noisy variants of each sample.
    """
    aug_embs = [embeddings]
    aug_labels = [labels]
    rng = np.random.RandomState(42)
    for _ in range(n_copies):
        noise = rng.normal(0, noise_std, embeddings.shape).astype(np.float32)
        aug_embs.append(embeddings + noise)
        aug_labels.append(labels)
    return np.concatenate(aug_embs), np.concatenate(aug_labels)


def train(embeddings_dir: str, device: str = "mps", epochs: int = 50,
          batch_size: int = 16, lr: float = 1e-3, patience: int = 8,
          model_dir: str = "models/filler_verifier"):
    """Train the FillerVerifierMLP on pre-extracted embeddings.

    Uses BCEWithLogitsLoss, AdamW, early stopping, and checkpointing.
    Automatically uses small model + augmentation when dataset < 200 samples.

    Args:
        embeddings_dir: Directory with embeddings.npy and labels.npy.
        device: Training device.
        epochs: Max training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        model_dir: Where to save best_model.pt.
    """
    # Load data
    embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    labels = np.load(os.path.join(embeddings_dir, "labels.npy"))
    logger.info(f"Loaded {len(embeddings)} samples ({int(labels.sum())} fillers, "
                f"{int((1 - labels).sum())} non-fillers)")

    small_dataset = len(embeddings) < 200
    if small_dataset:
        logger.info("Small dataset detected — using lightweight model + augmentation")

    # Stratified train/val/test split: 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        embeddings, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Augment training data for small datasets
    if small_dataset:
        X_train, y_train = augment_embeddings(X_train, y_train, noise_std=0.02, n_copies=4)
        logger.info(f"After augmentation: Train = {len(X_train)}")

    train_ds = FillerEmbeddingDataset(X_train, y_train)
    val_ds = FillerEmbeddingDataset(X_val, y_val)
    test_ds = FillerEmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model — use small variant for small datasets
    model = FillerVerifierMLP(input_dim=embeddings.shape[1], small=small_dataset)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    os.makedirs(model_dir, exist_ok=True)
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch).squeeze(-1)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.cpu().numpy())
        val_loss /= len(val_ds)

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)

        scheduler.step(val_f1)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"Prec: {val_prec:.4f} | "
            f"Rec: {val_rec:.4f}"
        )

        # ── Checkpointing ──
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
            }, os.path.join(model_dir, "best_model.pt"))
            logger.info(f"  ✓ Saved best model (F1={val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── Test evaluation ──
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)

    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"),
                            map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(y_batch.numpy())

    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)

    logger.info(f"\nTest F1:        {test_f1:.4f}")
    logger.info(f"Test Precision: {test_prec:.4f}")
    logger.info(f"Test Recall:    {test_rec:.4f}")
    logger.info(f"\n{classification_report(test_labels, test_preds, target_names=['Not Filler', 'Filler'], zero_division=0)}")

    if test_f1 >= 0.90:
        logger.info("✓ PASSED: Test F1 ≥ 0.90")
    else:
        logger.warning(f"✗ BELOW TARGET: Test F1 = {test_f1:.4f} (target ≥ 0.90)")
        logger.warning("  Consider adding more labeled data or adjusting hyperparameters.")

    return test_f1


# ═══════════════════════════════════════════════════════════════
# K-FOLD CROSS-VALIDATION TRAINING
# ═══════════════════════════════════════════════════════════════

def train_kfold(embeddings_dir: str, device: str = "mps", epochs: int = 50,
                batch_size: int = 16, lr: float = 1e-3, patience: int = 8,
                model_dir: str = "models/filler_verifier", k: int = 5):
    """Train with stratified k-fold CV for reliable metrics on small datasets.

    Trains k models, reports mean F1 across folds, and saves the best fold model.
    """
    from sklearn.model_selection import StratifiedKFold

    embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    labels = np.load(os.path.join(embeddings_dir, "labels.npy"))
    logger.info(f"Loaded {len(embeddings)} samples ({int(labels.sum())} fillers, "
                f"{int((1 - labels).sum())} non-fillers)")

    small_dataset = len(embeddings) < 200

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_f1s = []
    best_fold_f1 = 0.0
    best_fold_state = None

    os.makedirs(model_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels), 1):
        logger.info(f"\n{'='*40} Fold {fold}/{k} {'='*40}")

        X_train, y_train = embeddings[train_idx], labels[train_idx]
        X_val, y_val = embeddings[val_idx], labels[val_idx]

        # Augment training data
        if small_dataset:
            X_train, y_train = augment_embeddings(X_train, y_train, noise_std=0.02, n_copies=4)

        train_ds = FillerEmbeddingDataset(X_train, y_train)
        val_ds = FillerEmbeddingDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = FillerVerifierMLP(input_dim=embeddings.shape[1], small=small_dataset)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        fold_best_f1 = 0.0
        fold_best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch).squeeze(-1)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_batch.numpy())

            vf1 = f1_score(val_true, val_preds, zero_division=0)

            if vf1 > fold_best_f1:
                fold_best_f1 = vf1
                fold_best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        fold_prec = precision_score(val_true, val_preds, zero_division=0)
        fold_rec = recall_score(val_true, val_preds, zero_division=0)
        fold_f1s.append(fold_best_f1)
        logger.info(f"Fold {fold} — F1: {fold_best_f1:.4f}  Prec: {fold_prec:.4f}  Rec: {fold_rec:.4f}")

        if fold_best_f1 > best_fold_f1:
            best_fold_f1 = fold_best_f1
            best_fold_state = fold_best_state

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD RESULTS ({k} folds)")
    logger.info(f"{'='*60}")
    logger.info(f"Per-fold F1: {[f'{f:.4f}' for f in fold_f1s]}")
    logger.info(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"Best fold F1: {best_fold_f1:.4f}")

    # Save best fold model
    torch.save({
        "model_state_dict": best_fold_state,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "best_fold_f1": best_fold_f1,
        "fold_f1s": fold_f1s,
        "small_model": small_dataset,
    }, os.path.join(model_dir, "best_model.pt"))
    logger.info(f"\nSaved best model to {model_dir}/best_model.pt")

    if mean_f1 >= 0.90:
        logger.info("✓ PASSED: Mean F1 ≥ 0.90")
    else:
        logger.warning(f"✗ BELOW TARGET: Mean F1 = {mean_f1:.4f} (target ≥ 0.90)")
        logger.warning("  More labeled data is likely needed.")

    return mean_f1


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train filler verifier MLP")
    parser.add_argument("--preextract", action="store_true",
                        help="Pre-extract Whisper embeddings (run this first)")
    parser.add_argument("--data", default="data/filler_labels.json",
                        help="Path to labeled dataset JSON")
    parser.add_argument("--embeddings_dir", default="data/filler_embeddings",
                        help="Directory for embeddings")
    parser.add_argument("--device", default="mps",
                        help="Device: mps, cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--model_dir", default="models/filler_verifier")
    parser.add_argument("--kfold", action="store_true",
                        help="Use k-fold CV (recommended for small datasets)")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    if args.preextract:
        preextract_embeddings(args.data, args.embeddings_dir, args.device)
    elif args.kfold:
        train_kfold(
            embeddings_dir=args.embeddings_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            model_dir=args.model_dir,
            k=args.k,
        )
    else:
        train(
            embeddings_dir=args.embeddings_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            model_dir=args.model_dir,
        )
