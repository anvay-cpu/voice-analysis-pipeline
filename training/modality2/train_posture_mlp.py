"""Training script for Posture MLP model.

Self-supervised: uses rule-based scores as labels, trains MLP to
smooth and generalize them. No manual annotation required.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.body.posture_scorer import PostureMLP, compute_rule_score, NUM_KEYPOINTS, KEYPOINT_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    num_samples: int = 5000,
    noise_std: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data from rule-based scorer.

    Creates random plausible pose keypoints, scores them with rules,
    and adds noise for augmentation.

    Returns:
        (keypoints_flat, scores): arrays of shape (N, 132) and (N,)
    """
    rng = np.random.default_rng(42)
    all_keypoints = []
    all_scores = []

    for _ in range(num_samples):
        kp = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

        # Generate a plausible upper-body pose (normalized coords)
        # Base positions with some randomness
        lean = rng.uniform(-0.08, 0.08)  # lateral lean
        slouch = rng.uniform(-0.05, 0.05)  # forward slouch
        tilt = rng.uniform(-0.04, 0.04)  # shoulder tilt

        # Nose
        kp[0] = [0.5 + lean * 0.5, 0.20 + slouch, 0, 0.95]
        # Ears
        kp[7] = [0.45 + lean * 0.5, 0.22, 0, 0.9]   # left ear
        kp[8] = [0.55 + lean * 0.5, 0.22, 0, 0.9]   # right ear
        # Shoulders
        kp[11] = [0.40 + lean, 0.35 + tilt, 0, 0.95]  # left shoulder
        kp[12] = [0.60 + lean, 0.35 - tilt, 0, 0.95]  # right shoulder
        # Hips
        hip_asym = rng.uniform(-0.03, 0.03)
        kp[23] = [0.43 + hip_asym, 0.60, 0, 0.9]  # left hip
        kp[24] = [0.57 - hip_asym, 0.60, 0, 0.9]  # right hip

        # Fill remaining keypoints with reasonable values
        # Elbows
        kp[13] = [0.33, 0.48, 0, 0.85]
        kp[14] = [0.67, 0.48, 0, 0.85]
        # Wrists
        kp[15] = [0.30, 0.55, 0, 0.8]
        kp[16] = [0.70, 0.55, 0, 0.8]
        # Eyes
        kp[1] = [0.47, 0.18, 0, 0.9]
        kp[2] = [0.46, 0.18, 0, 0.9]
        kp[3] = [0.45, 0.18, 0, 0.9]
        kp[4] = [0.53, 0.18, 0, 0.9]
        kp[5] = [0.54, 0.18, 0, 0.9]
        kp[6] = [0.55, 0.18, 0, 0.9]
        # Mouth
        kp[9] = [0.48, 0.24, 0, 0.85]
        kp[10] = [0.52, 0.24, 0, 0.85]
        # Knees, ankles, feet — less important but fill them
        for idx in range(25, 33):
            kp[idx] = [0.5 + rng.uniform(-0.1, 0.1), 0.7 + (idx - 25) * 0.03, 0, 0.5]

        # Add noise
        noise = rng.normal(0, noise_std, kp.shape).astype(np.float32)
        noise[:, 3] = 0  # don't noise visibility
        kp_noisy = kp + noise
        kp_noisy[:, 0:2] = np.clip(kp_noisy[:, 0:2], 0, 1)
        kp_noisy[:, 3] = np.clip(kp_noisy[:, 3], 0, 1)

        rule_result = compute_rule_score(kp_noisy)
        score = rule_result["combined"] / 100.0  # normalize to 0-1

        all_keypoints.append(kp_noisy.flatten())
        all_scores.append(score)

    return np.array(all_keypoints), np.array(all_scores, dtype=np.float32)


def train(
    num_samples: int = 5000,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    save_dir: str = "models/posture_mlp",
    device: str = "mps",
):
    """Train the posture MLP on synthetic data."""
    logger.info("Generating %d synthetic training samples...", num_samples)
    X, y = generate_synthetic_data(num_samples)

    # Train/val split
    n_val = int(len(X) * val_split)
    indices = np.random.default_rng(42).permutation(len(X))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Check device availability
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"

    model = PostureMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    patience = 15
    patience_counter = 0

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Training PostureMLP on %s (%d train, %d val)", device, len(train_idx), n_val)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(train_idx)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_mae = (val_pred.cpu() - y_val).abs().mean().item()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_MAE=%.3f | best=%.3f",
                epoch, train_loss, val_mae, best_val_mae,
            )

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Convert MAE back to 0-100 scale for reporting
    mae_100 = best_val_mae * 100
    logger.info(
        "Training complete. Best val MAE: %.4f (0-1) = %.2f (0-100 scale)",
        best_val_mae, mae_100,
    )
    passed = mae_100 <= 0.6
    logger.info("Target MAE ≤ 0.6 (0-100): %s", "PASS ✓" if passed else "FAIL ✗")

    return best_val_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Posture MLP")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--save-dir", type=str, default="models/posture_mlp")
    args = parser.parse_args()

    train(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )
