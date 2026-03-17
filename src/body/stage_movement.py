"""Step 9: Stage movement & proxemics analysis.

Tracks the speaker's centroid (hip midpoint) across frames, computes
movement metrics (displacement, velocity, convex hull, directional entropy,
autocorrelation), classifies movement pattern, and scores stage usage.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe pose keypoint indices
LEFT_HIP = 23
RIGHT_HIP = 24

# Movement pattern thresholds
HULL_AREA_ANCHORED = 0.05   # < 5% of frame area → anchored
AUTOCORR_PACING = 0.6       # autocorrelation > 0.6 → pacing
VELOCITY_LOW = 0.005         # normalized units/frame

# Movement patterns
ANCHORED = "Anchored"
PACING = "Pacing"
ROAMING = "Roaming"
PURPOSEFUL = "Purposeful"

# Stage usage scores
PATTERN_SCORES = {
    ANCHORED: 5.0,
    PACING: 3.0,
    PURPOSEFUL: 8.0,
    ROAMING: 7.0,
}


def compute_centroids(
    poses: dict[int, np.ndarray | None],
) -> list[tuple[int, float, float]]:
    """Compute speaker centroid (hip midpoint) per frame.

    Args:
        poses: Dict mapping frame_idx to (33, 4) keypoints or None.

    Returns:
        List of (frame_idx, cx, cy) tuples for frames with valid poses.
        cx, cy are normalized to [0, 1].
    """
    centroids = []
    for idx in sorted(poses.keys()):
        kp = poses[idx]
        if kp is None:
            continue
        lh = kp[LEFT_HIP]
        rh = kp[RIGHT_HIP]
        # Skip if both hips have low visibility
        if lh[3] < 0.3 and rh[3] < 0.3:
            continue
        cx = (lh[0] + rh[0]) / 2.0
        cy = (lh[1] + rh[1]) / 2.0
        centroids.append((idx, float(cx), float(cy)))
    return centroids


def compute_displacement(centroids: list[tuple[int, float, float]]) -> float:
    """Total displacement: sum of frame-to-frame Euclidean distances."""
    if len(centroids) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(centroids)):
        dx = centroids[i][1] - centroids[i - 1][1]
        dy = centroids[i][2] - centroids[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def compute_mean_velocity(
    centroids: list[tuple[int, float, float]],
) -> float:
    """Mean velocity in normalized units per frame."""
    if len(centroids) < 2:
        return 0.0
    total_dist = compute_displacement(centroids)
    frame_span = centroids[-1][0] - centroids[0][0]
    if frame_span <= 0:
        return 0.0
    return total_dist / frame_span


def compute_convex_hull_area(
    centroids: list[tuple[int, float, float]],
) -> float:
    """Convex hull area of centroid positions (0-1 normalized frame area).

    Uses the Shoelace formula on the convex hull computed via
    Graham scan / gift wrapping.
    """
    if len(centroids) < 3:
        return 0.0

    points = np.array([[c[1], c[2]] for c in centroids])

    # Remove duplicate points
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(unique)
        return float(hull.volume)  # 2D: volume = area
    except (ImportError, Exception):
        # Fallback: bounding box area
        mins = unique.min(axis=0)
        maxs = unique.max(axis=0)
        return float((maxs[0] - mins[0]) * (maxs[1] - mins[1]))


def compute_directional_entropy(
    centroids: list[tuple[int, float, float]],
    n_bins: int = 8,
) -> float:
    """Directional entropy of movement vectors.

    Higher entropy = more random/varied directions.
    Range: 0 (single direction) to log2(n_bins) (uniform).
    """
    if len(centroids) < 3:
        return 0.0

    angles = []
    for i in range(1, len(centroids)):
        dx = centroids[i][1] - centroids[i - 1][1]
        dy = centroids[i][2] - centroids[i - 1][2]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            continue
        angle = math.atan2(dy, dx)  # -pi to pi
        angles.append(angle)

    if len(angles) < 2:
        return 0.0

    # Bin angles into n_bins sectors
    bins = np.zeros(n_bins)
    for a in angles:
        # Map [-pi, pi] to [0, 2*pi], then to bin index
        a_pos = a + math.pi
        bin_idx = int(a_pos / (2 * math.pi) * n_bins) % n_bins
        bins[bin_idx] += 1

    # Normalize to probabilities
    total = bins.sum()
    if total == 0:
        return 0.0
    probs = bins / total

    # Shannon entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_autocorrelation(
    centroids: list[tuple[int, float, float]],
    max_lag: int = 30,
) -> float:
    """Detect repetitive pacing via autocorrelation of x-position.

    Returns the maximum autocorrelation coefficient for lags > 5 frames.
    High value (> 0.6) indicates repetitive back-and-forth movement.
    """
    if len(centroids) < max_lag + 5:
        return 0.0

    x = np.array([c[1] for c in centroids])
    x = x - x.mean()

    var = np.var(x)
    if var < 1e-10:
        return 0.0

    max_corr = 0.0
    n = len(x)
    for lag in range(6, min(max_lag + 1, n // 2)):
        corr = np.mean(x[:n - lag] * x[lag:]) / var
        if corr > max_corr:
            max_corr = corr

    return float(max_corr)


def classify_movement(
    hull_area: float,
    mean_velocity: float,
    autocorrelation: float,
    directional_entropy: float,
) -> str:
    """Classify movement pattern from metrics.

    Returns one of: Anchored, Pacing, Roaming, Purposeful.
    """
    if hull_area < HULL_AREA_ANCHORED and mean_velocity < VELOCITY_LOW:
        return ANCHORED

    if autocorrelation > AUTOCORR_PACING:
        return PACING

    max_entropy = math.log2(8)  # 3.0 for 8 bins
    if hull_area > 0.10 and directional_entropy > 0.6 * max_entropy:
        return ROAMING

    return PURPOSEFUL


def score_stage_usage(pattern: str) -> float:
    """Score stage usage on 1-10 scale based on movement pattern."""
    return PATTERN_SCORES.get(pattern, 5.0)


class StageMovementAnalyzer:
    """Analyze speaker stage movement from pose keypoints."""

    def analyze(
        self,
        poses: dict[int, np.ndarray | None],
        fps: float = 30.0,
    ) -> dict:
        """Full stage movement analysis.

        Args:
            poses: Dict mapping frame_idx to (33, 4) keypoints or None.
            fps: Video frame rate (for velocity in units/sec).

        Returns:
            Dict with centroids, metrics, pattern, score.
        """
        centroids = compute_centroids(poses)

        if len(centroids) < 2:
            logger.warning("Too few valid frames for movement analysis")
            return {
                "centroids": [],
                "total_displacement": 0.0,
                "mean_velocity": 0.0,
                "convex_hull_area": 0.0,
                "directional_entropy": 0.0,
                "autocorrelation": 0.0,
                "movement_pattern": ANCHORED,
                "stage_usage_score": PATTERN_SCORES[ANCHORED],
                "num_frames": len(centroids),
            }

        displacement = compute_displacement(centroids)
        velocity = compute_mean_velocity(centroids)
        hull_area = compute_convex_hull_area(centroids)
        entropy = compute_directional_entropy(centroids)
        autocorr = compute_autocorrelation(centroids)

        pattern = classify_movement(hull_area, velocity, autocorr, entropy)
        score = score_stage_usage(pattern)

        # Velocity in normalized units/sec
        velocity_per_sec = velocity * fps

        logger.info(
            "Stage movement: pattern=%s, score=%.1f, hull=%.4f, "
            "vel=%.4f/s, entropy=%.2f, autocorr=%.2f",
            pattern, score, hull_area, velocity_per_sec,
            entropy, autocorr,
        )

        return {
            "centroids": [
                {"frame_idx": c[0], "x": round(c[1], 4), "y": round(c[2], 4)}
                for c in centroids
            ],
            "total_displacement": round(displacement, 4),
            "mean_velocity": round(velocity_per_sec, 4),
            "convex_hull_area": round(hull_area, 6),
            "directional_entropy": round(entropy, 4),
            "autocorrelation": round(autocorr, 4),
            "movement_pattern": pattern,
            "stage_usage_score": score,
            "num_frames": len(centroids),
        }
