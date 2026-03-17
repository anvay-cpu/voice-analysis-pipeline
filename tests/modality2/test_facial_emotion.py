"""Tests for facial emotion recognition (Phase 8)."""

import numpy as np
import pytest
import torch

from src.body.facial_emotion import (
    EMOTION_CLASSES,
    INFERENCE_TRANSFORM,
    NUM_CLASSES,
    FacialEmotionClassifier,
    FacialEmotionNet,
)


# ─── Model Architecture Tests ───


def test_model_output_shape():
    """FacialEmotionNet produces (B, 7) logits."""
    model = FacialEmotionNet(num_classes=7)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 7)


def test_model_custom_num_classes():
    """Model supports different class counts."""
    model = FacialEmotionNet(num_classes=3)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)


def test_model_has_features_avgpool_classifier():
    """Model has the expected sub-modules."""
    model = FacialEmotionNet()
    assert hasattr(model, "features")
    assert hasattr(model, "avgpool")
    assert hasattr(model, "classifier")


def test_model_classifier_head_structure():
    """Classifier head has Dropout->Linear->ReLU->Dropout->Linear."""
    model = FacialEmotionNet()
    layers = list(model.classifier.children())
    assert len(layers) == 5
    assert isinstance(layers[0], torch.nn.Dropout)
    assert isinstance(layers[1], torch.nn.Linear)
    assert layers[1].in_features == 1280
    assert layers[1].out_features == 256
    assert isinstance(layers[2], torch.nn.ReLU)
    assert isinstance(layers[3], torch.nn.Dropout)
    assert isinstance(layers[4], torch.nn.Linear)
    assert layers[4].out_features == NUM_CLASSES


# ─── Constants Tests ───


def test_emotion_classes():
    """7 emotion classes defined."""
    assert len(EMOTION_CLASSES) == 7
    assert "Angry" in EMOTION_CLASSES
    assert "Happy" in EMOTION_CLASSES
    assert "Neutral" in EMOTION_CLASSES
    assert "Disgust" in EMOTION_CLASSES
    assert "Fear" in EMOTION_CLASSES
    assert "Sad" in EMOTION_CLASSES
    assert "Surprise" in EMOTION_CLASSES


def test_num_classes_matches():
    """NUM_CLASSES matches EMOTION_CLASSES length."""
    assert NUM_CLASSES == len(EMOTION_CLASSES)


# ─── Transform Tests ───


def test_inference_transform_output():
    """Transform converts (H, W, 3) uint8 to (3, 224, 224) tensor."""
    img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    tensor = INFERENCE_TRANSFORM(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_inference_transform_normalizes():
    """Transform applies ImageNet normalization (not raw 0-1 from ToTensor)."""
    img = np.ones((48, 48, 3), dtype=np.uint8) * 128
    tensor = INFERENCE_TRANSFORM(img)
    # 128/255 ≈ 0.502; after normalization by ImageNet stats,
    # values differ from raw 0.502 — check channel means are shifted
    raw = 128.0 / 255.0
    channel_means = [tensor[c].mean().item() for c in range(3)]
    assert any(abs(m - raw) > 0.05 for m in channel_means)


# ─── Classifier Tests ───


def test_classifier_no_model_fallback():
    """Classifier without model returns Neutral fallback."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    assert clf.model is None
    result = clf._fallback()
    assert result["emotion"] == "Neutral"
    assert result["confidence"] == 0.0
    assert len(result["probabilities"]) == 7


def test_classifier_fallback_structure():
    """Fallback has all required keys."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    result = clf._fallback()
    assert "emotion" in result
    assert "emotion_idx" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_classify_crop_no_model():
    """classify_crop returns fallback when no model loaded."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    crop = np.zeros((48, 48, 3), dtype=np.uint8)
    result = clf.classify_crop(crop)
    assert result["emotion"] == "Neutral"
    assert result["confidence"] == 0.0


def test_classify_frame_no_bbox_no_model():
    """classify_frame without bbox returns fallback when no model."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.classify_frame(frame)
    assert result["emotion"] == "Neutral"


def test_classify_frame_with_bbox_no_model():
    """classify_frame with bbox returns fallback when no model."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.classify_frame(frame, face_bbox=(100, 100, 200, 200))
    assert result["emotion"] == "Neutral"


def test_classify_frame_empty_bbox():
    """classify_frame with zero-area bbox returns fallback."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.classify_frame(frame, face_bbox=(100, 100, 100, 100))
    assert result["emotion"] == "Neutral"


def test_classify_frame_clips_bbox():
    """classify_frame clips bbox to image bounds."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.classify_frame(frame, face_bbox=(-50, -50, 700, 500))
    assert result["emotion"] == "Neutral"


def test_classes_attribute():
    """Classifier stores classes list."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    assert clf.classes == list(EMOTION_CLASSES)
    assert len(clf.classes) == 7


# ─── Track Video Tests ───


def test_track_video_empty():
    """track_video with empty list returns empty."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    results = clf.track_video([])
    assert results == []


def test_track_video_missing_frames():
    """track_video handles missing frame files gracefully."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    results = clf.track_video(["nonexistent1.jpg", "nonexistent2.jpg"])
    assert len(results) == 2
    for r in results:
        assert r["emotion"] == "Neutral"
        assert "frame_idx" in r


def test_track_video_frame_idx():
    """track_video assigns correct frame indices."""
    clf = FacialEmotionClassifier(model_path="nonexistent_model.pt")
    results = clf.track_video(["a.jpg", "b.jpg", "c.jpg"])
    assert [r["frame_idx"] for r in results] == [0, 1, 2]


# ─── Model with Inference Tests (requires model file) ───


@pytest.fixture
def loaded_classifier():
    """Load classifier with real model if available."""
    from pathlib import Path
    model_path = "models/facial_emotion/best_model.pt"
    if not Path(model_path).exists():
        pytest.skip("Facial emotion model not available")
    return FacialEmotionClassifier(model_path=model_path)


def test_loaded_model_classify_crop(loaded_classifier):
    """Real model classifies a face crop."""
    crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = loaded_classifier.classify_crop(crop)
    assert result["emotion"] in EMOTION_CLASSES
    assert 0.0 < result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


def test_loaded_model_probabilities_sum(loaded_classifier):
    """Probabilities sum to ~1.0."""
    crop = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    result = loaded_classifier.classify_crop(crop)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 0.01
