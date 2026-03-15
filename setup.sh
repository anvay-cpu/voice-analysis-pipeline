#!/bin/bash
set -e

echo "=== AI Public Speaking Assistant — Environment Setup ==="

# Check Python
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# Create conda environment
echo "Creating conda environment..."
conda create -n voice-pipeline python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate voice-pipeline

# Install PyTorch (M1 optimized)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install all dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install FFmpeg
echo "Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing via Homebrew..."
    brew install ffmpeg
fi

# Verify MPS
echo "Verifying M1 GPU (MPS)..."
python3 -c "
import torch
mps = torch.backends.mps.is_available()
print(f'MPS available: {mps}')
if not mps:
    print('WARNING: MPS not available. Training will use CPU (slower).')
    print('Ensure you have macOS 12.3+ and PyTorch 2.0+')
"

# Set MPS fallback
if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" ~/.zshrc 2>/dev/null; then
    echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> ~/.zshrc
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate voice-pipeline"
