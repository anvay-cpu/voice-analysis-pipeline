#!/bin/bash
# colab_sync.sh — Push local changes to GitHub, pull on Colab
# Usage: ./scripts/colab_sync.sh [push|pull|both]
#
# Workflow:
#   Local edit → ./scripts/colab_sync.sh push   → pushes to GitHub
#   Run on Colab → ./scripts/colab_sync.sh pull → pulls from GitHub to local
#   Both:         ./scripts/colab_sync.sh both  → push then pull

set -e

REMOTE_HOST="colab"
REMOTE_DIR="/root/project"
ACTION="${1:-both}"

case "$ACTION" in
  push)
    echo "→ Pushing local changes to GitHub..."
    git add -A && git commit -m "sync: local → colab" --allow-empty && git push
    echo "→ Pulling on Colab..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && git pull"
    echo "✓ Colab is up to date."
    ;;
  pull)
    echo "→ Pulling Colab changes to GitHub..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && git add -A && git commit -m 'sync: colab → local' --allow-empty && git push"
    echo "→ Pulling locally..."
    git pull
    echo "✓ Local is up to date."
    ;;
  both)
    echo "→ Syncing local → GitHub → Colab..."
    git add -A && git commit -m "sync: local → colab" --allow-empty && git push
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && git pull"
    echo "✓ Synced."
    ;;
  *)
    echo "Usage: $0 [push|pull|both]"
    exit 1
    ;;
esac
