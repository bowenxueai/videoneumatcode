#!/bin/bash
# Wire the relative ./data folder that Stage B (the LRM) reads, for a fresh checkout.
# Idempotent. Needs WAN_MODELS = a dir holding Wan-AI/Wan2.1-T2V-14B and Wan2.1-T2V-1.3B
# (download_weights.sh puts them there). The small rig/emb files ship in ./assets.
set -euo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$REL/.." && pwd)"
: "${WAN_MODELS:?set WAN_MODELS (dir with Wan-AI/Wan2.1-T2V-14B and Wan2.1-T2V-1.3B)}"
cd "$REPO"
mkdir -p data
ln -sfn "$WAN_MODELS"                        data/models                 # Wan base (14B for gen, 1.3B for the LRM)
ln -sfn "$REPO/assets/null_text_emb.pt"      data/null_text_emb.pt        # LRM null-text embedding
ln -sfn "$REPO/assets/matsynth_scene_state"  data/matsynth_scene_state    # novel-view cameras for validation
echo "[setup_data] data/models -> $WAN_MODELS ; null_text_emb + matsynth_scene_state <- assets/"
