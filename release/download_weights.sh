#!/bin/bash
# Fetch every weight the one-click pipeline needs into a local folder, then wire ./data.
# Requires: pip install huggingface_hub  (`hf` / `huggingface-cli` on PATH).
#
#   release/download_weights.sh [DEST]        # DEST default: ./weights
set -euo pipefail
DEST="${1:-./weights}"
mkdir -p "$DEST"
DL="huggingface-cli download"
command -v huggingface-cli >/dev/null || DL="hf download"

echo "[1/4] VideoNeuMat weights (T2V/LoRA/I2V generators + LRM) -> yley/videoneumat"
$DL yley/videoneumat --local-dir "$DEST/videoneumat"

echo "[2/4] Wan2.1-T2V-14B base for generation (public, ~65G)"
$DL Wan-AI/Wan2.1-T2V-14B --local-dir "$DEST/Wan-AI/Wan2.1-T2V-14B"

echo "[3/4] Wan2.1-T2V-1.3B base for the LRM (DiT + VAE + tokenizer; skips the 11G T5)"
$DL Wan-AI/Wan2.1-T2V-1.3B \
    --include "diffusion_pytorch_model*.safetensors" "Wan2.1_VAE.pth" "config.json" "google/*" \
    --local-dir "$DEST/Wan-AI/Wan2.1-T2V-1.3B"

echo "[4/4] (optional — only for I2V generation) Wan2.1-I2V-14B-720P base:"
echo "      $DL Wan-AI/Wan2.1-I2V-14B-720P --local-dir $DEST/Wan-AI/Wan2.1-I2V-14B-720P"

WDIR="$(cd "$DEST" && pwd)"
echo
echo "Done. Defaults in release/config.env already resolve to WEIGHTS_DIR=<repo>/weights,"
echo "so if DEST is the repo's ./weights you are ready. Otherwise export:"
echo "  WAN_MODELS=$WDIR   MATERIAL_CKPT=$WDIR/videoneumat/t2v/videoneumat-t2v.safetensors"
echo "  LRM_LATENT=$WDIR/videoneumat/lrm/videoneumat-lrm.pth   LRM_MLP=$WDIR/videoneumat/lrm/videoneumat-decoder.pth"
echo "Wire ./data (models + rig + null-text) with:"
echo "  WAN_MODELS=$WDIR bash release/setup_data.sh"
