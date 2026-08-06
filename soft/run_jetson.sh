#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export AURA_DEPLOY_TARGET="${AURA_DEPLOY_TARGET:-jetson}"
export AURA_CHAT_BACKEND="${AURA_CHAT_BACKEND:-rule}"
export AURA_VOICE_CHAT_TTS_BACKEND="${AURA_VOICE_CHAT_TTS_BACKEND:-piper}"
export AURA_VOICE_CHAT_TTS_VOICE="${AURA_VOICE_CHAT_TTS_VOICE:-vi}"
export AURA_FACE_MODEL_PATH="${AURA_FACE_MODEL_PATH:-${PROJECT_ROOT}/best_model.h5}"
export AURA_AUDIO_MODEL_PATH="${AURA_AUDIO_MODEL_PATH:-${PROJECT_ROOT}/angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras}"

# Set this to your Vietnamese Piper .onnx model on Jetson, for example:
# export AURA_VOICE_CHAT_PIPER_MODEL=/home/jetson/piper/vi_VN-vais1000-medium.onnx
if [[ -z "${AURA_VOICE_CHAT_PIPER_MODEL:-}" ]]; then
  echo "[run_jetson] AURA_VOICE_CHAT_PIPER_MODEL is not set; TTS will fallback to espeak/spd-say if available."
fi

cd "${SCRIPT_DIR}"
python3 run_chatbot.py
