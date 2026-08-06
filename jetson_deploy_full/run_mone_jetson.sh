#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  . .venv/bin/activate
fi

export AURA_DEPLOY_TARGET="${AURA_DEPLOY_TARGET:-jetson}"
export AURA_CHAT_BACKEND="${AURA_CHAT_BACKEND:-rule}"
export AURA_WAKE_WORD_ENABLED="${AURA_WAKE_WORD_ENABLED:-0}"
export AURA_FACE_ENABLED="${AURA_FACE_ENABLED:-0}"
export AURA_FACE_ID_ENABLED="${AURA_FACE_ID_ENABLED:-0}"

python3 soft/run_chatbot.py
