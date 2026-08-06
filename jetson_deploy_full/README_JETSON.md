# MONE Jetson Deploy Package

Goi nay da giu cau truc path giong repo goc. Sau khi copy sang Jetson, giai nen vao `~/mone` la co the chay.

## File chinh trong goi

```text
soft/run_chatbot.py
soft/emotion_chat/**/*.py
config/emotion_support.json
angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras
label_mapping_final.pkl
best_model.h5
voice_emotion_inference_jetson.py
src/**/*.py
models/voice_emotion_label_mapping.pkl
models/voice_emotion_stage1_best.h5
config/voice_emotion_config.yaml
config/voice_emotion_thresholds.json
```

## Cai moi truong

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libhdf5-dev libopenblas-dev \
  libsndfile1 ffmpeg portaudio19-dev python3-opencv espeak-ng

cd ~/mone
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-runtime.txt
```

TensorFlow tren Jetson nen cai theo dung JetPack/Python version dang dung. Sau khi cai xong, test:

```bash
python3 - <<'PY'
import tensorflow as tf
print(tf.__version__)
PY
```

## Chay MONE runtime lan dau

Nen test voice/chat truoc, tat camera va Face ID:

```bash
source .venv/bin/activate
export AURA_DEPLOY_TARGET=jetson
export AURA_CHAT_BACKEND=rule
export AURA_WAKE_WORD_ENABLED=0
export AURA_FACE_ENABLED=0
export AURA_FACE_ID_ENABLED=0
python3 soft/run_chatbot.py
```

Khi da on dinh, co the bat camera:

```bash
export AURA_FACE_ENABLED=1
export AURA_FACE_ID_ENABLED=1
python3 soft/run_chatbot.py
```

## Test voice emotion bang file WAV

```bash
source .venv/bin/activate
python3 voice_emotion_inference_jetson.py \
  --model models/voice_emotion_stage1_best.h5 \
  --audio test.wav
```

## Ghi chu

- Goi nay khong gom dataset train, `soft/archive`, `runs`, `reports`, `.venv`, file zip lon.
- Runtime `soft/` mac dinh dung `angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras`.
- Script `voice_emotion_inference_jetson.py` dung model gon `models/voice_emotion_stage1_best.h5`.
- Neu khong co internet cho Google STT, dat `AURA_VOICE_CHAT_ENABLED=0` de chat bang ban phim.
