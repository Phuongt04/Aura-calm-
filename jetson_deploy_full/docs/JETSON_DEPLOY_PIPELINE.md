# Jetson Deploy Pipeline

Tai repo hien co 2 cach day xuong Jetson:

1. **Voice emotion inference gon**: chi chay nhan dien cam xuc tu file/mic audio.
2. **MONE runtime day du**: chay chatbot, wake word, STT/TTS, voice emotion, face emotion/Face ID neu bat.

Nen day runtime, model va config. Khong day dataset train, report, run log, virtualenv, file zip lon.

## 1. Pipeline train/export tren may PC

Neu train lai voice model:

```bash
python scripts/build_audio_metadata.py --data-dir data
python scripts/validate_audio_dataset.py --metadata data/audio_metadata.csv
python scripts/create_audio_splits.py --metadata data/audio_metadata.csv
python train_voice_emotion.py --config config/voice_emotion_config.yaml --stage all
python evaluate_voice_emotion.py --model models/voice_emotion_vietnamese_finetuned.h5 --metadata data/test_metadata.csv
```

Neu can export them:

```bash
python scripts/export_voice_model.py --model models/voice_emotion_vietnamese_finetuned.h5 --tflite-fp16
```

Luu y hien trang repo:

- README voice dang nhac `models/voice_emotion_vietnamese_finetuned.h5`, nhung file nay chua thay trong `models/`.
- Model voice gon dang co: `models/voice_emotion_stage1_best.h5`.
- Runtime `soft/` dang mac dinh dung model: `angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras`.
- Face model dang co: `best_model.h5`.

## 2. Goi A - voice emotion inference gon

Dung khi chi can test model cam xuc giong noi tren Jetson.

File can copy:

```text
voice_emotion_inference_jetson.py
src/__init__.py
src/audio_preprocessing.py
src/config_utils.py
src/voice_emotion_labels.py
config/voice_emotion_config.yaml
config/voice_emotion_thresholds.json
models/voice_emotion_label_mapping.pkl
models/voice_emotion_stage1_best.h5
```

Neu da train xong model final thi thay `models/voice_emotion_stage1_best.h5` bang:

```text
models/voice_emotion_vietnamese_finetuned.h5
```

Lenh test tren Jetson:

```bash
cd ~/mone
python3 voice_emotion_inference_jetson.py \
  --model models/voice_emotion_stage1_best.h5 \
  --audio test.wav
```

## 3. Goi B - MONE runtime day du

Dung khi Jetson chay robot/chatbot.

File bat buoc:

```text
soft/run_chatbot.py
soft/emotion_chat/**/*.py
config/emotion_support.json
angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras
label_mapping_final.pkl
```

File tuy chon:

```text
best_model.h5                         # can neu bat face emotion
soft/user_data/                       # can neu muon giu profile/history/face samples cu
soft/music/                           # can neu muon mo nhac local
assets/audio/*.wav                    # can neu dung relaxation audio trong emotion_support.json
```

Khuyen nghi khi moi deploy lan dau tren Jetson:

```bash
export AURA_DEPLOY_TARGET=jetson
export AURA_CHAT_BACKEND=rule
export AURA_WAKE_WORD_ENABLED=0
export AURA_FACE_ENABLED=0
export AURA_FACE_ID_ENABLED=0
```

Sau khi voice-only on dinh moi bat face:

```bash
export AURA_FACE_ENABLED=1
export AURA_FACE_ID_ENABLED=1
```

Lenh chay:

```bash
cd ~/mone
python3 soft/run_chatbot.py
```

## 4. Thu muc/file khong can copy sang Jetson

```text
.venv/
.venv_face_convert/
voice-emotion-data/
data/
soft/archive/
soft/archive.zip
runs/
reports/
paper/
Cell Press_20260106/
*.zip
train_voice_emotion.py
evaluate_voice_emotion.py
kaggle_angry_boost_train_cells.py
scripts/build_audio_metadata.py
scripts/create_audio_splits.py
scripts/validate_audio_dataset.py
```

Chi copy script train/evaluate neu muon train truc tiep tren Jetson, nhung khong khuyen nghi.

## 5. Moi truong Python tren Jetson

Runtime toi thieu:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libhdf5-dev libopenblas-dev \
  libsndfile1 ffmpeg portaudio19-dev python3-opencv espeak-ng

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy h5py librosa sounddevice SpeechRecognition pyttsx3 openai
```

TensorFlow tren Jetson phu thuoc JetPack/Python version. Cai ban TensorFlow tu NVIDIA/Jetson phu hop voi JetPack dang dung, roi test:

```bash
python3 - <<'PY'
import tensorflow as tf
print(tf.__version__)
PY
```

Neu chi chay `.tflite`, co the thay TensorFlow bang `tflite-runtime`, nhung code hien tai van dang load Keras `.h5/.keras`.

## 6. Checklist truoc khi copy

- Chot entry point: `voice_emotion_inference_jetson.py` hay `soft/run_chatbot.py`.
- Chot model voice: `models/voice_emotion_stage1_best.h5` hay `angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras`.
- Neu chay `soft/run_chatbot.py`, nho copy `config/emotion_support.json`.
- Neu chua dung camera, set `AURA_FACE_ENABLED=0` va `AURA_FACE_ID_ENABLED=0`.
- Neu khong co internet cho Google STT, set `AURA_VOICE_CHAT_ENABLED=0` de test bang ban phim truoc.
- Test load model truoc, roi moi test mic/camera/TTS.
