# JETSON DEPLOY REPORT - MONE FULL PACKAGE

Ngay tao goi: 2026-07-31
May nguon: D:\HAHA
Ten goi: jetson_deploy_full.zip
Thu muc sau khi giai nen khuyen nghi: ~/mone

## 1. Muc tieu goi

Goi nay dung de day xuong Jetson va chay he thong MONE runtime day du:

- Nhan dien cam xuc giong noi.
- Chatbot MONE bang rule backend mac dinh tren Jetson.
- Wake word, ghi am den khi im lang, STT/TTS neu moi truong ho tro.
- Face emotion va Face ID neu bat camera/model.
- Co them script voice-only de test model bang file WAV.

Goi nay khong gom dataset train va cac output nang.

## 2. File goi da dong

Runtime chatbot:

```text
soft/run_chatbot.py
soft/emotion_chat/**/*.py
```

Config:

```text
config/emotion_support.json
config/voice_emotion_config.yaml
config/voice_emotion_thresholds.json
```

Model:

```text
angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras
label_mapping_final.pkl
best_model.h5
models/voice_emotion_stage1_best.h5
models/voice_emotion_label_mapping.pkl
```

Voice-only inference:

```text
voice_emotion_inference_jetson.py
src/**/*.py
```

Tai lieu/launcher:

```text
README_JETSON.md
JETSON_DEPLOY_REPORT.md
docs/JETSON_DEPLOY_PIPELINE.md
requirements-runtime.txt
run_mone_jetson.sh
```

## 3. File khong co trong goi

Khong copy cac phan nay xuong Jetson:

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
*.zip khac
train/evaluate scripts khong can cho runtime
```

Ly do: cac file nay dung cho train/evaluate/bao cao, khong can cho runtime va rat nang.

## 4. Copy goi sang Jetson

Tu may Windows/PC, dung mot trong cac cach sau.

Neu co SSH:

```bash
scp jetson_deploy_full.zip jetson@JETSON_IP:/home/jetson/
```

Tren Jetson:

```bash
cd /home/jetson
rm -rf mone
mkdir -p mone
unzip jetson_deploy_full.zip -d mone
cd mone
```

Neu dung USB:

```bash
cp /media/$USER/USB/jetson_deploy_full.zip /home/jetson/
cd /home/jetson
rm -rf mone
mkdir -p mone
unzip jetson_deploy_full.zip -d mone
cd mone
```

## 5. Cai moi truong tren Jetson

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

TensorFlow tren Jetson phai cai theo dung JetPack/Python version. Sau khi cai, test:

```bash
python3 - <<'PY'
import tensorflow as tf
print(tf.__version__)
PY
```

Neu OpenCV qua pip bi loi camera/GUI, uu tien dung `python3-opencv` tu apt.

## 6. Chay test theo thu tu an toan

### Buoc 1 - Test import/runtime

```bash
cd ~/mone
source .venv/bin/activate
python3 - <<'PY'
import numpy, h5py, librosa, cv2
print("basic imports OK")
PY
```

### Buoc 2 - Test load TensorFlow

```bash
python3 - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__)
PY
```

### Buoc 3 - Test MONE khong camera, khong wake word

```bash
export AURA_DEPLOY_TARGET=jetson
export AURA_CHAT_BACKEND=rule
export AURA_WAKE_WORD_ENABLED=0
export AURA_FACE_ENABLED=0
export AURA_FACE_ID_ENABLED=0
python3 soft/run_chatbot.py
```

Hoac:

```bash
chmod +x run_mone_jetson.sh
./run_mone_jetson.sh
```

### Buoc 4 - Test voice-only bang WAV

Can co file `test.wav` trong `~/mone`.

```bash
python3 voice_emotion_inference_jetson.py \
  --model models/voice_emotion_stage1_best.h5 \
  --audio test.wav
```

### Buoc 5 - Bat camera sau khi voice da on

```bash
export AURA_FACE_ENABLED=1
export AURA_FACE_ID_ENABLED=1
python3 soft/run_chatbot.py
```

## 7. Bien moi truong quan trong

Che do Jetson:

```bash
export AURA_DEPLOY_TARGET=jetson
```

Chat rule, khong can Ollama:

```bash
export AURA_CHAT_BACKEND=rule
```

Tat/bat wake word:

```bash
export AURA_WAKE_WORD_ENABLED=0
export AURA_WAKE_WORD_ENABLED=1
```

Tat/bat voice chat:

```bash
export AURA_VOICE_CHAT_ENABLED=0
export AURA_VOICE_CHAT_ENABLED=1
```

Tat/bat camera:

```bash
export AURA_FACE_ENABLED=0
export AURA_FACE_ID_ENABLED=0
export AURA_FACE_ENABLED=1
export AURA_FACE_ID_ENABLED=1
```

Dung model khac neu can:

```bash
export AURA_AUDIO_MODEL_PATH="/home/jetson/mone/angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras"
export AURA_FACE_MODEL_PATH="/home/jetson/mone/best_model.h5"
```

## 8. Entry point

Chay chatbot/robot:

```bash
python3 soft/run_chatbot.py
```

Chay script goi san:

```bash
./run_mone_jetson.sh
```

Chay nhan dien voice bang file:

```bash
python3 voice_emotion_inference_jetson.py --model models/voice_emotion_stage1_best.h5 --audio test.wav
```

## 9. Ghi chu ve model

Model runtime day du:

```text
angry_boost_outputs (1)/models/cnn_gru_angry_boost_best.keras
```

Nhan runtime:

```text
neutral -> binhthuong
happy   -> vui
sad     -> buon
angry   -> gian
```

Model face emotion:

```text
best_model.h5
```

Model voice-only:

```text
models/voice_emotion_stage1_best.h5
```

README cu co nhac `models/voice_emotion_vietnamese_finetuned.h5`, nhung file nay khong co trong repo hien tai. Goi nay dung `models/voice_emotion_stage1_best.h5` cho voice-only va model `.keras` angry-boost cho runtime.

## 10. Loi thuong gap

Loi khong tim thay TensorFlow:

```text
ModuleNotFoundError: No module named 'tensorflow'
```

Cach xu ly: cai TensorFlow dung ban JetPack. Khong nen cai bua ban desktop neu Jetson/Python khong hop.

Loi mic/sounddevice:

```text
PortAudioError
```

Cach xu ly:

```bash
sudo apt install -y portaudio19-dev
python3 -m pip install sounddevice
python3 - <<'PY'
import sounddevice as sd
print(sd.query_devices())
PY
```

Loi STT Google:

```text
Khong goi duoc dich vu nhan dang giong noi
```

Cach xu ly: kiem tra internet, hoac test bang ban phim:

```bash
export AURA_VOICE_CHAT_ENABLED=0
```

Loi camera:

```text
Khong the mo camera
```

Cach xu ly:

```bash
ls /dev/video*
export AURA_FACE_CAMERA_INDEX=0
```

Loi Face ID do OpenCV thieu module `cv2.face`:

```text
AttributeError: module 'cv2' has no attribute 'face'
```

Cach xu ly: tam tat Face ID:

```bash
export AURA_FACE_ID_ENABLED=0
```

Neu can Face ID offline bang LBPH, cai OpenCV contrib phu hop voi Jetson.

## 11. Checklist truoc khi giao Jetson

- Da copy `jetson_deploy_full.zip`.
- Da giai nen vao `~/mone`.
- Da cai TensorFlow va test import.
- Da chay `./run_mone_jetson.sh` voi camera tat.
- Da test mic hoac tam tat voice chat de test ban phim.
- Sau khi runtime on dinh moi bat `AURA_FACE_ENABLED=1`.
- Khong copy dataset train xuong Jetson.
