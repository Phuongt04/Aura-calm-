# Voice Emotion Recognition Pipeline

Pipeline mới dùng 4 nhãn chuẩn:

- `neutral`
- `happy`
- `sad`
- `angry`

Model cũ `final_production_model.h5` không bị ghi đè. Model mới được lưu vào:

- `models/voice_emotion_stage1_best.h5`
- `models/voice_emotion_vietnamese_finetuned.h5`

## Chạy dữ liệu

```bash
python scripts/build_audio_metadata.py --data-dir data
python scripts/validate_audio_dataset.py --metadata data/audio_metadata.csv
python scripts/create_audio_splits.py --metadata data/audio_metadata.csv
```

## Train

```bash
python train_voice_emotion.py --config config/voice_emotion_config.yaml --stage all
```

Mỗi lần train tạo một thư mục mới trong:

```text
runs/voice_emotion/YYYYMMDD_HHMMSS/
```

## Evaluate

```bash
python evaluate_voice_emotion.py --model models/voice_emotion_vietnamese_finetuned.h5 --metadata data/test_metadata.csv
```

Output chính:

- `reports/classification_report.json`
- `reports/classification_report.csv`
- `reports/confusion_matrix.png`
- `reports/model_comparison.csv`

## Inference trên Jetson Nano

```bash
python voice_emotion_inference_jetson.py --model models/voice_emotion_vietnamese_finetuned.h5 --audio test.wav
```

Script inference tránh cú pháp Python 3.8+ và chỉ phụ thuộc model giọng nói.

## Export tùy chọn

```bash
python scripts/export_voice_model.py --model models/voice_emotion_vietnamese_finetuned.h5 --tflite-fp16
```
