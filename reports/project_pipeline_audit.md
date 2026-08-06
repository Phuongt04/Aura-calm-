# Project Pipeline Audit

Ngay 2026-07-14, workspace `D:\HAHA` duoc quet theo nhom file project, bo qua `.venv`, `.venv_face_convert` va `__pycache__`.

## Cac vung chinh

- `soft/`: pipeline cu cua robot/chatbot MONE. Chua nhan dien cam xuc giong noi, face, fusion, chatbot, personalization, TTS/STT va cac script train cu.
- `data/`: raw WAV 4 lop tieng Viet/TESS dang nam theo folder `binhthuong`, `buon`, `gian`, `vui`; hien da co metadata moi.
- `soft/archive/`: dataset archive da chia san `train/phase1`, `train/phase2`, `val`, `test`, nhan English `neutral`, `happy`, `sad`, `angry`.
- `soft/dataset/`: cache `.npy` feature cu, shape `(10, 64, 64, 1)`, dung cho model CNN + BiLSTM cu.
- `soft/archive_feature_cache/`: cache `.npy` tu `soft/archive` cho train mixed archive cu.
- `mixed_archive_training/`: ket qua cac lan fine-tune cu, gom checkpoint, metrics, confusion matrix, metadata; khong phai raw dataset.
- `cv_audio_runs/`: ket qua 5-fold CV tren `soft/dataset`.
- `src/`, `scripts/`, `train_voice_emotion.py`, `evaluate_voice_emotion.py`, `voice_emotion_inference_jetson.py`: pipeline voice emotion moi.
- `hard/`, `paper/`, `don_phieu/`, `datawrite/`, `Cell Press_20260106/`: tai lieu, paper, hardware/3D, bao cao; khong thuoc voice model runtime.

## Pipeline cu dang chay nhu the nao

### Inference giong noi cu

File chinh:

- `soft/emotion_chat/config.py`
- `soft/emotion_chat/audio.py`
- `soft/emotion_chat/model_loader.py`
- `soft/emotion_chat/main.py`

Luong xu ly:

1. Load `final_production_model.h5`.
2. Load `label_mapping_final.pkl`, nhan cu: `binhthuong`, `buon`, `gian`, `vui`.
3. Thu am bang microphone hoac demo audio.
4. `prepare_for_model()` giam nhieu nhe, pre-emphasis, trim silence, chuan hoa peak.
5. Chia audio thanh cua so 3 giay, stride 1.5 giay.
6. Moi cua so bi cat thanh 10 chunk.
7. Moi chunk tao Mel spectrogram roi `cv2.resize` ve `64 x 64`.
8. Input model cu la `(10, 64, 64, 1)`.
9. Model du doan tung window, lay trung binh probability.
10. Neu co face model thi module fusion hop nhat voice + face; neu xung dot uu tien cam xuc tieu cuc hon hoac hoi lai.

Thong so runtime cu:

- sample rate: `22050`
- duration/window: `3.0s`
- window stride: `1.5s`
- input feature: sequence 10 anh Mel `64x64`
- model: CNN + Bidirectional LSTM
- model size: khoang `100.8 MB`

### Train cu tren `soft/dataset`

File chinh: `soft/train.py`.

Luong train:

1. Load tat ca `.npy` trong `soft/dataset/{binhthuong,buon,gian,vui}`.
2. Kiem tra shape `(10, 64, 64, 1)`.
3. Chay stratified 5-fold theo label.
4. Model: TimeDistributed Conv2D 32/64/128 -> Flatten -> BiLSTM 128 -> BiLSTM 64 -> Dense 128 -> softmax.
5. Augment trong spectrogram: gain, gaussian noise, background noise, engine-like noise, roll frame, freq/time mask.
6. Chon checkpoint theo `val_accuracy`.
7. Train final tren toan bo `soft/dataset` va save `final_production_model.h5`.

Ket qua cu tren `soft/dataset`:

- samples: `1862`
- class counts: `binhthuong 462`, `buon 467`, `gian 467`, `vui 466`
- OOF accuracy: `0.9581`
- OOF macro F1 xap xi cao theo tung lop, nhung day la tren cache cu va co nguy co qua lac quan.

### Fine-tune cu tren `soft/archive`

File chinh: `soft/train_mixed_archive.py`.

Luong train:

1. Load model bat dau tu `final_production_model.h5`.
2. Option `old-dataset`: `soft/dataset`.
3. Option `archive-root`: `soft/archive`.
4. Convert WAV archive sang feature cu `(10, 64, 64, 1)` bang cung `audio_pipeline.preprocess`.
5. Cache feature vao `soft/archive_feature_cache`.
6. Train voi `train-source`: `mixed`, `archive-only`, hoac `old-only`.
7. Val/test co san tu `soft/archive/val` va `soft/archive/test`.
8. Co callback tinh `val_macro_f1`, nhung mot so run dau van chon checkpoint theo accuracy.
9. Co class weight va multiplier thu cong, vi du tang `vui`.

Ket qua run cu tren archive:

- train archive: `7465`
- val: `792`
- test: `944`
- test Macro F1 cac run chinh: khoang `0.428` den `0.475`
- run tot nhat theo test macro F1 trong cac metadata da doc: `mixed_archive_training/20260709_202425`, test macro F1 `0.4751`, accuracy `0.4767`
- run moi nhat `20260711_133843`: test macro F1 `0.4582`, accuracy `0.4915`

## Van de chinh cua pipeline cu

1. CV tren `soft/dataset` qua cao nhung test tren archive thap, cho thay domain shift hoac leakage/overfit tren feature cache cu.
2. Feature cu resize spectrogram thanh anh `64x64`, lam meo truc thoi gian.
3. Input la 10 anh Mel rieng le, khong dung Log-Mel + Delta + Delta-delta lien tuc theo thoi gian.
4. Runtime cu sample rate `22050`, archive metadata lai target `16000`; khong co metadata/preprocessing single source of truth.
5. Nhieu output model nam long o root, de nham dau la production model.
6. `soft/preprocessing/*.py` co hard-code path Windows cu `D:\Aura clam`.
7. `soft/train.py` co the ghi de `final_production_model.h5`.
8. Khong co validation dataset quality report truoc khi train.
9. Chua co split metadata tai lap cho toan bo data gop.
10. Inference uncertain cu dua vao margin/agreement theo phan tram, chua co calibration threshold tren validation.

## Pipeline moi da co

- `scripts/build_audio_metadata.py`: quet nhieu data roots, hien dung `data` + `soft/archive`.
- `scripts/validate_audio_dataset.py`: phat hien file hong, duration, sample rate mismatch, silence, clipping, duplicate/group.
- `scripts/create_audio_splits.py`: split train/val/test theo `label + source`, giu group `original_id`.
- `src/audio_preprocessing.py`: preprocessing chung cho train/inference, 16 kHz, 4s, Log-Mel + Delta + Delta-delta, shape `(64, 401, 3)`.
- `src/audio_augmentation.py`: waveform augmentation nhe va SpecAugment.
- `src/models/cnn_gru_ser.py`: CNN-GRU nhe, `422,655` parameters.
- `train_voice_emotion.py`: train 2 giai doan, checkpoint theo `val_macro_f1`, class/source weighting, calibration threshold.
- `evaluate_voice_emotion.py`: Accuracy, Balanced Accuracy, Macro P/R/F1, per-class metrics, confusion matrix, comparison baseline.
- `voice_emotion_inference_jetson.py`: inference doc lap, Python 3.6 compatible.

Dataset moi da gop:

- total metadata rows: `11063`
- valid: `11062`
- invalid: `1`
- source counts: `vietnamese_kaggle 9201`, `tess 1599`, `unknown 263`
- class counts: `neutral 2597`, `sad 2357`, `angry 3028`, `happy 3081`

## Nguyen tac sap xep file

Khong xoa file. Khong move cac file active runtime cu:

- `final_production_model.h5`
- `label_mapping_final.pkl`
- `soft/emotion_chat/*`
- `soft/train.py`
- `soft/train_mixed_archive.py`
- `soft/dataset/`
- `soft/archive/`
- `data/`

Cac file ket qua cu nam long o root nen dua vao:

- `models/legacy_audio_cnn_lstm/`
- `reports/legacy_audio_cnn_lstm/`
- `docs/project_notes/`

Pipeline moi giu o:

- `src/`
- `scripts/`
- `config/`
- `models/`
- `reports/`
- `runs/`

## Sap xep da thuc hien

Khong xoa file. Cac file active cua pipeline cu va moi van giu nguyen vi tri.

Da dua checkpoint thu nghiem/legacy vao:

- `models/legacy_audio_cnn_lstm/best_model_fold_1.h5`
- `models/legacy_audio_cnn_lstm/best_model_improved.h5`
- `models/legacy_audio_cnn_lstm/final_model_archive_macro_f1*.h5`
- `models/legacy_audio_cnn_lstm/final_model_archive_vui125_buon085.h5`
- `models/legacy_audio_cnn_lstm/final_production_model_mixed_archive.h5`

Da dua report/curve legacy vao:

- `reports/legacy_audio_cnn_lstm/confusion_matrix_oof.*`
- `reports/legacy_audio_cnn_lstm/cross_validation_results.json`
- `reports/legacy_audio_cnn_lstm/cv_results_*.png`
- `reports/legacy_audio_cnn_lstm/final_model_*`
- `reports/legacy_audio_cnn_lstm/model_accuracy_loss_curves.png`
- `reports/legacy_audio_cnn_lstm/training_history.json`

Da dua ghi chu/tai lieu root vao:

- `docs/project_notes/`
- `docs/project_documents/`

Van giu nguyen cac file runtime quan trong:

- `final_production_model.h5`
- `label_mapping_final.pkl`
- `soft/`
- `data/`
- `src/`
- `scripts/`
- `train_voice_emotion.py`
- `evaluate_voice_emotion.py`
- `voice_emotion_inference_jetson.py`
