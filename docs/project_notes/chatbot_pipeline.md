# Pipeline hoat dong cua chatbot MONE

## So do tong the

```mermaid
flowchart TD
    A([Bat dau]) --> B[Khoi tao he thong]
    B --> C{Ca nhan hoa duoc bat?}
    C -->|Co| D[Nhan dien nguoi dung bang Face ID hoac nhap user ID]
    C -->|Khong| E[Chay voi che do guest]
    D --> F[Nap ho so nguoi dung va lich su gan day]
    E --> G[Nap model]
    F --> G

    G --> G1[Nap model cam xuc giong noi]
    G --> G2{Nhan dien khuon mat duoc bat?}
    G2 -->|Co| G3[Nap model cam xuc khuon mat]
    G2 -->|Khong| H[Cho tu khoa danh thuc]
    G1 --> H
    G3 --> H

    H --> I{Nghe thay wake word?}
    I -->|Khong / Thoat| Z([Ket thuc])
    I -->|Co| J[Thu am cau noi cua nguoi dung]

    J --> K{Tin hieu am thanh hop le?}
    K -->|Qua nho / qua on| H
    K -->|Hop le| L[Tien xu ly am thanh]

    L --> L1[Giam nhieu nhe va pre-emphasis]
    L1 --> L2[Chia thanh cac cua so 3 giay]
    L2 --> L3[Tao Mel-spectrogram 64x64]
    L3 --> L4[Du doan cam xuc bang CNN + BiLSTM]
    L4 --> M[Lay nhan cam xuc giong noi va do tin cay]

    M --> N{Co model khuon mat?}
    N -->|Co| O[Nhan dien cam xuc khuon mat]
    N -->|Khong| P[Chi dung cam xuc giong noi]
    O --> Q[Hop nhat cam xuc giong noi va khuon mat]
    P --> R[Cam xuc cuoi cung]
    Q --> S{Hai nguon co xung dot?}
    S -->|Khong| R
    S -->|Co| T[Uu tien cam xuc tieu cuc hon hoac hoi lai nguoi dung]
    T --> R

    R --> U[Xay prompt theo cam xuc va ho so ca nhan]
    U --> V[Mo dau hoi thoai theo cam xuc]
    V --> W[Lang nghe / nhan tin nhan nguoi dung]

    W --> X{Nguoi dung yeu cau thoat?}
    X -->|Co| Y[Luu phien chat, cap nhat profile va bo nho dai han]
    X -->|Khong| AA{Lenh giai tri?}
    AA -->|Co| AB[Ke chuyen cuoi / mo nhac]
    AA -->|Khong| AC[Tao phan hoi bang LLM Ollama hoac rule fallback]
    AB --> AD[Noi / hien thi cau tra loi]
    AC --> AD
    AD --> W
    Y --> AE{Thu lai phien moi?}
    AE -->|Co| H
    AE -->|Khong| Z
```

## Mo ta ngan gon

1. He thong khoi dong, doc cau hinh va xac dinh che do chay: chi giong noi hoac giong noi ket hop khuon mat.
2. Neu bat ca nhan hoa, chatbot nhan dien nguoi dung bang Face ID hoac yeu cau nhap user ID, sau do nap profile va cac phien hoi thoai gan day.
3. He thong nap model cam xuc am thanh tu `final_production_model.h5` va file nhan tu `label_mapping_final.pkl`. Neu bat nhan dien khuon mat, nap them model khuon mat.
4. Chatbot cho nguoi dung goi wake word, vi du `mone`, `mone oi`, `aura`, `aura oi`.
5. Khi duoc goi, chatbot thu am den khi nguoi dung im lang hoac dat thoi gian toi da.
6. Tin hieu am thanh duoc kiem tra chat luong bang RMS, peak va noise floor ratio. Neu qua nho hoac qua nhieu, he thong bo qua va quay lai che do cho.
7. Am thanh hop le duoc tien xu ly: giam nhieu nhe, chuan hoa bien do, chia cua so 3 giay, tao Mel-spectrogram kich thuoc 64x64.
8. Model CNN + BiLSTM du doan cam xuc giong noi theo 4 lop: `binhthuong`, `buon`, `gian`, `vui`.
9. Neu co model khuon mat, he thong du doan them cam xuc tu camera.
10. Module fusion hop nhat hai nguon cam xuc. Neu hai nguon dong y thi lay nhan do. Neu xung dot, he thong uu tien cam xuc tieu cuc hon hoac hoi lai nguoi dung de lam ro.
11. Cam xuc cuoi cung duoc dua vao prompt cua chatbot, ket hop voi profile, lich su gan day va bo nho dai han de tao ngu canh ca nhan hoa.
12. Chatbot phan hoi bang LLM Ollama. Neu cau lenh la giai tri, he thong xu ly truc tiep nhu ke chuyen cuoi hoac mo nhac.
13. Moi luot hoi thoai duoc luu vao session log. Khi ket thuc, he thong cap nhat profile, tom tat dinh ky va bo nho dai han.
14. Nguoi dung co the bat dau phien moi hoac ket thuc chuong trinh.

## Pipeline rut gon de ve tren slide

```mermaid
flowchart LR
    A[Nguoi dung] --> B[Wake word]
    B --> C[Thu am]
    C --> D[Kiem tra chat luong]
    D --> E[Tien xu ly am thanh]
    E --> F[Mel-spectrogram]
    F --> G[Model CNN + BiLSTM]
    G --> H[Cam xuc giong noi]
    A --> I[Camera]
    I --> J[Cam xuc khuon mat]
    H --> K[Fusion cam xuc]
    J --> K
    K --> L[Prompt ca nhan hoa]
    L --> M[LLM / Rule chatbot]
    M --> N[Phan hoi bang text/voice]
    N --> O[Luu lich su va cap nhat profile]
```

## Luong xu ly am thanh, STT va TTS

```mermaid
flowchart TD
    A([Bat dau luot tuong tac bang giong noi]) --> B{Dang cho wake word?}
    B -->|Co| C[Thu am ngan theo cua so wake-listen]
    C --> D[STT bang Google Speech Recognition]
    D --> E{Co nghe thay wake word?}
    E -->|Khong| C
    E -->|Co| F[TTS noi: Minh nghe day]
    B -->|Khong| G[Chuyen thang sang thu am cau noi]
    F --> G

    G --> H[Thu am bang microphone]
    H --> H1[Lay mau 22050 Hz, mono, int16]
    H1 --> H2[Doc tung chunk 0.2 giay]
    H2 --> H3[Tinh RMS de phat hien co giong noi]
    H3 --> H4{Nguoi dung da im lang du lau?}
    H4 -->|Chua| H2
    H4 -->|Roi| I[Xuat waveform raw audio]

    I --> J{Dung cho nhan dien cam xuc}
    J --> K[Kiem tra RMS, peak, noise floor]
    K --> L{Tin hieu hop le?}
    L -->|Khong| M[Bo qua va quay lai che do cho]
    L -->|Co| N[Giam nhieu nhe va chuan hoa]
    N --> O[Chia cua so 3 giay, stride 1.5 giay]
    O --> P[Tao Mel-spectrogram 64x64]
    P --> Q[Model CNN + BiLSTM du doan cam xuc]

    I --> R{Dung cho noi dung hoi thoai}
    R --> S[Dong goi audio thanh SpeechRecognition AudioData]
    S --> T[Google STT, language vi-VN]
    T --> U{Nhan dang duoc van ban?}
    U -->|Khong| V[Tra chuoi rong va yeu cau noi lai]
    U -->|Co| W[Them text nguoi dung vao lich su chat]

    W --> X[LLM Ollama hoac rule tao cau tra loi]
    Q --> X
    X --> Y[In cau tra loi ra man hinh]
    Y --> Z{TTS duoc bat?}
    Z -->|Khong| AA([Ket thuc luot])
    Z -->|Co| AB[pyttsx3 khoi tao engine]
    AB --> AC[Dat toc do doc rate 165]
    AC --> AD[Doc cau tra loi bang loa]
    AD --> AA
```

## Giai thich luong am thanh

1. Wake word: he thong thu cac doan am thanh ngan voi tan so `22050 Hz`, sau do dung STT de kiem tra nguoi dung co goi `mone`, `mone oi`, `aura` hoac `aura oi` hay khong.
2. Thu am cau noi: khi da duoc danh thuc, microphone ghi am theo tung chunk `0.2 giay`. He thong tinh RMS cua moi chunk de biet khi nao nguoi dung bat dau noi va khi nao da im lang.
3. Ket thuc thu am: neu nguoi dung im lang du `3.0 giay` sau khi da noi, hoac dat thoi gian toi da `15.0 giay`, he thong dung thu.
4. Nhan dien cam xuc: waveform duoc chuan hoa, giam nhieu nhe, chia thanh cac cua so `3.0 giay`, tao Mel-spectrogram `64 x 64`, roi dua vao model CNN + BiLSTM.
5. STT cho noi dung chat: cung audio dau vao duoc dong goi thanh `AudioData` va gui qua Google Speech Recognition voi ngon ngu `vi-VN` de lay text nguoi dung.
6. Tao phan hoi: text nguoi dung, cam xuc du doan va ngu canh ca nhan hoa duoc dua vao LLM Ollama hoac rule fallback.
7. TTS: cau tra loi duoc in ra man hinh; neu `VOICE_CHAT_TTS_ENABLED=True`, `pyttsx3` se doc cau tra loi qua loa voi toc do `rate=165`.

## Thong so chinh

- Tan so lay mau khi xu ly am thanh: `22050 Hz`.
- Thoi luong cua so am thanh cho model: `3.0 giay`.
- Buoc truot khi xu ly doan noi dai: `1.5 giay`.
- Kich thuoc anh Mel-spectrogram: `64 x 64`.
- So frame dau vao moi cua so: `10`.
- So lop cam xuc: `4` gom `binhthuong`, `buon`, `gian`, `vui`.
