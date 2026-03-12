# 🎭 Emotion AI
### Real-Time Facial Emotion Recognition System

---

## 📦 What's Inside This Package

```
EmotionAI_Client/
│
├── 📄 README.md                    ← You are here
├── 📄 requirements.txt             ← Python packages needed
│
├── 📁 app/                         ← Web application
│   ├── app.py                      ← Main backend server
│   ├── emotion_ai.db               ← Auto-created user database
│   ├── templates/
│   │   └── index.html              ← Frontend UI
│   └── static/                     ← CSS, JS, images
│
├── 📁 model/                       
│   └── efficientnet_v2_s_finetuned.pth
│
├── 📁 scripts/                     ← One-click launchers
│   ├── setup_and_run.bat           ← Windows (double-click)
│   └── setup_and_run.sh            ← Mac / Linux
│
└── 📁 docs/                        ← Documentation
    └── USER_GUIDE.md
```

---

## ⚡ Quick Start (5 Minutes)

### Step 1 — Run the App

**Windows:**
```
Double-click → scripts/setup_and_run.bat
```

**Mac / Linux:**
```bash
bash scripts/setup_and_run.sh
```

**Manual (any OS):**
```bash
pip install -r requirements.txt
cd app
python app.py
```

### Step 2 — Open Browser
```
http://localhost:5000
```

---

## 🖥️ System Requirements

| Requirement | Minimum          | Recommended      |
|-------------|------------------|------------------|
| OS          | Windows 10 / Mac / Ubuntu | Any modern OS |
| Python      | 3.10+            | 3.11             |
| RAM         | 4 GB             | 8 GB             |
| CPU         | Any modern       | Intel i5 / Ryzen 5 |
| GPU         | Not required     | NVIDIA (faster)  |
| Webcam      | Required for live mode | HD webcam |
| Storage     | 2 GB free        | 5 GB             |

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 📷 Live Webcam | Real-time emotion detection from camera |
| 🖼️ Image Upload | Detect emotion from any face photo |
| 📊 Probability Bars | See confidence % for all 7 emotions |
| 🔐 User Auth | Register & login with secure passwords |
| 🤖 XAI Ready | GradCAM, LIME, SHAP explainability built-in |
| ⚡ Fast | ~50ms per frame on CPU, ~10ms on GPU |

---

## 😊 Supported Emotions

| Emotion  | Emoji |
|----------|-------|
| Angry    | 😠    |
| Disgust  | 🤢    |
| Fear     | 😨    |
| Happy    | 😊    |
| Neutral  | 😐    |
| Sad      | 😢    |
| Surprise | 😲    |

---

## 🧠 Model Information

| Property       | Value                        |
|----------------|------------------------------|
| Architecture   | EfficientNetV2-S             |
| Dataset        | FER2013 (35,887 images)      |
| Test Accuracy  | 72.95%                       |
| Input Size     | 224 × 224 px                 |
| Framework      | PyTorch 2.0 + timm           |

---

## 🔧 Troubleshooting

**App won't start:**
```
Make sure Python 3.10+ is installed
Run: python --version
```

**Model not loading:**
```
Check model/ folder has: efficientnet_v2_s_finetuned.pth
File size should be ~85 MB
```

**Webcam not working:**
```
Allow camera access in browser
Try a different browser (Chrome recommended)
```

**Slow predictions:**
```
Normal on CPU — ~1-2 seconds per frame
Use GPU for real-time speed
```

**Port already in use:**
```bash
# Change port in app/app.py last line:
app.run(port=5001)   # try 5001 or 8080
```

---

## 🔗 API Endpoints (For Developers)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict_image` | POST | Upload image → get emotion |
| `/predict_frame` | POST | Base64 frame → annotated frame |
| `/register` | POST | Create user account |
| `/login` | POST | Authenticate user |
| `/health` | GET | Check server status |

**Example API call:**
```bash
curl -X POST http://localhost:5000/predict_image \
  -F "image=@your_face.jpg"
```

**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.94,
  "emoji": "😊",
  "color": "#2ECC71",
  "all_probs": {
    "Angry": 0.01,
    "Happy": 0.94,
    ...
  }
}
```
