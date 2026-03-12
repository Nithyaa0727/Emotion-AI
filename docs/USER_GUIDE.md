# Emotion AI — User Guide

## Getting Started

### 1. Register / Login
- Open http://localhost:5000
- Click **Sign Up** to create an account
- Enter username and password
- Click **Login** to access the app

---

### 2. Live Webcam Mode

1. Click **"Live Webcam"** button on the home screen
2. Allow browser camera access when prompted
3. Position your face in front of the camera
4. The app will:
   - Draw a colored box around your face
   - Show detected emotion label
   - Display confidence percentage
   - Show probability bars for all 7 emotions

**Tips for best accuracy:**
- Face the camera directly
- Ensure good lighting (avoid backlight)
- Keep face clearly visible, not too far away
- Remove sunglasses or heavy accessories

---

### 3. Image Upload Mode

1. Click **"Upload Image"** button
2. Click the upload area or drag & drop a photo
3. Supported formats: JPG, PNG, JPEG
4. Click **"Detect Emotion"**
5. View the prediction result with:
   - Detected emotion and emoji
   - Confidence score
   - Probability chart for all emotions

**Tips for best results:**
- Use clear, front-facing face photos
- Avoid group photos (uses first detected face)
- Good resolution images work better
- Well-lit photos give more accurate results

---

### 4. Understanding Results

**Confidence Score:**
- 90-100% = Very confident prediction
- 70-89%  = Confident prediction
- 50-69%  = Moderate confidence
- Below 50% = Low confidence (expression may be ambiguous)

**Probability Bars:**
- Shows likelihood for all 7 emotions
- Longer bar = higher probability
- Numbers show exact percentage

---

### 5. Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam black screen | Allow camera in browser settings |
| No face detected | Move closer, improve lighting |
| Wrong emotion detected | Exaggerate the expression |
| App loads slowly | First load downloads model weights, wait 30s |
| Page not loading | Check http://localhost:5000 (not https) |

---

## Privacy

- All processing happens **locally on your computer**
- No images or video are sent to any server
- Your face data never leaves your device
- User accounts are stored locally in `emotion_ai.db`
