"""
Emotion AI — Flask Web Application (v2)
Author : Alliance University Capstone Team
Model  : EfficientNetV2-S (72.95% on FER2013)
"""

import os
import base64
import sqlite3
import json
import io
from datetime import datetime, timedelta
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_file)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'emotionai_secret_2024_v2')
app.permanent_session_lifetime = timedelta(days=7)

# ── Constants ────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE  = 224
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model',
                          'efficientnet_v2_s_finetuned.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'efficientnet_v2_s_finetuned.pth'

EMOTION_COLORS = {
    'Angry':    '#FF4444', 'Disgust': '#8B4513',
    'Fear':     '#9B59B6', 'Happy':   '#2ECC71',
    'Neutral':  '#95A5A6', 'Sad':     '#3498DB',
    'Surprise': '#F39C12',
}
EMOTION_EMOJIS = {
    'Angry':    '😠', 'Disgust': '🤢', 'Fear':    '😨',
    'Happy':    '😊', 'Neutral': '😐', 'Sad':     '😢',
    'Surprise': '😲',
}

# ── Model ────────────────────────────────────────────────────
class EffNetV2(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        in_f = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 512),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model():
    m = EffNetV2().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(ckpt['model_state_dict'])
        acc = ckpt.get('val_acc', 0) * 100
        print(f'✅ Model loaded — Val Accuracy: {acc:.2f}%')
    else:
        print(f'⚠️  Model not found at {MODEL_PATH}')
    m.eval()
    return m


model = load_model()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ── GradCAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model     = model
        self.gradients = None
        self.activations= None
        # hook last conv block of EfficientNetV2-S
        target = model.backbone.blocks[-1]
        target.register_forward_hook(self._save_activation)
        target.register_backward_hook(self._save_gradient)

    def _save_activation(self, mod, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, mod, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam    = (weights * self.activations).sum(dim=1, keepdim=True)
        cam    = F.relu(cam)
        cam    = cam.squeeze().cpu().numpy()
        cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAM(model)
_last_face_rgb = None


def generate_gradcam_image(img_rgb: np.ndarray, class_idx: int) -> str:
    """Returns base64 PNG of GradCAM overlay."""
    tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    cam = gradcam.generate(tensor, class_idx)
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_rgb.astype(np.float32) +
               0.45 * heatmap_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return 'data:image/png;base64,' + base64.b64encode(buf).decode()


# ── Database ─────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), 'emotion_ai.db')


def get_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY,
                username   TEXT UNIQUE NOT NULL,
                password   TEXT NOT NULL,
                created_at TEXT
            )''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY,
                username   TEXT NOT NULL,
                emotion    TEXT NOT NULL,
                confidence REAL NOT NULL,
                all_probs  TEXT,
                source     TEXT DEFAULT "webcam",
                timestamp  TEXT NOT NULL
            )''')
        conn.commit()


init_db()

# ── Auth decorator ───────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            if request.is_json or request.path.startswith('/predict') or request.path.startswith('/export'):
                return jsonify({'error': 'unauthorized', 'message': 'Login required'}), 401
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ── Helpers ──────────────────────────────────────────────────
def predict_emotion(img_rgb: np.ndarray) -> dict:
    tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return {
        'emotion':    EMOTIONS[idx],
        'confidence': float(probs[idx]),
        'all_probs':  {e: float(p) for e, p in zip(EMOTIONS, probs)},
        'emoji':      EMOTION_EMOJIS[EMOTIONS[idx]],
        'color':      EMOTION_COLORS[EMOTIONS[idx]],
        'class_idx':  idx,
    }


def detect_face(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    passes = [
        (1.10, 5, (48, 48)), (1.05, 4, (40, 40)),
        (1.05, 3, (30, 30)), (1.03, 2, (24, 24)),
    ]
    for sf, mn, ms in passes:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=sf, minNeighbors=mn, minSize=ms)
        if len(faces):
            return max(faces, key=lambda r: r[2] * r[3])
    eq = cv2.equalizeHist(gray)
    for sf, mn, ms in passes:
        faces = face_cascade.detectMultiScale(
            eq, scaleFactor=sf, minNeighbors=mn, minSize=ms)
        if len(faces):
            return max(faces, key=lambda r: r[2] * r[3])
    return None


def process_frame(frame: np.ndarray):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    results = []
    for (x, y, w, h) in faces:
        pad = int(0.1 * w)
        x1  = max(0, x - pad); y1 = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)
        face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        result   = predict_emotion(face_rgb)
        result['bbox'] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        results.append(result)
        hex_col = result['color'].lstrip('#')
        bgr     = tuple(int(hex_col[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
        label = f"{result['emotion']} {result['confidence']*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame, results


def save_prediction(username, emotion, confidence, all_probs, source='webcam'):
    with get_db() as conn:
        conn.execute(
            'INSERT INTO predictions (username, emotion, confidence, all_probs, source, timestamp) VALUES (?,?,?,?,?,?)',
            (username, emotion, confidence, json.dumps(all_probs), source,
             datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    logged_in = 'username' in session
    return render_template('index.html',
                           emotion_colors=EMOTION_COLORS,
                           emotion_emojis=EMOTION_EMOJIS,
                           logged_in=logged_in,
                           username=session.get('username', ''))


@app.route('/predict_image', methods=['POST'])
@login_required
def predict_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400
    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400
    face = detect_face(img)
    if face is None:
        return jsonify({
            'error':   'no_face',
            'message': 'No human face detected. Please upload a clear, well-lit, front-facing photo.'
        }), 422

    x, y, w, h = face

    # ── NEW: reject if face region is too small relative to image ──
    img_area  = img.shape[0] * img.shape[1]
    face_area = w * h
    if face_area / img_area < 0.03:          # face < 3% of image → likely a false positive
        return jsonify({
            'error':   'no_face',
            'message': 'No clear face found. Make sure your face is visible and well-lit.'
        }), 422

    img_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    global _last_face_rgb
    _last_face_rgb = img_rgb.copy()
    result  = predict_emotion(img_rgb)

    # ── NEW: reject low-confidence predictions ──
    if result['confidence'] < 0.55:
        return jsonify({
            'error':   'low_confidence',
            'message': 'Could not confidently detect an emotion. Please use a clearer, front-facing photo.'
        }), 422

    gradcam_b64 = generate_gradcam_image(img_rgb, result['class_idx'])
    result['gradcam'] = gradcam_b64
    save_prediction(session['username'], result['emotion'],
                    result['confidence'], result['all_probs'], source='upload')
    return jsonify(result)

@app.route('/predict_frame', methods=['POST'])
@login_required
def predict_frame():
    data = request.json.get('frame', '')
    if not data:
        return jsonify({'error': 'No frame'}), 400
    img_data  = base64.b64decode(data.split(',')[1])
    arr       = np.frombuffer(img_data, np.uint8)
    frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    out, res  = process_frame(frame)
    _, buf    = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

    gradcam_b64 = None
    if res:
        r = res[0]
        # GradCAM on first face
        x, y, w, h = r['bbox']
        face_bgr = out[y:y+h, x:x+w] if w > 0 and h > 0 else frame[:48, :48]
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        global _last_face_rgb
        _last_face_rgb = face_rgb.copy()
        try:
            gradcam_b64 = generate_gradcam_image(face_rgb, r['class_idx'])
        except Exception:
            gradcam_b64 = None
        save_prediction(session['username'], r['emotion'],
                        r['confidence'], r['all_probs'], source='webcam')

    return jsonify({
        'frame':    frame_b64,
        'results':  res,
        'count':    len(res),
        'no_face':  len(res) == 0,
        'gradcam':  gradcam_b64,
    })


@app.route('/register', methods=['POST'])
def register():
    d  = request.json or {}
    un = d.get('username', '').strip()
    pw = d.get('password', '').strip()
    if not un or not pw:
        return jsonify({'error': 'Username and password required'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO users (username, password, created_at) VALUES (?,?,?)',
                (un, generate_password_hash(pw),
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already taken'}), 400


@app.route('/login', methods=['POST'])
def login():
    d  = request.json or {}
    un = d.get('username', '').strip()
    pw = d.get('password', '').strip()
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM users WHERE username=?', (un,)).fetchone()
    if row and check_password_hash(row[2], pw):
        session.permanent = True
        session['username'] = un
        return jsonify({'success': True, 'username': un})
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/dashboard')
@login_required
def dashboard_data():
    un = session['username']
    with get_db() as conn:
        rows = conn.execute(
            'SELECT emotion, confidence, source, timestamp FROM predictions '
            'WHERE username=? ORDER BY timestamp DESC LIMIT 200', (un,)
        ).fetchall()
    preds = [{'emotion': r[0], 'confidence': r[1],
               'source': r[2], 'timestamp': r[3]} for r in rows]
    # Aggregations
    emotions_list = [p['emotion'] for p in preds]
    counts = Counter(emotions_list)
    # Timeline — last 20
    timeline = preds[:20][::-1]
    # Avg confidence per emotion
    conf_by_emotion = {}
    for p in preds:
        conf_by_emotion.setdefault(p['emotion'], []).append(p['confidence'])
    avg_conf = {e: round(sum(v)/len(v)*100, 1) for e, v in conf_by_emotion.items()}
    return jsonify({
        'total':      len(preds),
        'counts':     dict(counts),
        'timeline':   timeline,
        'avg_conf':   avg_conf,
        'recent':     preds[:10],
    })


@app.route('/export/pdf')
@login_required
def export_pdf():
    un = session['username']
    with get_db() as conn:
        rows = conn.execute(
            'SELECT emotion, confidence, source, timestamp FROM predictions '
            'WHERE username=? ORDER BY timestamp DESC LIMIT 100', (un,)
        ).fetchall()
        user_row = conn.execute(
            'SELECT created_at FROM users WHERE username=?', (un,)
        ).fetchone()

    preds = [{'emotion': r[0], 'confidence': r[1],
               'source': r[2], 'timestamp': r[3]} for r in rows]

    counts = Counter(p['emotion'] for p in preds)
    total  = len(preds)
    top_emotion = max(counts.items(), key=lambda x: x[1])[0] if counts else 'N/A'
    webcam_count = sum(1 for p in preds if p['source'] == 'webcam')
    upload_count = sum(1 for p in preds if p['source'] == 'upload')
    avg_conf = round(sum(p['confidence'] for p in preds) / total * 100, 1) if total else 0
    joined = user_row[0][:10] if user_row and user_row[0] else 'N/A'

    EMOTION_COLORS_HEX = {
        'Angry': '#ef4444', 'Disgust': '#a16207', 'Fear': '#9B59B6',
        'Happy': '#22c55e', 'Neutral': '#94a3b8', 'Sad': '#3b82f6', 'Surprise': '#f59e0b'
    }
    EMOTION_EMOJIS_MAP = {
        'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
        'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
    }

    # Distribution rows
    dist_rows = ''
    for e, c in sorted(counts.items(), key=lambda x: -x[1]):
        pct = c / total * 100 if total else 0
        color = EMOTION_COLORS_HEX.get(e, '#6366f1')
        emoji = EMOTION_EMOJIS_MAP.get(e, '')
        dist_rows += f"""
        <tr>
          <td><span class="emoji">{emoji}</span> <span style="color:{color};font-weight:600">{e}</span></td>
          <td>{c}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
            </div>
          </td>
          <td style="color:{color};font-weight:600">{pct:.1f}%</td>
        </tr>"""

    # Prediction rows
    pred_rows = ''
    for i, p in enumerate(preds[:50], 1):
        color = EMOTION_COLORS_HEX.get(p['emotion'], '#6366f1')
        emoji = EMOTION_EMOJIS_MAP.get(p['emotion'], '')
        src_badge = (
            '<span class="badge badge-webcam">📷 webcam</span>'
            if p['source'] == 'webcam'
            else '<span class="badge badge-upload">🖼 upload</span>'
        )
        conf_pct = p['confidence'] * 100
        pred_rows += f"""
        <tr>
          <td class="num">{i}</td>
          <td><span class="emoji">{emoji}</span> <span style="color:{color};font-weight:600">{p['emotion']}</span></td>
          <td>
            <div class="conf-wrap">
              <div class="conf-fill" style="width:{conf_pct:.1f}%;background:{color}"></div>
              <span class="conf-label">{conf_pct:.1f}%</span>
            </div>
          </td>
          <td>{src_badge}</td>
          <td class="ts">{p['timestamp']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
        <title>EmotionAI Report — {un}</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
        <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --indigo: #6366f1;
            --cyan:   #22d3ee;
            --bg:     #07090f;
            --card:   rgba(255,255,255,0.03);
            --border: rgba(255,255,255,0.08);
            --text:   #e2e8f0;
            --muted:  #64748b;
            --dim:    #1e293b;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 0;
        }}

        /* ── Print bar ── */
        .print-bar {{
            position: fixed; top: 0; left: 0; right: 0; z-index: 100;
            background: rgba(7,9,15,0.92);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; justify-content: space-between;
            padding: 12px 32px;
        }}
        .print-bar-brand {{ display:flex; align-items:center; gap:10px; }}
        .print-bar-brand span {{ font-weight:700; font-size:18px;
            background:linear-gradient(135deg,#818cf8,#22d3ee);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
        .print-btn {{
            display: flex; align-items: center; gap: 8px;
            background: linear-gradient(135deg, #6366f1, #22d3ee);
            color: white; border: none; padding: 10px 24px;
            border-radius: 12px; font-size: 14px; font-weight: 600;
            cursor: pointer; font-family: 'Outfit', sans-serif;
            transition: opacity 0.2s; box-shadow: 0 4px 20px rgba(99,102,241,0.35);
        }}
        .print-btn:hover {{ opacity: 0.88; }}

        /* ── Page ── */
        .page {{
            max-width: 900px; margin: 0 auto;
            padding: 100px 24px 60px;
        }}

        /* ── Hero header ── */
        .report-header {{
            text-align: center; margin-bottom: 40px;
        }}
        .report-header .brain {{
            width: 64px; height: 64px; border-radius: 20px;
            background: rgba(99,102,241,0.12);
            border: 1px solid rgba(99,102,241,0.25);
            display: flex; align-items: center; justify-content: center;
            font-size: 32px; margin: 0 auto 16px;
        }}
        .report-header h1 {{
            font-size: 32px; font-weight: 800; letter-spacing: -0.5px;
            background: linear-gradient(135deg,#818cf8,#22d3ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .report-header .meta {{
            font-size: 13px; color: var(--muted); line-height: 1.8;
        }}
        .report-header .meta b {{ color: var(--text); }}

        /* ── Section headings ── */
        .section-title {{
            font-size: 11px; font-weight: 700; color: var(--muted);
            text-transform: uppercase; letter-spacing: 2px;
            margin: 36px 0 14px;
            display: flex; align-items: center; gap: 8px;
        }}
        .section-title::after {{
            content: ''; flex: 1; height: 1px; background: var(--border);
        }}

        /* ── Stat cards ── */
        .stats-grid {{
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
            margin-bottom: 8px;
        }}
        .stat-card {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; padding: 18px; text-align: center;
        }}
        .stat-val {{
            font-size: 30px; font-weight: 800; line-height: 1;
            background: linear-gradient(135deg,#818cf8,#22d3ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .stat-val.happy  {{ background: linear-gradient(135deg,#22c55e,#86efac); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
        .stat-val.cyan   {{ background: linear-gradient(135deg,#22d3ee,#67e8f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
        .stat-val.amber  {{ background: linear-gradient(135deg,#f59e0b,#fcd34d); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
        .stat-label {{ font-size: 11px; color: var(--muted); font-weight: 500; }}

        /* ── Tables ── */
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        thead tr {{ background: rgba(99,102,241,0.1); border-bottom: 1px solid rgba(99,102,241,0.2); }}
        thead th {{
            padding: 12px 14px; text-align: left;
            font-size: 10px; font-weight: 700; color: #a5b4fc;
            text-transform: uppercase; letter-spacing: 1px;
        }}
        tbody tr {{ border-bottom: 1px solid var(--border); transition: background 0.15s; }}
        tbody tr:hover {{ background: rgba(255,255,255,0.02); }}
        tbody td {{ padding: 10px 14px; vertical-align: middle; }}
        td.num {{ color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 11px; width: 36px; }}
        td.ts  {{ color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
        .emoji {{ font-size: 16px; margin-right: 4px; }}

        /* ── Bar cells ── */
        .bar-wrap {{
            background: rgba(255,255,255,0.05); border-radius: 99px;
            height: 6px; width: 140px; overflow: hidden;
        }}
        .bar-fill {{ height: 100%; border-radius: 99px; }}
        .conf-wrap {{ display:flex; align-items:center; gap:8px; }}
        .conf-fill {{ height: 5px; border-radius:99px; min-width:2px; }}
        .conf-label {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text); white-space:nowrap; }}

        /* ── Badges ── */
        .badge {{
            display: inline-flex; align-items: center; gap: 4px;
            padding: 3px 10px; border-radius: 99px;
            font-size: 11px; font-weight: 500;
        }}
        .badge-webcam {{ background: rgba(34,211,238,0.1); color: #22d3ee; border: 1px solid rgba(34,211,238,0.2); }}
        .badge-upload  {{ background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }}

        /* ── Card wrapper ── */
        .card {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 16px; overflow: hidden;
        }}

        /* ── Footer ── */
        .footer {{
            text-align: center; margin-top: 48px; padding-top: 24px;
            border-top: 1px solid var(--border);
            font-size: 12px; color: var(--muted); line-height: 2;
        }}
        .footer .grad {{
            background: linear-gradient(135deg,#818cf8,#22d3ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}

        /* ── Print styles ── */
        @media print {{
            body {{ background: white; color: #1e293b; }}
            .print-bar {{ display: none !important; }}
            .page {{ padding-top: 24px; }}
            .card {{ border: 1px solid #e2e8f0; }}
            thead tr {{ background: #6366f1 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            thead th {{ color: white !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            .stat-card {{ border: 1px solid #e2e8f0; background: #f8fafc; }}
            .stat-val, .report-header h1, .footer .grad, .print-bar-brand span {{
            background: none !important; -webkit-text-fill-color: #6366f1 !important;
            }}
            .bar-wrap {{ background: #e2e8f0; }}
            .badge-webcam {{ background: #e0f2fe; color: #0284c7; }}
            .badge-upload  {{ background: #fef3c7; color: #b45309; }}
            tbody tr:hover {{ background: transparent; }}
        }}
        </style>
        </head>
        <body>

        <!-- Print bar -->
        <div class="print-bar no-print">
        <div class="print-bar-brand">
            <span style="font-size:22px">🧠</span>
            <span>EmotionAI</span>
        </div>
        <button class="print-btn" onclick="window.print()">
            🖨️ &nbsp;Save as PDF
        </button>
        </div>

        <div class="page">

        <!-- Header -->
        <div class="report-header">
            <div class="brain">🧠</div>
            <h1>EmotionAI Session Report</h1>
            <div class="meta">
            User: <b>{un}</b> &nbsp;·&nbsp;
            Member since: <b>{joined}</b> &nbsp;·&nbsp;
            Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b><br/>
            Alliance University Capstone 2025 &nbsp;·&nbsp;
            EfficientNetV2-S &nbsp;·&nbsp; FER2013 · 72.95%
            </div>
        </div>

        <!-- Stats -->
        <div class="section-title">📊 Summary</div>
        <div class="stats-grid">
            <div class="stat-card">
            <div class="stat-val">{total}</div>
            <div class="stat-label">Total Predictions</div>
            </div>
            <div class="stat-card">
            <div class="stat-val happy">{top_emotion}</div>
            <div class="stat-label">Top Emotion</div>
            </div>
            <div class="stat-card">
            <div class="stat-val cyan">{avg_conf}%</div>
            <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
            <div class="stat-val amber">{webcam_count} / {upload_count}</div>
            <div class="stat-label">Webcam / Uploads</div>
            </div>
        </div>

        <!-- Emotion Distribution -->
        <div class="section-title">🎭 Emotion Distribution</div>
        <div class="card">
            <table>
            <thead>
                <tr>
                <th>Emotion</th>
                <th>Count</th>
                <th>Distribution</th>
                <th>Percentage</th>
                </tr>
            </thead>
            <tbody>{dist_rows}</tbody>
            </table>
        </div>

        <!-- Recent Predictions -->
        <div class="section-title">🕐 Recent Predictions (Last 50)</div>
        <div class="card">
            <table>
            <thead>
                <tr>
                <th>#</th>
                <th>Emotion</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>{pred_rows}</tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="footer">
            <span class="grad">EmotionAI</span> &nbsp;·&nbsp; Alliance University Capstone 2025<br/>
            Pavana &nbsp;·&nbsp; Likhitha &nbsp;·&nbsp; Ankita &nbsp;·&nbsp; Nithya<br/>
            PyTorch 2.0 &nbsp;·&nbsp; timm &nbsp;·&nbsp; EfficientNetV2-S
        </div>

        </div>
        </body>
        </html>"""

    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route('/session_user')
def session_user():
    if 'username' in session:
        return jsonify({'username': session['username']})
    return jsonify({'username': None}), 401

@app.route('/generate_lime', methods=['POST'])
@login_required
def generate_lime():
    global _last_face_rgb
    if _last_face_rgb is None:
        return jsonify({'error': 'No face cached yet'}), 400
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        explainer = lime_image.LimeImageExplainer()
        def predict_fn(imgs):
            results = []
            for img in imgs:
                t = transform(image=img.astype(np.uint8))['image'].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    p = F.softmax(model(t), dim=1)[0].cpu().numpy()
                results.append(p)
            return np.array(results)
        explanation = explainer.explain_instance(
            _last_face_rgb, predict_fn, top_labels=1, num_samples=500)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=False,
            num_features=10, hide_rest=False)
        lime_img = mark_boundaries(temp / 255.0, mask)
        lime_bgr = cv2.cvtColor((lime_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.png', lime_bgr)
        b64 = 'data:image/png;base64,' + base64.b64encode(buf).decode()
        return jsonify({'lime': b64})
    except ImportError:
        return jsonify({'error': 'lime not installed. Run: pip install lime'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_shap', methods=['POST'])
@login_required
def generate_shap():
    global _last_face_rgb
    if _last_face_rgb is None:
        return jsonify({'error': 'No face cached yet'}), 400
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tensor = transform(image=_last_face_rgb)['image'].unsqueeze(0).to(DEVICE)
        for m in model.modules():
            if isinstance(m, nn.SiLU):
                m.inplace = False
        bg = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        e = shap.GradientExplainer(model, bg)
        shap_vals = e.shap_values(tensor)
        fig, axes = plt.subplots(1, 7, figsize=(21, 3), facecolor='#07090f')
        for i, emotion in enumerate(EMOTIONS):
            sv = shap_vals[i][0].transpose(1, 2, 0)
            sv_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
            axes[i].imshow(sv_norm, cmap='RdBu_r', vmin=0, vmax=1)
            axes[i].set_title(emotion, color='white', fontsize=8)
            axes[i].axis('off')
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='#07090f', dpi=100)
        plt.close()
        buf.seek(0)
        b64 = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()
        return jsonify({'shap': b64})
    except ImportError:
        return jsonify({'error': 'shap not installed. Run: pip install shap matplotlib'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status':   'ok',
        'device':   str(DEVICE),
        'model':    os.path.basename(MODEL_PATH),
        'emotions': EMOTIONS,
    })


if __name__ == '__main__':
    print('🚀  Emotion AI v2 starting...')
    print(f'    Device : {DEVICE}')
    print(f'    Model  : {MODEL_PATH}')
    print('    URL    : http://localhost:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)