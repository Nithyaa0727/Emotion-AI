# ============================================================
# FULL TRAINING SCRIPT - EfficientNetV2-S on FER2013
# Generates: efficientnet_v2_s_finetuned.pth
# ============================================================

# ── BEFORE RUNNING: install dependencies ────────────────────
# pip install timm albumentations opencv-python scikit-learn matplotlib torch torchvision

import os, random, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── STEP 1: Seed & Device ────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  No GPU found. Training on CPU will be very slow.")

# ── STEP 2: Config ───────────────────────────────────────────
class CFG:
    # ⚠️ UPDATE THESE to your local FER2013 dataset folder paths
    TRAIN_DIR    = 'datasets/FER2013/train'   
    TEST_DIR     = 'datasets/FER2013/test'    
    OUTPUT_DIR   = 'model/'
    SAVE_PATH    = 'model/efficientnet_v2_s_finetuned.pth'
    EMOTIONS     = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    NUM_CLASSES  = 7
    IMG_SIZE     = 224
    BATCH_SIZE   = 32    # Reduce to 16 if you run out of memory
    NUM_EPOCHS   = 50
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTH = 0.1
    MIXUP_ALPHA  = 0.4
    NUM_WORKERS  = 0     # Set to 0 on Windows to avoid DataLoader errors
    MODEL_NAME   = 'efficientnet_v2_s'
    DROPOUT      = 0.3

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# Validate paths exist before starting
assert os.path.exists(CFG.TRAIN_DIR), f"❌ Train folder not found: {CFG.TRAIN_DIR}"
assert os.path.exists(CFG.TEST_DIR),  f"❌ Test folder not found: {CFG.TEST_DIR}"

print("✅ Config ready!")
total_train = sum(len(os.listdir(os.path.join(CFG.TRAIN_DIR, c)))
                  for c in os.listdir(CFG.TRAIN_DIR))
total_test  = sum(len(os.listdir(os.path.join(CFG.TEST_DIR, c)))
                  for c in os.listdir(CFG.TEST_DIR))
print(f"   Train images: {total_train} | Test images: {total_test}")

# ── STEP 3: Transforms ──────────────────────────────────────
train_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── STEP 4: Dataset ──────────────────────────────────────────
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = sorted(os.listdir(root_dir))
        for label_idx, cls_name in enumerate(self.class_names):
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_folder, img_name), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

train_dataset = FER2013Dataset(CFG.TRAIN_DIR, transform=train_transform)
val_dataset   = FER2013Dataset(CFG.TEST_DIR,  transform=val_transform)

# Class weights to handle imbalance
class_counts  = np.array([len(os.listdir(os.path.join(CFG.TRAIN_DIR, c)))
                           for c in sorted(os.listdir(CFG.TRAIN_DIR))])
class_weights = torch.FloatTensor(1.0 / class_counts).to(DEVICE)
class_weights = class_weights / class_weights.sum() * CFG.NUM_CLASSES

train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
                          num_workers=CFG.NUM_WORKERS,
                          pin_memory=torch.cuda.is_available())

print(f"✅ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# ── STEP 5: Mixup ────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ── STEP 6: Label Smoothing Loss ─────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=7, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes
        self.weight = weight

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_val
        log_prob = nn.functional.log_softmax(pred, dim=1)
        if self.weight is not None:
            loss = -(smooth_one_hot * log_prob * self.weight.unsqueeze(0)).sum(dim=1)
        else:
            loss = -(smooth_one_hot * log_prob).sum(dim=1)
        return loss.mean()

# ── STEP 7: Model ────────────────────────────────────────────
class EmotionModel(nn.Module):
    def __init__(self, model_name='efficientnet_v2_s', num_classes=7, dropout=0.3):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),          # index 0
            nn.Linear(in_features, 512),  # index 1
            nn.GELU(),                    # index 2
            nn.Dropout(dropout / 2),      # index 3
            nn.Linear(512, num_classes)   # index 4
        )

    def forward(self, x):
        return self.backbone(x)

model = EmotionModel(CFG.MODEL_NAME, num_classes=CFG.NUM_CLASSES, dropout=CFG.DROPOUT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {CFG.MODEL_NAME} | Params: {total_params/1e6:.1f}M")

# ── STEP 8: Loss / Optimizer / Scheduler ─────────────────────
criterion = LabelSmoothingLoss(classes=CFG.NUM_CLASSES,
                                smoothing=CFG.LABEL_SMOOTH,
                                weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
scheduler = OneCycleLR(optimizer, max_lr=CFG.LR,
                       steps_per_epoch=len(train_loader),
                       epochs=CFG.NUM_EPOCHS,
                       pct_start=0.1, anneal_strategy='cos')
scaler = GradScaler()

# ── STEP 9: Train / Validate Functions ───────────────────────
def train_epoch(model, loader, optimizer, criterion, scheduler, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, y_a, y_b, lam = mixup_data(images, labels, CFG.MIXUP_ALPHA)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (lam * predicted.eq(y_a).sum().item()
                    + (1 - lam) * predicted.eq(y_b).sum().item())
        total += labels.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels

# ── STEP 10: Training Loop ───────────────────────────────────
best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n🚀 Starting Training...")
print(f"   Epochs: {CFG.NUM_EPOCHS} | Batch: {CFG.BATCH_SIZE} | LR: {CFG.LR}")
print("=" * 65)

for epoch in range(CFG.NUM_EPOCHS):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                        criterion, scheduler, scaler)
    val_loss, val_acc, preds, labels_list = validate(model, val_loader, criterion)
    elapsed = time.time() - t0

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'model_name': CFG.MODEL_NAME
        }, CFG.SAVE_PATH)
        marker = '  ✅ BEST SAVED'
    else:
        marker = ''

    print(f"Epoch [{epoch+1:2d}/{CFG.NUM_EPOCHS}] "
          f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
          f"Loss: {val_loss:.4f} | Time: {elapsed:.0f}s{marker}")

print(f"\n🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")
print(f"💾 Model saved to: {CFG.SAVE_PATH}")

# ── STEP 11: Final Evaluation with All Metrics ───────────────
print("\n📊 Loading best model for final evaluation...")
checkpoint = torch.load(CFG.SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

_, _, all_preds, all_labels = validate(model, val_loader, criterion)

print("\n===== FINAL RESULTS =====")
print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
print(f"Precision: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
print("\nPer-class Report:")
print(classification_report(all_labels, all_preds,
                             target_names=CFG.EMOTIONS, zero_division=0))

# ── STEP 12: Training Curves ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'],   label='Val Loss')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history['train_acc'], label='Train Acc')
axes[1].plot(history['val_acc'],   label='Val Acc')
axes[1].set_title('Accuracy Curve')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(CFG.OUTPUT_DIR, 'training_curves.png'), dpi=100)
plt.show()
print(f"✅ Training curves saved to: {CFG.OUTPUT_DIR}training_curves.png")