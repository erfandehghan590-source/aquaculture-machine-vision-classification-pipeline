from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Paths
# ==========================
MODEL_PATH = r"..\SalmonScan_YOLOv8n\weights\best.pt"
TEST_PATH = r"..\SalmonScan_Split\val"   # یا test

# ==========================
# Load model
# ==========================
model = YOLO(MODEL_PATH)

# نام کلاس‌ها از روی فولدرها
class_names = sorted([p.name for p in Path(TEST_PATH).iterdir() if p.is_dir()])

class_to_idx = {name: i for i, name in enumerate(class_names)}

y_true = []
y_pred = []

# ==========================
# Prediction
# ==========================
for class_name in class_names:

    folder = Path(TEST_PATH) / class_name

    for img_path in folder.glob("*.*"):

        result = model.predict(
            source=str(img_path),
            verbose=False
        )[0]

        pred = result.probs.top1

        y_true.append(class_to_idx[class_name])
        y_pred.append(pred)

# ==========================
# Metrics
# ==========================
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()