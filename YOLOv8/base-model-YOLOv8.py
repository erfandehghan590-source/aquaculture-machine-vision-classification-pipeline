from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# رفع خطای WeightsUnpickler در PyTorch 2.6+
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_utils import load_config, get_device
from training_runner import run_single_seed
# ====================== CONFIG & SETUP ======================
CONFIG_PATH = "MVconfig.yaml"
config = load_config(CONFIG_PATH)

MODEL_NAME = "YOLOv8n-cls"
MODEL_TAG = "yolov8n_cls"
DEVICE = get_device()

raw_seeds = config["SEEDS"]
SEEDS = [raw_seeds] if isinstance(raw_seeds, int) else raw_seeds

# ====================== MODEL BUILDER ======================
def build_yolo_model(num_classes: int = 2) -> nn.Module:
    # دانلود یا لود مدل پایه
    yolo = YOLO("yolov8n-cls.pt")
    model = yolo.model
    classify_head = model.model[-1]

    if hasattr(classify_head, 'linear'):
        in_features = classify_head.linear.in_features
        classify_head.linear = nn.Linear(in_features, num_classes)
    elif hasattr(classify_head, 'fc'):
        in_features = classify_head.fc.in_features
        classify_head.fc = nn.Linear(in_features, num_classes)
    else:
        last_layer = classify_head[-1] if isinstance(classify_head, nn.Sequential) else classify_head
        in_features = last_layer.in_features
        if isinstance(classify_head, nn.Sequential):
            classify_head[-1] = nn.Linear(in_features, num_classes)
        else:
            model.model[-1] = nn.Linear(in_features, num_classes)

    return model

# ====================== MAIN EXECUTION ======================
def main():
    mode = config.get("MODE", "train")
    print(f"Device: {DEVICE} | Model: {MODEL_NAME} | Mode: {mode} | Seeds: {SEEDS}", flush=True)

    all_accs = []
    all_losses = []

    for i, seed in enumerate(SEEDS):
        # اجرای پایپ‌لاین ایزوله برای هر Seed
        # در مود train: به صورت خودکار K-Fold روی train اعمال شده و مدل بهینه ذخیره می‌شود
        # در مود eval: وزن‌های مدل لود شده و صرفاً روی TestSet ارزیابی صورت می‌گیرد
        acc, loss = run_single_seed(
            seed=seed,
            config=config,
            model_builder_fn=build_yolo_model,
            model_name=MODEL_NAME,
            model_tag=MODEL_TAG,
            script_dir=SCRIPT_DIR,
            device=DEVICE,
            is_first_seed=(i == 0),
        )
        all_accs.append(acc)
        all_losses.append(loss)

    # نمایش خلاصه آماری چند Seed
    if len(all_accs) > 1:
        mean_acc = np.mean(all_accs) * 100
        std_acc = np.std(all_accs) * 100
        mean_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)

        print("\n" + "=" * 55)
        print(f"FINAL SUMMARY: {MODEL_NAME} (Over {len(SEEDS)} Seeds)")
        print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"Test Loss:     {mean_loss:.4f} ± {std_loss:.4f}")
        print("=" * 55)


if __name__ == "__main__":
    main()
