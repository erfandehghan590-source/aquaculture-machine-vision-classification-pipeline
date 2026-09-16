from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# =====================================================================
# Setup Project Root & Imports
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_utils import load_config, get_device
from training_runner import run_single_seed
# =====================================================================
# Execution Configuration & Setup
# =====================================================================
CONFIG_PATH = "MVconfig.yaml"
config = load_config(CONFIG_PATH)

MODEL_NAME = "ConvNeXt-Tiny"
MODEL_TAG = "convnext_tiny"
DEVICE = get_device()

raw_seeds = config["SEEDS"]
SEEDS = [raw_seeds] if isinstance(raw_seeds, int) else raw_seeds

# =====================================================================
# Model Architecture Builder
# =====================================================================
def build_convnext_tiny_model(num_classes: int = 2) -> nn.Module:
    """بارگذاری ConvNeXt-Tiny پیش‌آموزش‌دیده و تنظیم لایه طبقه‌بندی نهایی"""
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    # در معماری ConvNeXt لایه خطی نهایی در classifier[2] قرار دارد
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
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
            model_builder_fn=build_convnext_tiny_model,
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
