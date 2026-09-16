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
from proto_specified_utils import (
    convnext_reshape_transform,
)
from training_runner import run_single_seed_proto

# =====================================================================
# Execution Configuration & Setup
# =====================================================================
CONFIG_PATH = "MVconfig.yaml"
config = load_config(CONFIG_PATH)

# اطمینان از وجود کلیدهای EMBEDDING_DIM و WEIGHT_DECAY در کانفیگ (جهت خوانش داخلی توسط تابع)
config.setdefault("EMBEDDING_DIM", 256)
config.setdefault("WEIGHT_DECAY", 1e-4)

MODEL_NAME = "ProtoNet-ConvNeXt-Tiny"
MODEL_TAG = "protonet_convnext_tiny"
DEVICE = get_device()

raw_seeds = config["SEEDS"]
SEEDS = [raw_seeds] if isinstance(raw_seeds, int) else raw_seeds

# =====================================================================
# ProtoNet Architecture & Target Layers Definition
# =====================================================================
class ProtoNet(nn.Module):
    """معماری بر پایه نگاشت ویژگی‌های ConvNeXt-Tiny به فضای متریک"""
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        embeddings = self.backbone(x)
        return embeddings


def get_convnext_target_layers(model: nn.Module):
    """تعیین آخرین بلوک نقشه ویژگی برای استخراج گرادیان Grad-CAM"""
    return [model.backbone.features[-1][-1]]


# =====================================================================
# Main Multi-Seed Execution Pipeline
# =====================================================================
def main():
    print(f"Device: {DEVICE} | Model: {MODEL_NAME} | Seeds: {SEEDS}", flush=True)

    test_accs = []
    test_losses = []

    for i, seed in enumerate(SEEDS):
        acc, loss = run_single_seed_proto(
            seed=seed,
            config=config,
            model_builder_fn=ProtoNet,
            model_name=MODEL_NAME,
            model_tag=MODEL_TAG,
            script_dir=SCRIPT_DIR,
            device=DEVICE,
            target_layers_fn=get_convnext_target_layers,
            reshape_transform=convnext_reshape_transform,
            is_first_seed=(i == 0),
        )
        test_accs.append(acc)
        test_losses.append(loss)

    if len(SEEDS) > 1:
        mean_acc = np.mean(test_accs) * 100
        std_acc = np.std(test_accs) * 100
        mean_loss = np.mean(test_losses)
        std_loss = np.std(test_losses)

        print("\n" + "=" * 60, flush=True)
        print(f"FINAL MULTI-SEED SUMMARY: {MODEL_NAME} ({len(SEEDS)} Seeds: {SEEDS})", flush=True)
        print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%", flush=True)
        print(f"Test Loss:     {mean_loss:.4f} ± {std_loss:.4f}", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
