import os
import copy
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_utils import (
    set_seed,
    load_config,
    get_device,
    get_train_transform,
    get_test_transform,
    get_ram_usage_mb,
    collect_hyperparameters,
    log_training_run,
    load_model_weights,
    plot_learning_curves,
    log_test_metrics,
)

from proto_specified_utils import (
    compute_dataset_prototypes,
    train_one_epoch_proto,
    evaluate_proto,
    save_gradcams_for_predicted_infected_proto,
    resnet_reshape_transform,
)

# =====================================================================
# Execution Configuration
# =====================================================================
CONFIG_PATH = "MVconfig.yaml"

config = load_config(CONFIG_PATH)

MODE = config['MODE']
DATA_ROOT = config["DATA_ROOT_SPLITED"]
DATA_SUBSET = config["DATA_SUBSET"]
IMG_SIZE = config["IMG_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
LEARNING_RATE = float(config["LEARNING_RATE"])
VAL_RATIO = config["VAL_RATIO"]
TEST_RATIO = config["TEST_RATIO"]
NUM_CLASSES = config["NUM_CLASSES"]
INFECTED_CLASS_NAME = config["INFECTED_CLASS_NAME"]
XAI_ENABLE = config["XAI_ENABLE"]

WEIGHT_DECAY = 1e-4

EMBEDDING_DIM = 256
MODEL_NAME = "ProtoNet-ResNet18"
MODEL_TAG = "protonet_resnet18"

# =====================================================================
# Initialization & Setup
# =====================================================================
SEED = 42
set_seed(SEED)

DEVICE = get_device()
print("Using device:", DEVICE, flush=True)

PIN_MEMORY = DEVICE.type == "cuda"

# =====================================================================
# Directory Paths
# =====================================================================
TRAIN_DATA_DIR = os.path.join(DATA_ROOT, "train")
VAL_DATA_DIR = os.path.join(DATA_ROOT, "val")
TEST_DATA_DIR = os.path.join(DATA_ROOT, "test")

RUN_LOG_PATH = str(SCRIPT_DIR / f"{MODEL_TAG}_training_log_{DATA_SUBSET.lower()}.jsonl")
GRADCAM_OUTPUT_DIR = str(
    SCRIPT_DIR / f"gradcam_{MODEL_TAG.upper()}_{DATA_SUBSET.lower()}_infected_predictions"
)
MODEL_SAVE_PATH = str(SCRIPT_DIR / f"{MODEL_TAG}_{DATA_SUBSET.lower()}_salmonscan.pth")
PROTOTYPES_SAVE_PATH = str(SCRIPT_DIR / f"{MODEL_TAG}_{DATA_SUBSET.lower()}_prototypes.pt")
CONFUSION_MATRIX_PATH = str(SCRIPT_DIR / f"{MODEL_TAG}_cm_{DATA_SUBSET.lower()}.png")
LC_PATH =  str(SCRIPT_DIR / f"{MODEL_TAG}_lc_{DATA_SUBSET.lower()}.png")

print("Train dir:", TRAIN_DATA_DIR, flush=True)
print("Val dir:", VAL_DATA_DIR, flush=True)
print("Test dir:", TEST_DATA_DIR, flush=True)
print("Log path:", RUN_LOG_PATH, flush=True)
print("Grad-CAM output dir:", GRADCAM_OUTPUT_DIR, flush=True)
print("Model save path:", MODEL_SAVE_PATH, flush=True)
print("Prototypes save path:", PROTOTYPES_SAVE_PATH, flush=True)

# =====================================================================
# Transforms
# =====================================================================
train_transform = get_train_transform(IMG_SIZE)
eval_transform = get_test_transform(IMG_SIZE)
cam_transform = get_test_transform(IMG_SIZE)

# =====================================================================
# Datasets and DataLoaders
# =====================================================================
train_dataset = datasets.ImageFolder(
    root=TRAIN_DATA_DIR,
    transform=train_transform,
)

val_base_dataset = datasets.ImageFolder(
    root=VAL_DATA_DIR,
    transform=eval_transform,
)

test_base_dataset = datasets.ImageFolder(
    root=TEST_DATA_DIR,
    transform=eval_transform,
)

val_dataset = Subset(
    val_base_dataset,
    list(range(len(val_base_dataset))),
)

test_dataset = Subset(
    test_base_dataset,
    list(range(len(test_base_dataset))),
)

class_names = train_dataset.classes

print("Classes:", class_names, flush=True)
print(
    f"Dataset summary -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}",
    flush=True,
)
print(
    "Total images:",
    len(train_dataset) + len(val_dataset) + len(test_dataset),
    flush=True,
)

if INFECTED_CLASS_NAME not in class_names:
    raise ValueError(
        f"Class '{INFECTED_CLASS_NAME}' not found in dataset classes: {class_names}"
    )

INFECTED_CLASS_IDX = class_names.index(INFECTED_CLASS_NAME)
print("Infected class index:", INFECTED_CLASS_IDX, flush=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=PIN_MEMORY,
)

# DataLoader for computing prototype representations from the training set (no shuffle)
train_eval_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=PIN_MEMORY,
)

# =====================================================================
# ProtoNet Architecture (ResNet18 Backbone)
# =====================================================================
class ProtoNet(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        embeddings = self.backbone(x)
        return embeddings


model = ProtoNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =====================================================================
# Main Execution Pipeline
# =====================================================================
def main():
    prototypes = None

    if MODE == "train":
        print(f"\n--- Running Mode: TRAIN ({MODEL_NAME}) ---", flush=True)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_prototypes = None

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        epoch_times = []
        epoch_ram_mb = []
        epoch_gpu_peak_mb = []

        for epoch in range(NUM_EPOCHS):
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                torch.cuda.synchronize()

            epoch_start = time.perf_counter()

            train_loss, train_acc = train_one_epoch_proto(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=DEVICE,
                num_classes=NUM_CLASSES,
            )

            current_prototypes = compute_dataset_prototypes(
                model=model,
                loader=train_eval_loader,
                device=DEVICE,
                num_classes=NUM_CLASSES,
            )

            val_loss, val_acc, _, _ = evaluate_proto(
                model=model,
                loader=val_loader,
                prototypes=current_prototypes,
                device=DEVICE,
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                gpu_peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
            else:
                gpu_peak_mb = 0.0

            epoch_time = time.perf_counter() - epoch_start
            ram_usage = get_ram_usage_mb()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            epoch_times.append(epoch_time)
            epoch_ram_mb.append(ram_usage)
            epoch_gpu_peak_mb.append(gpu_peak_mb)

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s | "
                f"RAM: {ram_usage:.2f} MB | "
                f"GPU Peak: {gpu_peak_mb:.2f} MB",
                flush=True,
            )

            # Checkpoint selection based on lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_prototypes = current_prototypes.detach().clone()

                print(
                    f"  -> Best model updated. Val Loss: {best_val_loss:.4f} (Val Acc: {best_val_acc:.4f})",
                    flush=True,
                )

        # Restore best model weights and prototypes
        model.load_state_dict(best_model_wts)

        if best_prototypes is None:
            best_prototypes = compute_dataset_prototypes(
                model=model,
                loader=train_eval_loader,
                device=DEVICE,
                num_classes=NUM_CLASSES,
            )

        prototypes = best_prototypes

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        torch.save(prototypes, PROTOTYPES_SAVE_PATH)

        print(f"\nBest Model saved to: {MODEL_SAVE_PATH}", flush=True)
        print(f"Prototypes saved to: {PROTOTYPES_SAVE_PATH}", flush=True)
        print(
            f"Best Validation Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.4f}",
            flush=True,
        )

        plot_learning_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            MODEL_NAME=MODEL_NAME,
            LC_PATH=LC_PATH
        )

        hyperparameters = collect_hyperparameters(
            seed=SEED,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            model=MODEL_NAME,
            model_tag=MODEL_TAG,
            data_subset=DATA_SUBSET,
            device=str(DEVICE),
            train_size=len(train_dataset),
            val_size=len(val_dataset),
            embedding_dim=EMBEDDING_DIM,
            weight_decay=WEIGHT_DECAY,
            optimizer="AdamW",
            pretrained="ImageNet",
        )

        log_training_run(
            hyperparameters=hyperparameters,
            train_accs=train_accs,
            val_accs=val_accs,
            train_losses=train_losses,
            val_losses=val_losses,
            log_path=RUN_LOG_PATH,
            epoch_times=epoch_times,
            epoch_ram_mb=epoch_ram_mb,
            epoch_gpu_peak_mb=epoch_gpu_peak_mb,
        )

    elif MODE == "eval":
        print(f"\n--- Running Mode: EVAL ({MODEL_NAME}) ---", flush=True)

        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"No saved model weights found at: {MODEL_SAVE_PATH}. "
                f"Please train the model first."
            )

        if not os.path.exists(PROTOTYPES_SAVE_PATH):
            raise FileNotFoundError(
                f"No saved prototypes found at: {PROTOTYPES_SAVE_PATH}. "
                f"Please train the model first."
            )

        load_model_weights(
            model=model,
            model_save_path=MODEL_SAVE_PATH,
            device=DEVICE,
        )

        prototypes = torch.load(
            PROTOTYPES_SAVE_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
        prototypes = prototypes.to(DEVICE)

        print(f"Weights successfully loaded from: {MODEL_SAVE_PATH}", flush=True)
        print(f"Prototypes successfully loaded from: {PROTOTYPES_SAVE_PATH}", flush=True)

    else:
        raise ValueError("MODE must be either 'train' or 'eval'.")

    if prototypes is None:
        raise RuntimeError(
            "Prototypes are None. Something went wrong during training/loading."
        )

    # =================================================================
    # Final Test Set Evaluation
    # =================================================================
    test_loss, test_acc, y_true, y_pred = evaluate_proto(
        model=model,
        loader=test_loader,
        prototypes=prototypes,
        device=DEVICE,
    )

    print(f"\n===== Final Test Results: {MODEL_NAME} =====", flush=True)
    print(f"Test Loss: {test_loss:.4f}", flush=True)
    print(f"Test Accuracy: {test_acc:.4f}", flush=True)

    log_test_metrics(
    log_path=RUN_LOG_PATH,
    test_loss=test_loss,
    test_acc=test_acc,
    model_name=MODEL_NAME,
    model_tag=MODEL_TAG,
    data_subset=DATA_SUBSET,
    test_size=len(test_dataset),
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
)

    print("\nClassification Report (Test Set):", flush=True)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (Test Set) - {DATA_SUBSET} - {MODEL_NAME}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")

    # =================================================================
    # Explainable AI (Grad-CAM on Test Set)
    # =================================================================
    target_layers = [model.backbone.layer4[-1]]

    save_gradcams_for_predicted_infected_proto(
        mode = XAI_ENABLE,
        model=model,
        prototypes=prototypes,
        test_dataset=test_dataset,
        full_dataset=test_base_dataset,
        class_names=class_names,
        infected_class_idx=INFECTED_CLASS_IDX,
        output_dir=GRADCAM_OUTPUT_DIR,
        target_layers=target_layers,
        cam_transform=cam_transform,
        device=DEVICE,
        img_size=IMG_SIZE,
        reshape_transform=resnet_reshape_transform,
        clear_previous=True,
        output_prefix=MODEL_TAG,
    )


if __name__ == "__main__":
    main()
