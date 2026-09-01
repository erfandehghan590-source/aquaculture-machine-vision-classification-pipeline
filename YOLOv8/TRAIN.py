import gc
import yaml
import json
import time
import psutil
from datetime import datetime

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from types import MethodType

# ======================================
# LOAD CONFIG
# ======================================

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

CONFIG_PATH = "../MVconfig.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def cfg(key, cast=None):
    if key not in config:
        raise KeyError(f"{key} not found in {CONFIG_PATH}")
    value = config[key]
    return cast(value) if cast else value


# ======================================
# CONFIG
# ======================================

DATA_ROOT = "../" + cfg("DATA_ROOT_SPLITED")

IMG_SIZE = cfg("IMG_SIZE", int)
BATCH_SIZE = cfg("BATCH_SIZE", int)
EPOCHS = cfg("NUM_EPOCHS", int)

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# GLOBAL HISTORY
# ============================================================

history = []

epoch_start = 0


# ============================================================
# EPOCH START
# ============================================================

def on_train_epoch_start(trainer):

    global epoch_start

    epoch_start = time.time()

    # Reset train accuracy counters
    trainer.train_correct = 0
    trainer.train_total = 0

    # Reset GPU peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# CUSTOM LOSS
# ============================================================
#
# IMPORTANT:
#
# This function replaces ClassificationModel.loss().
#
# The original YOLOv8 8.3.159 code does:
#
#     preds = self.forward(batch["img"])
#     return self.criterion(preds, batch)
#
# We do EXACTLY the same forward once,
# but before calculating the loss we use preds
# to calculate train accuracy.
#
# THERE IS NO SECOND FORWARD.
#
# ============================================================

def classification_loss_with_accuracy(
    model,
    batch,
    preds=None
):

    # Initialize criterion exactly like BaseModel.loss()
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()

    # ========================================================
    # ORIGINAL FORWARD
    # ========================================================

    if preds is None:
        preds = model.forward(batch["img"])

    # ========================================================
    # TRAIN ACCURACY
    # ========================================================

    trainer = getattr(
        model,
        "_accuracy_trainer",
        None
    )

    # Only calculate accuracy during TRAINING.
    #
    # This prevents validation/final validation
    # from changing train_correct/train_total.
    if (
        trainer is not None
        and model.training
    ):

        with torch.no_grad():

            # Classification output is [batch, num_classes]
            predicted = preds.argmax(dim=1)

            # Ground-truth labels
            target = batch["cls"]

            # Make sure target is on same device
            target = target.to(
                predicted.device
            )

            # Count correct predictions
            trainer.train_correct += (
                predicted == target
            ).sum().item()

            # Count total samples
            trainer.train_total += (
                target.numel()
            )

    # ========================================================
    # ORIGINAL YOLO LOSS
    # ========================================================

    return model.criterion(
        preds,
        batch
    )


# ============================================================
# CUSTOM CLASSIFICATION TRAINER
# ============================================================

class CustomClassificationTrainer(
    ClassificationTrainer
):

    def _setup_train(
        self,
        world_size=1
    ):

        # Run original YOLO setup
        super()._setup_train(
            world_size
        )

        # ----------------------------------------------------
        # Connect trainer to model
        # ----------------------------------------------------

        self.model._accuracy_trainer = self

        # ----------------------------------------------------
        # Replace ONLY the loss function
        # ----------------------------------------------------
        #
        # MethodType makes the function behave like a
        # normal bound method of ClassificationModel.
        #
        # ----------------------------------------------------

        self.model.loss = MethodType(
            classification_loss_with_accuracy,
            self.model
        )


# ============================================================
# EPOCH END
# ============================================================

def on_fit_epoch_end(trainer):

    global history

    elapsed = (
        time.time()
        - epoch_start
    )

    # ========================================================
    # RAM
    # ========================================================

    ram_mb = (
        psutil.Process()
        .memory_info()
        .rss
        / (1024 ** 2)
    )

    # ========================================================
    # GPU PEAK MEMORY
    # ========================================================

    gpu_peak = (

        torch.cuda.max_memory_allocated()
        / (1024 ** 2)

        if torch.cuda.is_available()

        else 0.0
    )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    metrics = (

        trainer.metrics

        if hasattr(
            trainer,
            "metrics"
        )

        else {}
    )

    # ========================================================
    # TRAIN ACCURACY
    # ========================================================

    if trainer.train_total > 0:

        train_acc = (
            trainer.train_correct
            / trainer.train_total
        )

    else:

        train_acc = None

    # ========================================================
    # TRAIN LOSS
    # ========================================================
    #
    # trainer.tloss in Ultralytics 8.3.159 is the
    # running average of training loss over batches.
    #
    # Therefore this is MUCH better than trainer.loss,
    # which represents the current/last batch loss.
    #
    # ========================================================

    if (
        hasattr(trainer, "tloss")
        and trainer.tloss is not None
    ):

        train_loss = (
            trainer.tloss.mean()
            .item()
        )

    else:

        train_loss = None

    # ========================================================
    # VALIDATION ACCURACY
    # ========================================================

    val_acc = metrics.get(
        "metrics/accuracy_top1",
        None
    )

    # ========================================================
    # VALIDATION LOSS
    # ========================================================

    val_loss = metrics.get(
        "val/loss",
        None
    )

    # ========================================================
    # RECORD
    # ========================================================

    record = {

        "epoch":
            trainer.epoch + 1,

        # Accuracy
        "train_acc":
            train_acc,

        "val_acc":
            val_acc,

        # Loss
        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        # Resources
        "time_sec":
            elapsed,

        "ram_mb":
            ram_mb,

        "gpu_peak_mb":
            gpu_peak,
    }

    # ========================================================
    # SAVE TO HISTORY
    # ========================================================

    history.append(record)

    print(record)


# ============================================================
# MODEL
# ============================================================

model = YOLO(
    "yolov8n-cls.pt"
)


# ============================================================
# CALLBACKS
# ============================================================

model.add_callback(
    "on_train_epoch_start",
    on_train_epoch_start
)

model.add_callback(
    "on_fit_epoch_end",
    on_fit_epoch_end
)


# ============================================================
# TRAIN
# ============================================================

model.train(

    data=DATA_ROOT,

    epochs=EPOCHS,

    imgsz=IMG_SIZE,

    batch=BATCH_SIZE,

    device=DEVICE,

    workers=0,

    amp=True,

    cache=True,

    verbose=False,

    # IMPORTANT:
    # Use our custom ClassificationTrainer
    trainer=CustomClassificationTrainer,
)


# ============================================================
# SAVE JSONL
# ============================================================

output = {

    "datetime":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "hyperparameters": {

        "seed": 42,

        "img_size":
            IMG_SIZE,

        "batch_size":
            BATCH_SIZE,

        "num_epochs":
            EPOCHS,

        "model":
            "YOLOv8n-cls",

        "device":
            str(DEVICE),
    },

    "epochs":
        history,
}


# ============================================================
# APPEND MODE
# ============================================================

with open(

    "results.jsonl",

    "a",

    encoding="utf-8"

) as f:

    f.write(

        json.dumps(

            output,

            ensure_ascii=False

        )

        + "\n"
    )


print(
    "\nSaved to results.jsonl"
)