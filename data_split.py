import os
import shutil
import random
from pathlib import Path
import yaml

# ---------------- CONFIG ----------------
CONFIG_PATH = "MVconfig.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

SOURCE_DIR = Path(config["DATA_ROOT_PRIMARY"])
OUTPUT_DIR = Path(config["DATA_ROOT_SPLITED"])
CLASSES = ["FreshFish", "InfectedFish"]

VAL_RATIO = float(config.get("VAL_RATIO", 0.15))
TEST_RATIO = float(config.get("TEST_RATIO", 0.15))
TRAIN_RATIO = 1.0 - (VAL_RATIO + TEST_RATIO)

RANDOM_SEED = 42
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ----------------------------------------

random.seed(RANDOM_SEED)

def get_images(folder: Path):
    if not folder.exists():
        return []
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
    ]

def split_and_copy(images, ratios, class_name):
    random.shuffle(images)
    total = len(images)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    splits_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_imgs in splits_data.items():
        dest_dir = OUTPUT_DIR / split_name / class_name
        for img in split_imgs:
            shutil.copy2(img, dest_dir / img.name)

# Clean existing output directory if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Recreate clean folder structure
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)

# Process datasets
for cls in CLASSES:
    # 1. Process Raw images
    raw_path = SOURCE_DIR / "Raw" / cls
    raw_images = get_images(raw_path)
    split_and_copy(raw_images, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO], cls)

    # 2. Process Augmented images
    aug_path = SOURCE_DIR / "Augmented" / cls
    if not aug_path.exists():
        aug_path = SOURCE_DIR / "My augment" / cls

    aug_images = get_images(aug_path)
    split_and_copy(aug_images, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO], cls)

print("Dataset split completed successfully.")
