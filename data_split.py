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

TEST_RATIO = float(config.get("TEST_RATIO", 0.2))
TRAIN_RATIO = 1.0 - TEST_RATIO

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

    splits_data = {
        "train": images[:train_end],
        "test": images[train_end:]
    }

    for split_name, split_imgs in splits_data.items():
        dest_dir = OUTPUT_DIR / split_name / class_name
        for img in split_imgs:
            shutil.copy2(img, dest_dir / img.name)

# ======================================================
# پاک کردن کامل پوشه خروجی قبل از شروع اسپلیت
# ======================================================
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# ساخت مجدد ساختار پوشه‌ها
for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)

# ======================================================
# فقط از پوشه Augmented استفاده می‌کنیم (Raw حذف شد)
# ======================================================
for cls in CLASSES:
    # فقط Augmented
    aug_path = SOURCE_DIR / "Augmented" / cls
    if not aug_path.exists():
        aug_path = SOURCE_DIR / "My augment" / cls

    aug_images = get_images(aug_path)
    split_and_copy(aug_images, [TRAIN_RATIO, TEST_RATIO], cls)

print("Dataset split completed successfully (only Augmented data used).")
