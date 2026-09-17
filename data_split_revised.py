import csv
import random
import shutil
from pathlib import Path
import yaml

# ============================================================
# CONFIGURATION
# ============================================================

# True  → augmented تصاویر دقیقاً به همان اسپلیت parent خودشان می‌روند (Train یا Test)
# False → فقط augmentedهای مربوط به parentهای Train نگه داشته می‌شوند (جلوگیری از leakage)
augment_in_test = True          # ← این را تغییر دهید

CONFIG_PATH = Path("MVconfig.yaml")

CLASSES = [
    "FreshFish",
    "InfectedFish",
]

RAW_FOLDER_NAME = "Raw"
AUGMENTED_FOLDER_NAME = "My augment"

TRAIN_SPLIT_NAME = "train"
TEST_SPLIT_NAME = "test"

RESULTS_DIR = Path("data_split_revised_results")
AUDIT_CSV_NAME = "dataset_split_audit.csv"

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# ============================================================
# LOAD & VALIDATE CONFIG
# ============================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH.resolve()}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SOURCE_DIR = Path(config["DATA_ROOT_PRIMARY"])
OUTPUT_DIR = Path(config["DATA_ROOT_SPLITED"])

TEST_RATIO = float(config.get("TEST_RATIO", 0.20))
TRAIN_RATIO = 1.0 - TEST_RATIO

if not (0.0 <= TEST_RATIO < 1.0):
    raise ValueError("TEST_RATIO must be in range [0, 1).")
if TRAIN_RATIO <= 0:
    raise ValueError(f"Invalid split ratios: TRAIN={TRAIN_RATIO}, TEST={TEST_RATIO}")
if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR.resolve()}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder: Path):
    """Return sorted list of valid image files inside a directory."""
    if not folder.exists():
        return []
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower()
    )


def extract_parent_stem(filename: str) -> str:
    """
    Extract parent ID from an augmented filename.
    Example: salmon_dis_13__rotation__angle_20deg.jpg → salmon_dis_13
    """
    stem = Path(filename).stem
    return stem.split("__", 1)[0] if "__" in stem else stem


def copy_image(source: Path, dest_folder: Path) -> Path:
    """Copy image while preserving metadata."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / source.name
    shutil.copy2(source, dest)
    return dest


def create_output_directories():
    """Remove previous split and create clean train/test structure."""
    if OUTPUT_DIR.exists():
        print(f"🧹 Removing previous split directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in (TRAIN_SPLIT_NAME, TEST_SPLIT_NAME):
        for cls in CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def split_raw_images(raw_images, rng):
    """Split Raw images into Train and Test (no separate Validation)."""
    images = list(raw_images)
    rng.shuffle(images)
    train_count = int(len(images) * TRAIN_RATIO)
    return images[:train_count], images[train_count:]


def save_audit_csv(records, path: Path):
    fieldnames = [
        "class_name", "image_type", "parent_id", "filename",
        "assigned_split", "status_reason", "relative_output_path"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)
    create_output_directories()

    audit_records = []
    summary = {}
    all_train_parents = set()
    all_test_parents = set()

    mode_str = "ALLOW augmented in Test (follow parent)" if augment_in_test else \
               "STRICT (augmented only in Train – no leakage)"

    print("\n" + "=" * 72)
    print("🚀 Dataset Split Pipeline (Train / Test only)")
    print("=" * 72)
    print(f"Mode                 : {mode_str}")
    print(f"Source directory     : {SOURCE_DIR}")
    print(f"Output directory     : {OUTPUT_DIR}")
    print(f"Train ratio          : {TRAIN_RATIO:.2f}  (pool for K-Fold)")
    print(f"Test ratio           : {TEST_RATIO:.2f}  (held-out)")
    print(f"Random seed          : {RANDOM_SEED}")
    print("=" * 72)
    print("Note: No separate Validation folder is created.")
    print("      K-Fold will create validation folds from the Train set.")
    print("=" * 72)

    for class_name in CLASSES:
        # ----------------------------------------------------
        # 1. Split Raw images
        # ----------------------------------------------------
        raw_dir = SOURCE_DIR / RAW_FOLDER_NAME / class_name
        raw_images = get_images(raw_dir)

        if not raw_images:
            print(f"\n⚠️  Warning: No Raw images found for '{class_name}' in {raw_dir}")
            continue

        train_raw, test_raw = split_raw_images(raw_images, rng)

        train_parent_ids = {img.stem for img in train_raw}
        test_parent_ids  = {img.stem for img in test_raw}

        all_train_parents.update(train_parent_ids)
        all_test_parents.update(test_parent_ids)

        # Copy Raw → Train
        for img in train_raw:
            dest = copy_image(img, OUTPUT_DIR / TRAIN_SPLIT_NAME / class_name)
            audit_records.append({
                "class_name": class_name,
                "image_type": "Raw",
                "parent_id": img.stem,
                "filename": img.name,
                "assigned_split": TRAIN_SPLIT_NAME,
                "status_reason": "Included in Train (Raw Split)",
                "relative_output_path": str(dest.relative_to(OUTPUT_DIR)),
            })

        # Copy Raw → Test
        for img in test_raw:
            dest = copy_image(img, OUTPUT_DIR / TEST_SPLIT_NAME / class_name)
            audit_records.append({
                "class_name": class_name,
                "image_type": "Raw",
                "parent_id": img.stem,
                "filename": img.name,
                "assigned_split": TEST_SPLIT_NAME,
                "status_reason": "Included in Test (Raw Split)",
                "relative_output_path": str(dest.relative_to(OUTPUT_DIR)),
            })

        # ----------------------------------------------------
        # 2. Process Augmented images
        # ----------------------------------------------------
        aug_dir = SOURCE_DIR / AUGMENTED_FOLDER_NAME / class_name
        aug_images = get_images(aug_dir)

        aug_to_train = 0
        aug_to_test  = 0
        aug_ignored  = 0
        aug_unmatched = 0

        for aug_img in aug_images:
            parent_id = extract_parent_stem(aug_img.name)

            if parent_id in train_parent_ids:
                # Parent is in Train → always go to Train
                dest = copy_image(aug_img, OUTPUT_DIR / TRAIN_SPLIT_NAME / class_name)
                aug_to_train += 1
                audit_records.append({
                    "class_name": class_name,
                    "image_type": "Augmented",
                    "parent_id": parent_id,
                    "filename": aug_img.name,
                    "assigned_split": TRAIN_SPLIT_NAME,
                    "status_reason": "Included in Train (Parent in Train)",
                    "relative_output_path": str(dest.relative_to(OUTPUT_DIR)),
                })

            elif parent_id in test_parent_ids:
                if augment_in_test:
                    # Mode True: follow parent → go to Test
                    dest = copy_image(aug_img, OUTPUT_DIR / TEST_SPLIT_NAME / class_name)
                    aug_to_test += 1
                    audit_records.append({
                        "class_name": class_name,
                        "image_type": "Augmented",
                        "parent_id": parent_id,
                        "filename": aug_img.name,
                        "assigned_split": TEST_SPLIT_NAME,
                        "status_reason": "Included in Test (Parent in Test)",
                        "relative_output_path": str(dest.relative_to(OUTPUT_DIR)),
                    })
                else:
                    # Mode False: ignore to prevent leakage
                    aug_ignored += 1
                    audit_records.append({
                        "class_name": class_name,
                        "image_type": "Augmented",
                        "parent_id": parent_id,
                        "filename": aug_img.name,
                        "assigned_split": "ignored",
                        "status_reason": "Ignored (Parent in Test – anti-leakage)",
                        "relative_output_path": "None",
                    })
            else:
                # Parent not found in any Raw split
                aug_unmatched += 1
                audit_records.append({
                    "class_name": class_name,
                    "image_type": "Augmented",
                    "parent_id": parent_id,
                    "filename": aug_img.name,
                    "assigned_split": "ignored",
                    "status_reason": "Ignored (Parent Not Found in Raw)",
                    "relative_output_path": "None",
                })

        # ----------------------------------------------------
        # 3. Per-class summary
        # ----------------------------------------------------
        summary[class_name] = {
            "raw_total":       len(raw_images),
            "raw_train":       len(train_raw),
            "raw_test":        len(test_raw),
            "aug_total":       len(aug_images),
            "aug_to_train":    aug_to_train,
            "aug_to_test":     aug_to_test,
            "aug_ignored":     aug_ignored,
            "aug_unmatched":   aug_unmatched,
            "final_train":     len(train_raw) + aug_to_train,
            "final_test":      len(test_raw)  + aug_to_test,
        }

    # --------------------------------------------------------
    # 4. Save audit CSV
    # --------------------------------------------------------
    audit_csv_path = RESULTS_DIR / AUDIT_CSV_NAME
    save_audit_csv(audit_records, audit_csv_path)

    # --------------------------------------------------------
    # 5. Leakage check (based on Raw parent IDs only)
    # --------------------------------------------------------
    overlap = all_train_parents & all_test_parents
    has_leakage = bool(overlap)

    # --------------------------------------------------------
    # 6. Final Report
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("✅ Dataset split completed")
    print("=" * 72)

    total_final_train = 0
    total_final_test  = 0

    for cls, s in summary.items():
        print(f"\n📊 Class: {cls}")
        print("-" * 55)
        print(f"  Raw total                    : {s['raw_total']:5d}")
        print(f"  Raw → Train                  : {s['raw_train']:5d}")
        print(f"  Raw → Test                   : {s['raw_test']:5d}")
        print(f"  Augmented total              : {s['aug_total']:5d}")
        print(f"  Augmented → Train            : {s['aug_to_train']:5d}")
        print(f"  Augmented → Test             : {s['aug_to_test']:5d}")
        print(f"  Augmented ignored (anti-leak): {s['aug_ignored']:5d}")
        print(f"  Augmented unmatched          : {s['aug_unmatched']:5d}")
        print("  " + "·" * 50)
        print(f"  Final Train (Raw + Aug)      : {s['final_train']:5d}")
        print(f"  Final Test  (Raw + Aug)      : {s['final_test']:5d}")

        total_final_train += s["final_train"]
        total_final_test  += s["final_test"]

    print("\n" + "=" * 72)
    print("📈 OVERALL TOTALS")
    print("=" * 72)
    print(f"  Final Train images : {total_final_train}")
    print(f"  Final Test  images : {total_final_test}")
    print(f"  Grand total        : {total_final_train + total_final_test}")

    print("\n" + "=" * 72)
    print("🔍 DATA LEAKAGE VERIFICATION (Raw parent IDs)")
    print("=" * 72)
    print(f"  Overlap Train ∩ Test : {len(overlap)} items")

    if not has_leakage:
        print("  🏆 ZERO DATA LEAKAGE — Train and Test parent sets are disjoint.")
    else:
        print("  ❌ CRITICAL: Data leakage detected between Train and Test parents!")
        print(f"     Overlapping IDs: {sorted(list(overlap))[:10]}{' ...' if len(overlap) > 10 else ''}")

    print(f"\n📁 Audit CSV saved at:")
    print(f"   → {audit_csv_path.resolve()}")
    print("=" * 72)
    print("\n📌 Next step:")
    print("   Use the 'train' folder as the pool for Parent-Level K-Fold.")
    print("   The 'test' folder is a fixed held-out set.")
    print("=" * 72)


if __name__ == "__main__":
    main()