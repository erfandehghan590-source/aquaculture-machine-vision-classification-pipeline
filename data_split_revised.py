import csv
import random
import shutil
from pathlib import Path
import yaml

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = Path("MVconfig.yaml")

CLASSES = [
    "FreshFish",
    "InfectedFish",
]

RAW_FOLDER_NAME = "Raw"
AUGMENTED_FOLDER_NAME = "My augment"

# فقط دو اسپلیت نهایی (چون K-Fold خودش Validation را می‌سازد)
TRAIN_SPLIT_NAME = "train"   # این پوشه در واقع Train+Val خواهد بود
TEST_SPLIT_NAME = "test"

RESULTS_DIR = Path("data_split_revised_results")
AUDIT_CSV_NAME = "dataset_split_audit.csv"

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# LOAD CONFIG
# ============================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH.resolve()}"
    )

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


SOURCE_DIR = Path(config["DATA_ROOT_PRIMARY"])
OUTPUT_DIR = Path(config["DATA_ROOT_SPLITED"])

# فقط نسبت Test مهم است. بقیه داده‌ها به Train می‌روند
# (K-Fold خودش از داخل Train، Validation می‌سازد)
TEST_RATIO = float(config.get("TEST_RATIO", 0.20))
TRAIN_RATIO = 1.0 - TEST_RATIO


# ============================================================
# VALIDATE CONFIG
# ============================================================

if TEST_RATIO < 0 or TEST_RATIO >= 1.0:
    raise ValueError("TEST_RATIO must be in range [0, 1).")

if TRAIN_RATIO <= 0:
    raise ValueError(
        f"Invalid split ratios: TRAIN={TRAIN_RATIO}, TEST={TEST_RATIO}"
    )

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source directory not found: {SOURCE_DIR.resolve()}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder: Path):
    """Return valid image files inside a directory."""
    if not folder.exists():
        return []

    return sorted(
        [
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower()
    )


def extract_parent_stem(filename: str) -> str:
    """
    Extract parent image ID from an augmented filename.

    Example:
        salmon_dis_13__rotation__angle_20deg.jpg → salmon_dis_13
    """
    filename_stem = Path(filename).stem
    if "__" in filename_stem:
        return filename_stem.split("__", 1)[0]
    return filename_stem


def copy_image(source_path: Path, destination_folder: Path):
    """Copy one image while preserving metadata."""
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = destination_folder / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path


def create_output_directories():
    """
    Remove previous split dataset and create a clean directory structure.
    فقط train و test ساخته می‌شوند.
    """
    if OUTPUT_DIR.exists():
        print(f"🧹 Removing previous split directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split_name in [TRAIN_SPLIT_NAME, TEST_SPLIT_NAME]:
        for class_name in CLASSES:
            split_class_dir = OUTPUT_DIR / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def split_raw_images(raw_images, rng):
    """
    Split Raw images into Train and Test.
    Validation جدا ساخته نمی‌شود (K-Fold آن را می‌سازد).
    """
    images = list(raw_images)
    rng.shuffle(images)

    total_images = len(images)
    train_count = int(total_images * TRAIN_RATIO)

    train_images = images[:train_count]
    test_images = images[train_count:]

    return train_images, test_images


def save_audit_csv(records, output_csv_path: Path):
    """Save the complete dataset split ledger to CSV."""
    fieldnames = [
        "class_name",
        "image_type",
        "parent_id",
        "filename",
        "assigned_split",
        "status_reason",
        "relative_output_path",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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

    print("\n" + "=" * 70)
    print("🚀 Starting dataset split pipeline (Train / Test only)")
    print("=" * 70)
    print(f"Source directory     : {SOURCE_DIR}")
    print(f"Output directory     : {OUTPUT_DIR}")
    print(f"Results directory    : {RESULTS_DIR}")
    print(f"Train ratio          : {TRAIN_RATIO:.2f}  (will be used for K-Fold)")
    print(f"Test ratio           : {TEST_RATIO:.2f}  (held-out final test)")
    print(f"Random seed          : {RANDOM_SEED}")
    print("=" * 70)
    print("Note: No separate Validation folder is created.")
    print("      K-Fold will create validation folds from the Train set.")
    print("=" * 70)

    for class_name in CLASSES:
        # ----------------------------------------------------
        # 1. Process and Split Raw Images → Train / Test
        # ----------------------------------------------------
        raw_dir = SOURCE_DIR / RAW_FOLDER_NAME / class_name
        raw_images = get_images(raw_dir)

        if not raw_images:
            print(f"\n⚠️ Warning: No Raw images found for class '{class_name}' in: {raw_dir}")
            continue

        train_raw, test_raw = split_raw_images(raw_images, rng)

        train_parent_ids = {img.stem for img in train_raw}
        test_parent_ids = {img.stem for img in test_raw}

        all_train_parents.update(train_parent_ids)
        all_test_parents.update(test_parent_ids)

        # Copy and record Raw Train
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

        # Copy and record Raw Test
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
        # 2. Process Augmented Images (Train-Only Policy)
        # ----------------------------------------------------
        augmented_dir = SOURCE_DIR / AUGMENTED_FOLDER_NAME / class_name
        augmented_images = get_images(augmented_dir)

        aug_train_count = 0
        aug_test_ignored_count = 0
        aug_unmatched_count = 0

        for aug_img in augmented_images:
            parent_id = extract_parent_stem(aug_img.name)

            if parent_id in train_parent_ids:
                dest = copy_image(aug_img, OUTPUT_DIR / TRAIN_SPLIT_NAME / class_name)
                aug_train_count += 1
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
                aug_test_ignored_count += 1
                audit_records.append({
                    "class_name": class_name,
                    "image_type": "Augmented",
                    "parent_id": parent_id,
                    "filename": aug_img.name,
                    "assigned_split": "ignored",
                    "status_reason": "Ignored (Parent in Test)",
                    "relative_output_path": "None",
                })
            else:
                aug_unmatched_count += 1
                audit_records.append({
                    "class_name": class_name,
                    "image_type": "Augmented",
                    "parent_id": parent_id,
                    "filename": aug_img.name,
                    "assigned_split": "ignored",
                    "status_reason": "Ignored (Parent Not Found in Raw)",
                    "relative_output_path": "None",
                })

        summary[class_name] = {
            "raw_total": len(raw_images),
            "raw_train": len(train_raw),
            "raw_test": len(test_raw),
            "aug_total": len(augmented_images),
            "aug_train_added": aug_train_count,
            "aug_test_ignored": aug_test_ignored_count,
            "aug_unmatched": aug_unmatched_count,
            "final_train_total": len(train_raw) + aug_train_count,
            "final_test_total": len(test_raw),
        }

    # --------------------------------------------------------
    # 3. Save Ledger Table
    # --------------------------------------------------------
    audit_csv_path = RESULTS_DIR / AUDIT_CSV_NAME
    save_audit_csv(audit_records, audit_csv_path)

    # --------------------------------------------------------
    # 4. Strict Leakage Verification
    # --------------------------------------------------------
    train_test_overlap = all_train_parents & all_test_parents
    has_leakage = bool(train_test_overlap)

    # --------------------------------------------------------
    # 5. Print Final Report
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ Dataset split completed successfully")
    print("=" * 70)

    for class_name, stats in summary.items():
        print(f"\n📊 Class: {class_name}")
        print("-" * 50)
        print(f"  Raw total                  : {stats['raw_total']}")
        print(f"  Raw Train                  : {stats['raw_train']}")
        print(f"  Raw Test                   : {stats['raw_test']}")
        print(f"  Augmented total            : {stats['aug_total']}")
        print(f"  Augmented added to Train   : {stats['aug_train_added']}")
        print(f"  Augmented ignored (Test)   : {stats['aug_test_ignored']}")
        print(f"  Augmented unmatched        : {stats['aug_unmatched']}")
        print("  " + "." * 46)
        print(f"  Final Train total (Raw+Aug): {stats['final_train_total']}")
        print(f"  Final Test (Raw only)      : {stats['final_test_total']}")

    print("\n" + "=" * 70)
    print("🔍 DATA LEAKAGE VERIFICATION:")
    print("=" * 70)
    print(f"  Overlap Train & Test  : {len(train_test_overlap)} items")

    if not has_leakage:
        print("  🏆 ZERO DATA LEAKAGE: Train and Test parent identities are strictly disjoint.")
    else:
        print("  ❌ CRITICAL ERROR: Data leakage detected between Train and Test!")

    print(f"\n📁 Audit Table Saved at:")
    print(f"  -> {audit_csv_path.resolve()}")
    print("=" * 70)
    print("\n📌 Next step:")
    print("   Use the 'train' folder as the pool for Parent-Level K-Fold.")
    print("   The 'test' folder remains a fixed held-out set.")
    print("=" * 70)


if __name__ == "__main__":
    main()
