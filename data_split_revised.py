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

TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"
TEST_SPLIT_NAME = "test"

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

VAL_RATIO = float(config.get("VAL_RATIO", 0.15))
TEST_RATIO = float(config.get("TEST_RATIO", 0.15))
TRAIN_RATIO = 1.0 - VAL_RATIO - TEST_RATIO


# ============================================================
# VALIDATE CONFIG
# ============================================================

if VAL_RATIO < 0 or TEST_RATIO < 0:
    raise ValueError("VAL_RATIO and TEST_RATIO must be non-negative.")

if TRAIN_RATIO <= 0:
    raise ValueError(
        f"Invalid split ratios: "
        f"TRAIN={TRAIN_RATIO}, VAL={VAL_RATIO}, TEST={TEST_RATIO}"
    )

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.")


if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source directory not found: {SOURCE_DIR.resolve()}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder: Path):
    """
    Return valid image files inside a directory.

    Only files with extensions listed in IMAGE_EXTENSIONS
    are returned.
    """
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
        salmon_dis_13__rotation__angle_20deg.jpg

    Output:
        salmon_dis_13
    """
    filename_stem = Path(filename).stem

    if "__" in filename_stem:
        return filename_stem.split("__", 1)[0]

    # Files without "__" are treated as their own parent.
    return filename_stem


def copy_image(source_path: Path, destination_folder: Path):
    """
    Copy one image while preserving metadata.
    """
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = destination_folder / source_path.name
    shutil.copy2(source_path, destination_path)


def create_output_directories():
    """
    Remove the previous split dataset and create a clean structure.
    """
    if OUTPUT_DIR.exists():
        print(f"🧹 Removing previous output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split_name in [
        TRAIN_SPLIT_NAME,
        VAL_SPLIT_NAME,
        TEST_SPLIT_NAME,
    ]:
        for class_name in CLASSES:
            split_class_dir = OUTPUT_DIR / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)


def split_raw_images(raw_images, rng):
    """
    Split Raw images into performed, validation, and test subsets.

    The split is performed only on Raw images.
    """
    images = list(raw_images)
    rng.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    return train_images, val_images, test_images


# ============================================================
# MAIN DATA-SPLITTING PIPELINE
# ============================================================

def main():
    rng = random.Random(RANDOM_SEED)

    create_output_directories()

    summary = {}

    print("\n" + "=" * 70)
    print("Starting dataset split")
    print("=" * 70)
    print(f"Source directory : {SOURCE_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Augmented folder : {AUGMENTED_FOLDER_NAME}")
    print(f"Train ratio      : {TRAIN_RATIO:.2f}")
    print(f"Validation ratio : {VAL_RATIO:.2f}")
    print(f"Test ratio       : {TEST_RATIO:.2f}")
    print(f"Random seed      : {RANDOM_SEED}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Split Raw images
    # --------------------------------------------------------

    for class_name in CLASSES:
        raw_dir = SOURCE_DIR / RAW_FOLDER_NAME / class_name
        raw_images = get_images(raw_dir)

        if not raw_images:
            print(
                f"\n⚠️ Warning: No Raw images found for "
                f"class '{class_name}' in:\n{raw_dir}"
            )

            summary[class_name] = {
                "raw_total": 0,
                "raw_train": 0,
                "raw_val": 0,
                "raw_test": 0,
                "aug_total": 0,
                "aug_train_added": 0,
                "aug_val_test_ignored": 0,
                "aug_unmatched": 0,
            }
            continue

        train_raw, val_raw, test_raw = split_raw_images(
            raw_images,
            rng
        )

        # Copy Raw Train images
        for image_path in train_raw:
            copy_image(
                image_path,
                OUTPUT_DIR / TRAIN_SPLIT_NAME / class_name
            )

        # Copy Raw Validation images
        for image_path in val_raw:
            copy_image(
                image_path,
                OUTPUT_DIR / VAL_SPLIT_NAME / class_name
            )

        # Copy Raw Test images
        for image_path in test_raw:
            copy_image(
                image_path,
                OUTPUT_DIR / TEST_SPLIT_NAME / class_name
            )

        # Store parent IDs of Raw Train/Val/Test
        train_parent_ids = {
            image_path.stem
            for image_path in train_raw
        }

        val_parent_ids = {
            image_path.stem
            for image_path in val_raw
        }

        test_parent_ids = {
            image_path.stem
            for image_path in test_raw
        }

        # ----------------------------------------------------
        # 2. Add only Train-related Augmented images
        # ----------------------------------------------------

         # ----------------------------------------------------
        # 2. Route Augmented images to the SAME split as
        #    their parent (no leakage, since the parent
        #    itself is only in one split).
        # ----------------------------------------------------

        augmented_dir = (
            SOURCE_DIR
            / AUGMENTED_FOLDER_NAME
            / class_name
        )

        augmented_images = get_images(augmented_dir)

        augmented_train_count = 0
        augmented_val_count = 0
        augmented_test_count = 0
        augmented_unmatched_count = 0

        for augmented_image in augmented_images:
            parent_id = extract_parent_stem(
                augmented_image.name
            )

            if parent_id in train_parent_ids:
                copy_image(
                    augmented_image,
                    OUTPUT_DIR / TRAIN_SPLIT_NAME / class_name
                )
                augmented_train_count += 1

            elif parent_id in val_parent_ids:
                copy_image(
                    augmented_image,
                    OUTPUT_DIR / VAL_SPLIT_NAME / class_name
                )
                augmented_val_count += 1

            elif parent_id in test_parent_ids:
                copy_image(
                    augmented_image,
                    OUTPUT_DIR / TEST_SPLIT_NAME / class_name
                )
                augmented_test_count += 1

            else:
                # Parent ID could not be matched to any Raw image.
                augmented_unmatched_count += 1

            summary[class_name] = {
            "raw_total": len(raw_images),
            "raw_train": len(train_raw),
            "raw_val": len(val_raw),
            "raw_test": len(test_raw),
            "aug_total": len(augmented_images),
            "aug_train_added": augmented_train_count,
            "aug_val_added": augmented_val_count,
            "aug_test_added": augmented_test_count,
            "aug_unmatched": augmented_unmatched_count,
            "final_train_total": (
                len(train_raw) + augmented_train_count
            ),
            "final_val_total": (
                len(val_raw) + augmented_val_count
            ),
            "final_test_total": (
                len(test_raw) + augmented_test_count
            ),
        }

    # --------------------------------------------------------
    # 3. Print final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ Dataset split completed successfully")
    print("=" * 70)



    for class_name, stats in summary.items():
        print(f"\n📊 Class: {class_name}")
        print("-" * 50)

        print(f"Raw total                  : {stats['raw_total']}")
        print(f"Raw Train                  : {stats['raw_train']}")
        print(f"Raw Validation             : {stats['raw_val']}")
        print(f"Raw Test                   : {stats['raw_test']}")
        print(f"Augmented total            : {stats['aug_total']}")
        print(f"Augmented added to Train  : {stats['aug_train_added']}")
        print(f"Augmented added to Val     : {stats['aug_val_added']}")
        print(f"Augmented added to Test    : {stats['aug_test_added']}")
        print(f"Augmented unmatched        : {stats['aug_unmatched']}")
        
        print(f"Final Train total          : {stats['final_train_total']}")
        print(f"Final Validation total     : {stats['final_val_total']}")
        print(f"Final Test total           : {stats['final_test_total']}")

    print("\n" + "=" * 70)
    print("Final dataset policy:")
    print("  Train = Raw Train + matched Augmented images")
    print("  Val   = Raw Validation only")
    print("  Test  = Raw Test only")
    print("=" * 70)


if __name__ == "__main__":
    main()
