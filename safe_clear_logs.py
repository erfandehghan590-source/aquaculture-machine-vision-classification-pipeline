import os
import shutil
from pathlib import Path
import yaml

# ==============================================================================
# تنظیمات پاکسازی
# ==============================================================================
DRY_RUN = False  # اگر True باشد، فقط فایل‌ها را نمایش می‌دهد و چیزی را حذف نمی‌کند.

ROOT_DIR = Path(
    "C:/Users/conceptD/Desktop/apply resume and research/"
    "research/aquaculture MV/ours/implimentations/"
)
CONFIG_PATH = ROOT_DIR / "model_comparison_config.yaml"

# پسوندهایی که به هیچ وجه نباید پاک شوند (White-list)
PROTECTED_EXTENSIONS = {".py", ".pth", ".pt", ".yaml", ".yml", ".gitignore"}


def clean_target_directory(target_dir: Path, protected_exts: set, dry_run: bool = False):
    """
    پیمایش بازگشتی در یک پوشه و حذف هر فایلی جز پسوندهای مجاز
    """
    if not target_dir.exists():
        print(f"⏩ [Skip] Directory not found: {target_dir}")
        return 0, 0

    deleted_files_count = 0
    deleted_dirs_count = 0

    # پیمایش از عمیق‌ترین سطوح به بالا (bottom-up) برای امکان حذف پوشه‌های خالی‌شده
    for root, dirs, files in os.walk(target_dir, topdown=False):
        current_path = Path(root)

        # 1. بررسی و حذف فایل‌ها
        for file in files:
            file_path = current_path / file
            if file_path.suffix.lower() not in protected_exts:
                if dry_run:
                    print(f"  [DRY-RUN - Would Delete File]: {file_path.relative_to(ROOT_DIR)}")
                else:
                    try:
                        file_path.unlink()
                        print(f"  🗑️ [Deleted File]: {file_path.name}")
                    except Exception as e:
                        print(f"  ❌ Error deleting file {file_path.name}: {e}")
                deleted_files_count += 1
            else:
                # فایل‌های کد و وزن دست نخورده باقی می‌مانند
                pass

        # 2. بررسی و حذف پوشه‌های خالی یا پوشه‌های تولیدی (مثل gradcam/runs)
        for d in dirs:
            dir_path = current_path / d
            # بررسی اینکه آیا در این پوشه فایل کد یا وزنی باقی مانده است یا خیر
            remaining_protected = [
                p for p in dir_path.rglob("*") 
                if p.is_file() and p.suffix.lower() in protected_exts
            ]

            if not remaining_protected:
                # اگر هیچ فایلی از لیست محافظت‌شده در آن نبود، کل پوشه حذف می‌شود
                if dry_run:
                    print(f"  [DRY-RUN - Would Delete Folder]: {dir_path.relative_to(ROOT_DIR)}")
                else:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"  📁🗑️ [Deleted Folder]: {dir_path.name}")
                    except Exception as e:
                        print(f"  ❌ Error deleting folder {dir_path.name}: {e}")
                deleted_dirs_count += 1

    return deleted_files_count, deleted_dirs_count


def main():
    print("=" * 80)
    print("🧹 STARTING WORKSPACE & LOG CLEANUP PIPELINE")
    if DRY_RUN:
        print("🔍 RUNNING IN DRY-RUN MODE (No files will be modified)")
    print("=" * 80)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH.resolve()}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. استخراج مسیرهای پوشه‌های مدل‌ها از روی کانفیگ
    target_dirs = set()
    for m in config.get("models", []):
        log_paths = m.get("log_paths", "")
        if "/" in log_paths or "\\" in log_paths:
            # استخراج نام پوشه قبل از اسلش (مانند mobilenetv3large)
            folder_part = log_paths.replace("\\", "/").split("/")[0]
            folder_full_path = ROOT_DIR / folder_part
            target_dirs.add(folder_full_path)

    # 2. اضافه کردن پوشه خروجی نتایج قبلی (output_dir) جهت پاکسازی کلی
    output_cfg = config.get("output_dir", "model_comparison_results")
    out_dir = Path(output_cfg) if Path(output_cfg).is_absolute() else (ROOT_DIR / output_cfg)
    target_dirs.add(out_dir)

    total_files_cleaned = 0
    total_dirs_cleaned = 0

    # 3. اجرای عملیات پاکسازی برای هر پوشه
    for folder in sorted(list(target_dirs)):
        print(f"\n📂 Cleaning: {folder.name} ({folder.resolve()})")
        f_count, d_count = clean_target_directory(
            target_dir=folder,
            protected_exts=PROTECTED_EXTENSIONS,
            dry_run=DRY_RUN
        )
        total_files_cleaned += f_count
        total_dirs_cleaned += d_count

    print("\n" + "=" * 80)
    print("✨ CLEANUP SUMMARY:")
    print(f"• Total Non-Code/Weight Files Removed : {total_files_cleaned}")
    print(f"• Total Empty/Artifact Folders Removed: {total_dirs_cleaned}")
    print("• Protected Exts (.py, .pth, .pt, .yaml) remained 100% untouched.")
    print("=" * 80)


if __name__ == "__main__":
    main()
