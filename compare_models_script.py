from pathlib import Path
from IPython.display import display
import yaml
from model_comparison_utils import (
    compare_models,
    plot_model_comparison,
    get_model_weights_table
)

CLASS_NAMES = ["FreshFish","InfectedFish"]

def resolve_path(path_value, root_dir: Path) -> Path | None:
    """
    Returns absolute Path if path_value is absolute,
    otherwise resolves it relative to root_dir.
    """
    if path_value is None:
        return None

    path_value = Path(path_value).expanduser()
    if path_value.is_absolute():
        return path_value.resolve()

    return (root_dir / path_value).resolve()


def load_model_config(config_path: str | Path) -> dict:
    """
    Loads YAML config and resolves all relative paths to absolute Paths.
    """
    config_path = Path(config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Fallback to config file parent directory if root_dir is not explicitly set
    if config.get("root_dir"):
        root_dir = Path(config["root_dir"]).expanduser().resolve()
    else:
        root_dir = config_path.parent

    config["root_dir"] = root_dir

    # Ground Truth and Output paths
    config["gt_folder"] = resolve_path(
        config.get("gt_folder", "ground_truth"),
        root_dir,
    )

    config["output_dir"] = resolve_path(
        config.get("output_dir", "model_comparison_results"),
        root_dir,
    )

    # General execution parameters
    config["threshold"] = float(config.get("threshold", 0.65))
    config["save_excel"] = bool(config.get("save_excel", True))
    config["enable_xai"] = bool(config.get("enable_xai", False))   # <-- NEW

    if "models" not in config or not isinstance(config["models"], list):
        raise ValueError("Config must contain a 'models' list.")

    # Resolve paths for each model entry
    for model_cfg in config["models"]:
        if "name" not in model_cfg:
            raise ValueError("Each model config must have a 'name' field.")

        if "log_path" not in model_cfg:
            raise ValueError(f"Model '{model_cfg['name']}' has no 'log_path'.")

        # heatmap_folder فقط وقتی XAI روشن باشد اجباری است
        if config["enable_xai"] and "heatmap_folder" not in model_cfg:
            raise ValueError(
                f"Model '{model_cfg['name']}' has no 'heatmap_folder' "
                f"(required because enable_xai=True)."
            )

        model_cfg["log_path"] = resolve_path(
            model_cfg["log_path"],
            root_dir,
        )

        if "heatmap_folder" in model_cfg:
            model_cfg["heatmap_folder"] = resolve_path(
                model_cfg["heatmap_folder"],
                root_dir,
            )
        else:
            model_cfg["heatmap_folder"] = None

    return config


# ============================================================
# Load Config
# ============================================================

ROOT_DIR = Path(
    "C:/Users/conceptD/Desktop/apply resume and research/"
    "research/aquaculture MV/ours/implimentations/"
)

CONFIG_PATH = ROOT_DIR / "model_comparison_config.yaml"

config = load_model_config(CONFIG_PATH)

MODELS = config["models"]
GT_FOLDER = Path(config["gt_folder"]) if config["gt_folder"] else None
OUTPUT_DIR = Path(config["output_dir"])
THRESHOLD = config["threshold"]
SAVE_EXCEL = config["save_excel"]
ENABLE_XAI = config["enable_xai"]

print("\nLoaded Configuration:")
print("=" * 80)
print("Config Path :", Path(CONFIG_PATH).resolve())
print("Root Dir    :", config["root_dir"])
print("GT Folder   :", GT_FOLDER)
print("Output Dir  :", OUTPUT_DIR)
print("Threshold   :", THRESHOLD)
print("Save Excel  :", SAVE_EXCEL)
print("Enable XAI  :", ENABLE_XAI)


# ============================================================
# Check Paths and Filter Valid Models
# ============================================================

print("\nChecking Paths:")
print("=" * 80)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory ready: {OUTPUT_DIR}")

# GT فقط وقتی XAI روشن باشد الزامی است
if ENABLE_XAI:
    if GT_FOLDER is None or not GT_FOLDER.is_dir():
        print(f"\nERROR: GT_FOLDER does not exist:\n{GT_FOLDER}")
        print("Model comparison with XAI cannot continue without Ground Truth masks.")
        VALID_MODELS = []
        SKIPPED_MODELS = []
    else:
        print(f"GT_FOLDER exists: {GT_FOLDER}")
else:
    print("XAI is disabled → GT_FOLDER check skipped.")

if not ENABLE_XAI or (GT_FOLDER is not None and GT_FOLDER.is_dir()):
    VALID_MODELS = []
    SKIPPED_MODELS = []

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        log_path = Path(model_cfg["log_path"])
        heatmap_folder = model_cfg.get("heatmap_folder")

        missing_items = []

        if not log_path.is_file():
            missing_items.append(f"Training log not found: {log_path}")

        # فقط وقتی XAI روشن است heatmap را چک کن
        if ENABLE_XAI:
            if heatmap_folder is None or not Path(heatmap_folder).is_dir():
                missing_items.append(
                    f"Heatmap folder not found: {heatmap_folder}"
                )

        if missing_items:
            print("\n" + "-" * 80)
            print(f"[SKIPPED] {model_name}")
            for item in missing_items:
                print(f"  - {item}")

            SKIPPED_MODELS.append({
                "name": model_name,
                "reasons": missing_items,
            })
        else:
            print("\n" + "-" * 80)
            print(f"[READY] {model_name}")
            print(f"  Log file      : {log_path}")
            if ENABLE_XAI and heatmap_folder is not None:
                print(f"  Heatmap folder: {heatmap_folder}")
            VALID_MODELS.append(model_cfg)

    print("\n" + "=" * 80)
    print(f"Total models in config : {len(MODELS)}")
    print(f"Ready for comparison   : {len(VALID_MODELS)}")
    print(f"Skipped models         : {len(SKIPPED_MODELS)}")

    if SKIPPED_MODELS:
        print("\nSkipped models:")
        for skipped in SKIPPED_MODELS:
            print(f"  - {skipped['name']}")

    # ============================================================
    # Compare Models
    # ============================================================

    if len(VALID_MODELS) == 0:
        print("\nNo valid models available for comparison.")
        print("Please verify 'log_path' (and 'heatmap_folder' if enable_xai=True) in YAML.")
    else:
        print("\n" + "=" * 80)
        print("STARTING MODEL COMPARISON PIPELINE")
        print(f"XAI evaluation : {'ENABLED' if ENABLE_XAI else 'DISABLED'}")
        print("=" * 80)

        # اگر تابع compare_models از پارامتر enable_xai پشتیبانی می‌کند:
        try:
            summary_df, all_epochs_df, xai_per_image_df = compare_models(
                model_configs=VALID_MODELS,
                gt_folder=GT_FOLDER if ENABLE_XAI else None,
                threshold=THRESHOLD,
                output_dir=OUTPUT_DIR,
                save_excel=SAVE_EXCEL,
                enable_xai=ENABLE_XAI,          # <-- NEW (اگر utility پشتیبانی کند)
            )
        except TypeError:
            # fallback برای نسخه‌های قدیمی utility که enable_xai ندارند
            summary_df, all_epochs_df, xai_per_image_df = compare_models(
                model_configs=VALID_MODELS,
                gt_folder=GT_FOLDER if ENABLE_XAI else None,
                threshold=THRESHOLD,
                output_dir=OUTPUT_DIR,
                save_excel=SAVE_EXCEL,
            )
            if not ENABLE_XAI:
                xai_per_image_df = None

        # ============================================================
        # Plot Comparison Charts
        # ============================================================

        if summary_df is not None and not summary_df.empty:
            plot_kwargs = {
                "model_summary_df": summary_df,
                "output_dir": OUTPUT_DIR,
            }
            # فقط وقتی XAI فعال است xai_df را پاس بده
            if ENABLE_XAI and xai_per_image_df is not None:
                plot_kwargs["xai_per_image_df"] = xai_per_image_df

            try:
                plot_model_comparison(**plot_kwargs)
            except TypeError:
                # fallback اگر signature تابع plot قدیمی باشد
                plot_model_comparison(
                    model_summary_df=summary_df,
                    output_dir=OUTPUT_DIR,
                    xai_per_image_df=xai_per_image_df if ENABLE_XAI else None,
                )

        # ============================================================
        # Display Final Summary  (expanded columns)
        # ============================================================

        if summary_df is not None and not summary_df.empty:
            # ستون‌های اصلی مقایسه
            columns_to_show = [
                "model",
                "best_epoch",
                "best_train_loss",
                "best_train_acc",
                "best_val_loss",
                "best_val_acc",
                "test_loss",          # <-- اضافه شد
                "test_acc",           # <-- اضافه شد
                "last_train_loss",
                "last_train_acc",
                "last_val_loss",
                "last_val_acc",
                "mean_epoch_time_sec",
                "mean_ram_mb",
                "max_ram_mb",
                "mean_gpu_peak_mb",
                "max_gpu_peak_mb",
            ]
            for cls in CLASS_NAMES:
                columns_to_show.extend([
                f"{cls}_sens",
                f"{cls}_spec",
                f"{cls}_f1",
])

            # اگر XAI روشن بود، متریک‌های آن را هم اضافه کن
            if ENABLE_XAI:
                columns_to_show.extend([
                    "matched_heatmaps",
                    "IoU_mean",
                    "Dice_mean",
                    "SoftIoU_mean",
                    "SoftDice_mean",
                    "SSIM_mean",
                ])

            available_columns = [
                col for col in columns_to_show if col in summary_df.columns
            ]

            print("\n" + "=" * 110)
            print("FINAL MODEL COMPARISON SUMMARY")
            print("=" * 110)

            # مرتب‌سازی بر اساس test_acc (اگر وجود داشته باشد)، در غیر این صورت best_val_acc
            sort_col = "test_acc" if "test_acc" in available_columns else "best_val_acc"

            display(
                summary_df[available_columns].sort_values(
                    by=sort_col,
                    ascending=False,
                    na_position="last",
                )
            )
        else:
            print("\nNo summary data was generated.")

        # ============================================================
        # Display All Epoch Results
        # ============================================================

        print("\n" + "=" * 110)
        print("ALL EPOCH METRICS")
        print("=" * 110)

        if all_epochs_df is not None and not all_epochs_df.empty:
            if {"model", "epoch"}.issubset(all_epochs_df.columns):
                display(all_epochs_df.sort_values(by=["model", "epoch"]))
            else:
                display(all_epochs_df)
        else:
            print("No epoch-level training logs found.")

        # استخراج جدول مشخصات وزن‌ها و نمایش در نوت‌بوک
        weights_info_df = get_model_weights_table(MODELS, output_dir=OUTPUT_DIR, save_excel=SAVE_EXCEL)

        print("\n" + "=" * 80)
        print("PRETRAINED WEIGHTS SPECIFICATIONS TABLE")
        print("=" * 80)
        display(weights_info_df)


        # ============================================================
        # Display XAI Per-Image Results (only if enabled)
        # ============================================================

        if ENABLE_XAI:
            if xai_per_image_df is not None and not xai_per_image_df.empty:
                print("\n" + "=" * 110)
                print("XAI PER-IMAGE RESULTS (HEAD 20)")
                print("=" * 110)

                sort_cols = [
                    col for col in ["model", "image_key"]
                    if col in xai_per_image_df.columns
                ]

                if sort_cols:
                    display(xai_per_image_df.sort_values(by=sort_cols).head(20))
                else:
                    display(xai_per_image_df.head(20))
            else:
                print(
                    "\nXAI evaluation was skipped because no matching "
                    "heatmaps or Ground Truth masks were found."
                )
        else:
            print("\nXAI evaluation is disabled (enable_xai=False).")

        print("\nAll pipeline results successfully saved in:")
        print(OUTPUT_DIR.resolve())