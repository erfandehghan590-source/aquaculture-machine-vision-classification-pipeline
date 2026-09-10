from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from adjustText import adjust_text
import torchvision.models as tv_models


# ============================================================
# 1. معیارهای مقایسه Heatmap
# ============================================================
def iou(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.65) -> float:
    """IoU دودویی پس از threshold کردن نقشه‌ها."""
    gt_bin = gt >= threshold
    pred_bin = pred >= threshold

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()

    # اگر هر دو نقشه کاملاً خالی باشند
    if union == 0:
        return 1.0

    return float(intersection / union)


def dice(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.65) -> float:
    """Dice coefficient دودویی پس از threshold کردن نقشه‌ها."""
    gt_bin = gt >= threshold
    pred_bin = pred >= threshold

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    denominator = gt_bin.sum() + pred_bin.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection) / denominator)


def soft_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """IoU پیوسته، بدون threshold."""
    intersection = np.minimum(gt, pred).sum()
    union = np.maximum(gt, pred).sum()

    return float(intersection / (union + 1e-8))


def soft_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice پیوسته، بدون threshold."""
    intersection = (gt * pred).sum()
    denominator = gt.sum() + pred.sum()

    return float((2.0 * intersection) / (denominator + 1e-8))


# ============================================================
# 2. آماده‌سازی Heatmap
# ============================================================
def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    نرمال‌سازی heatmap در بازه [0, 1].

    همچنین اگر آرایه شکل غیرمنتظره‌ای داشته باشد،
    آن را به float32 تبدیل می‌کند.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)

    # برای حالت‌هایی مثل (1, H, W) یا (H, W, 1)
    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(
            f"Heatmap must be 2D after squeeze. Got shape: {heatmap.shape}"
        )

    min_value = float(heatmap.min())
    max_value = float(heatmap.max())

    heatmap = heatmap - min_value

    if max_value - min_value > 0:
        heatmap = heatmap / (max_value - min_value)

    return heatmap.astype(np.float32)


def resize_heatmap_if_needed(
    heatmap: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    اگر اندازه‌ی heatmap با Ground Truth متفاوت باشد،
    با interpolation آن را resize می‌کند.

    نیازمند OpenCV نیست؛ از matplotlib/NumPy استفاده نمی‌کند،
    بلکه در صورت لزوم skimage transform را import می‌کند.
    """
    if heatmap.shape == target_shape:
        return heatmap

    from skimage.transform import resize

    resized = resize(
        heatmap,
        target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )

    return resized.astype(np.float32)


# ============================================================
# 3. تطبیق نام فایل‌های Ground Truth و Heatmap
# ============================================================
def extract_image_key(filename_or_path: str | Path) -> str:
    """
    استخراج کلید نام تصویر از فایل heatmap یا annotation.
    
    این تابع به دنبال الگوی 'salmon_dis_X' می‌گردد که در آن X یک یا چند رقم است.
    
    مثال‌ها
    -------
    salmon_dis_01.npy
        -> salmon_dis_01
    
    0007_salmon_dis_189_case-TP_true-InfectedFish_pred-InfectedFish_conf-0.972_heatmap.npy
        -> salmon_dis_189
        
    protonet_resnet50_salmon_dis_029_heatmap.npy
        -> salmon_dis_029
    """
    name = Path(filename_or_path).name  # استفاده از name به جای stem برای امنیت بیشتر روی نام فایل
    
    # جستجوی الگوی salmon_dis_ به همراه یک یا چند رقم (\d+) بدون حساسیت به حروف بزرگ و کوچک
    match = re.search(r"salmon_dis_\d+", name, flags=re.IGNORECASE)
    
    if match:
        return match.group(0).lower()  # خروجی همیشه به صورت حروف کوچک یکدست بازگردانده می‌شود
    
    # اگر الگو پیدا نشد، به عنوان رفتار زاپاس (fallback) stem فایل را برمی‌گردانیم
    return Path(filename_or_path).stem.strip().lower()



def build_heatmap_file_map(folder: str | Path) -> dict[str, Path]:
    """
    فایل‌های npy را از یک پوشه می‌خواند و به شکل:
    {image_key: file_path}
    برمی‌گرداند.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    file_map: dict[str, Path] = {}

    for file_path in folder.glob("*.npy"):
        key = extract_image_key(file_path.name)

        # در حالت تکراری بودن کلید، آخرین فایل جایگزین می‌شود.
        # بهتر است چنین تکراری‌ای در خروجی نداشته باشی.
        file_map[key] = file_path

    return file_map


# ============================================================
# 4. ارزیابی XAI یک مدل
# ============================================================
def evaluate_heatmaps_against_ground_truth(
    pred_folder: str | Path,
    gt_folder: str | Path,
    threshold: float = 0.65,
    model_name: Optional[str] = None,
    save_per_image_csv: Optional[str | Path] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Heatmapهای یک مدل را با Soft Ground Truthهای پزشکی مقایسه می‌کند.

    Parameters
    ----------
    pred_folder:
        پوشه‌ی heatmapهای پیش‌بینی‌شده‌ی یک مدل.

    gt_folder:
        پوشه‌ی annotation/Soft Ground Truth با فرمت .npy.

    threshold:
        آستانه‌ی معیارهای Binary IoU و Dice.

    model_name:
        نام مدل برای ثبت در خروجی.

    save_per_image_csv:
        در صورت تعیین، نتایج تک‌تصویری در فایل CSV ذخیره می‌شوند.

    Returns
    -------
    summary:
        دیکشنری شامل mean/std معیارها و تعداد فایل‌های مشترک.

    per_image_df:
        DataFrame شامل نتایج هر تصویر.
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    gt_map = build_heatmap_file_map(gt_folder)
    pred_map = build_heatmap_file_map(pred_folder)

    common_keys = sorted(set(gt_map) & set(pred_map))
    missing_gt = sorted(set(pred_map) - set(gt_map))
    missing_pred = sorted(set(gt_map) - set(pred_map))

    if not common_keys:
        raise RuntimeError(
            "No matching .npy files were found between GT and prediction folders.\n"
            f"GT folder: {gt_folder.resolve()}\n"
            f"Prediction folder: {pred_folder.resolve()}\n"
            "Check annotation filenames and extract_image_key()."
        )

    rows: list[dict[str, Any]] = []

    for key in common_keys:
        gt = np.load(gt_map[key])
        pred = np.load(pred_map[key])

        gt = normalize_heatmap(gt)
        pred = normalize_heatmap(pred)

        # در صورت تفاوت سایز annotation و heatmap
        pred = resize_heatmap_if_needed(pred, gt.shape)

        row = {
            "model": model_name,
            "image_key": key,
            "gt_file": gt_map[key].name,
            "pred_file": pred_map[key].name,
            "IoU": iou(gt, pred, threshold),
            "Dice": dice(gt, pred, threshold),
            "SoftIoU": soft_iou(gt, pred),
            "SoftDice": soft_dice(gt, pred),
            "SSIM": float(ssim(gt, pred, data_range=1.0)),
        }

        rows.append(row)

    per_image_df = pd.DataFrame(rows)

    metric_columns = ["IoU", "Dice", "SoftIoU", "SoftDice", "SSIM"]

    summary: dict[str, Any] = {
        "model": model_name,
        "heatmap_folder": str(pred_folder.resolve()),
        "gt_folder": str(gt_folder.resolve()),
        "threshold": threshold,
        "matched_heatmaps": len(common_keys),
        "prediction_heatmaps": len(pred_map),
        "ground_truth_heatmaps": len(gt_map),
        "predictions_without_gt": len(missing_gt),
        "ground_truth_without_prediction": len(missing_pred),
    }

    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(per_image_df[metric].mean())
        summary[f"{metric}_std"] = float(per_image_df[metric].std(ddof=0))

    if save_per_image_csv is not None:
        save_per_image_csv = Path(save_per_image_csv)
        save_per_image_csv.parent.mkdir(parents=True, exist_ok=True)
        per_image_df.to_csv(save_per_image_csv, index=False, encoding="utf-8-sig")

    print("=" * 85)
    print(f"Model: {model_name or pred_folder.name}")
    print(f"Matched files: {len(common_keys)}")
    print(f"Predictions without GT: {len(missing_gt)}")
    print(f"GT files without prediction: {len(missing_pred)}")
    print("-" * 85)
    print(f"IoU       : {summary['IoU_mean']:.4f} ± {summary['IoU_std']:.4f}")
    print(f"Dice      : {summary['Dice_mean']:.4f} ± {summary['Dice_std']:.4f}")
    print(f"Soft IoU  : {summary['SoftIoU_mean']:.4f} ± {summary['SoftIoU_std']:.4f}")
    print(f"Soft Dice : {summary['SoftDice_mean']:.4f} ± {summary['SoftDice_std']:.4f}")
    print(f"SSIM      : {summary['SSIM_mean']:.4f} ± {summary['SSIM_std']:.4f}")
    print("=" * 85)

    return summary, per_image_df


# ============================================================
# 5. خواندن لاگ‌های آموزشی JSONL
# ============================================================

def summarize_training_log(
    log_path: str | Path,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    اطلاعات اصلی آخرین ران یک مدل را از فایل JSONL استخراج می‌کند.

    شامل:
    - نام مدل
    - Hyperparameters
    - بهترین epoch بر اساس Val Accuracy
    - آخرین epoch
    - میانگین زمان/RAM/GPU Peak
    - نتایج نهایی Test Set و متریک‌های هر کلاس
    """
    log_path = Path(log_path)

    # ------------------------------------------------------------------
    # خواندن کل فایل و جدا کردن رکورد training از رکورد final_test
    # ------------------------------------------------------------------
    training_record = None
    test_acc = None
    test_loss = None
    test_size = None
    per_class_metrics = {}  # <-- ۱. متغیر ذخیره متریک‌های کلاس‌ها

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # آخرین رکورد training که epochs دارد را نگه می‌داریم
            if "epochs" in item and isinstance(item["epochs"], list) and item["epochs"]:
                training_record = item

            # آخرین رکورد final_test را نگه می‌داریم
            if item.get("event") == "final_test":
                test_acc = item.get("test_acc")
                test_loss = item.get("test_loss")
                test_size = item.get("test_size")
                per_class_metrics = item.get("per_class_metrics", {})  # <-- ۲. استخراج دیکشنری متریک‌ها

    if training_record is None:
        raise RuntimeError(f"No epoch information in log: {log_path}")

    hyperparameters = training_record.get("hyperparameters", {})
    epochs = training_record.get("epochs", [])

    if model_name is None:
        model_name = hyperparameters.get("model", log_path.stem)

    best_epoch_data = max(
        epochs,
        key=lambda item: item.get("val_acc", float("-inf")),
    )
    last_epoch_data = epochs[-1]

    summary: dict[str, Any] = {
        "model": model_name,
        "log_path": str(log_path.resolve()),
        "datetime": training_record.get("datetime"),
        "num_logged_epochs": len(epochs),

        # بهترین epoch
        "best_epoch": best_epoch_data.get("epoch"),
        "best_train_loss": best_epoch_data.get("train_loss"),
        "best_train_acc": best_epoch_data.get("train_acc"),
        "best_val_loss": best_epoch_data.get("val_loss"),
        "best_val_acc": best_epoch_data.get("val_acc"),

        # آخرین epoch
        "last_epoch": last_epoch_data.get("epoch"),
        "last_train_loss": last_epoch_data.get("train_loss"),
        "last_train_acc": last_epoch_data.get("train_acc"),
        "last_val_loss": last_epoch_data.get("val_loss"),
        "last_val_acc": last_epoch_data.get("val_acc"),

        # منابع
        "mean_epoch_time_sec": _safe_mean(epochs, "time_sec"),
        "mean_ram_mb": _safe_mean(epochs, "ram_mb"),
        "max_ram_mb": _safe_max(epochs, "ram_mb"),
        "mean_gpu_peak_mb": _safe_mean(epochs, "gpu_peak_mb"),
        "max_gpu_peak_mb": _safe_max(epochs, "gpu_peak_mb"),

        # نتایج تست کلی
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_size": test_size,
    }

    # ------------------------------------------------------------------
    # درج صریح متریک‌های کلاسی در Summary
    # ------------------------------------------------------------------
    if per_class_metrics:
        for cls_name, metrics in per_class_metrics.items():
            summary[f"{cls_name}_sens"] = metrics.get("sensitivity")
            summary[f"{cls_name}_spec"] = metrics.get("specificity")
            summary[f"{cls_name}_f1"] = metrics.get("f1_score")

    # Hyperparameterها
    for key, value in hyperparameters.items():
        summary[f"hp_{key}"] = value

    return summary

def get_model_weights_table(models_config: list, output_dir: Path = None, save_excel: bool = True) -> pd.DataFrame:
    """
    Reads weight_class directly from model_config string (e.g. 'MobileNet_V3_Large_Weights.DEFAULT')
    and extracts metadata without manually maintaining mapping dicts.
    """
    records = []

    for model in models_config:
        weight_str = model.get("weight_class")
        
        default_weight = None
        if weight_str:
            try:
                # تفکیک اسم کلاس وزن و ویژگی آن (مثلا MobileNet_V3_Large_Weights و DEFAULT)
                enum_name, attr_name = weight_str.split(".")
                weight_enum = getattr(tv_models, enum_name)
                default_weight = getattr(weight_enum, attr_name)
            except (AttributeError, ValueError):
                default_weight = None

        if default_weight is not None:
            meta = default_weight.meta
            metrics_dict = meta.get("_metrics", {})
            first_dataset = next(iter(metrics_dict.keys()), "ImageNet-1K")
            acc1 = metrics_dict.get(first_dataset, {}).get("acc@1", None)
            acc5 = metrics_dict.get(first_dataset, {}).get("acc@5", None)

            records.append({
                "model_name": model.get("name"),
                "weight_enum": str(default_weight),
                "pretrain_dataset": first_dataset,
                "pretrain_classes": len(meta.get("categories", [])),
                "params_count": meta.get("num_params", None),
                "min_input_size": str(meta.get("min_size", "N/A")),
                "imagenet_top1_acc": acc1,
                "imagenet_top5_acc": acc5,
                "recipe_url": meta.get("recipe", "N/A"),
            })
        else:
            records.append({
                "model_name": model.get("name"),
                "weight_enum": "Custom / Scratch / Non-TorchVision",
                "pretrain_dataset": "None / Custom",
                "pretrain_classes": None,
                "params_count": None,
                "min_input_size": None,
                "imagenet_top1_acc": None,
                "imagenet_top5_acc": None,
                "recipe_url": None,
            })

    df = pd.DataFrame(records)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "models_pretrained_weights_info.csv", index=False, encoding="utf-8-sig")
        if save_excel:
            df.to_excel(output_dir / "models_pretrained_weights_info.xlsx", index=False)

    return df

def _safe_mean(epoch_data: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(item[key])
        for item in epoch_data
        if item.get(key) is not None
    ]

    return float(np.mean(values)) if values else None


def _safe_max(epoch_data: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(item[key])
        for item in epoch_data
        if item.get(key) is not None
    ]

    return float(np.max(values)) if values else None

def epochs_to_dataframe(
    log_path: str | Path,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    همه‌ی اطلاعات epochهای آخرین Run یک مدل را به DataFrame تبدیل می‌کند.
    """
    log_path = Path(log_path)

    training_record = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # آخرین رکوردی که epochs دارد را نگه می‌داریم
            if "epochs" in item and isinstance(item["epochs"], list) and item["epochs"]:
                training_record = item

    if training_record is None:
        raise RuntimeError(f"No epoch information in log: {log_path}")

    hyperparameters = training_record.get("hyperparameters", {})
    epochs = training_record.get("epochs", [])

    if model_name is None:
        model_name = hyperparameters.get("model", log_path.stem)

    df = pd.DataFrame(epochs)
    df.insert(0, "model", model_name)
    df.insert(1, "run_datetime", training_record.get("datetime"))

    return df


# ============================================================
# 6. مقایسه‌ی جامع چند مدل
# ============================================================
def _empty_xai_summary(
    model_name: str,
    pred_folder: str | Path,
    gt_folder: Optional[str | Path],
    threshold: float,
) -> dict[str, Any]:
    """خلاصه‌ی XAI وقتی Annotation پزشکی در دسترس نیست."""
    pred_folder = Path(pred_folder)

    pred_count = 0
    if pred_folder.exists():
        pred_count = len(build_heatmap_file_map(pred_folder))

    return {
        "model": model_name,
        "heatmap_folder": str(pred_folder.resolve()),
        "gt_folder": str(Path(gt_folder).resolve()) if gt_folder is not None else None,
        "threshold": threshold,
        "matched_heatmaps": 0,
        "prediction_heatmaps": pred_count,
        "ground_truth_heatmaps": 0,
        "predictions_without_gt": pred_count,
        "ground_truth_without_prediction": 0,
        "IoU_mean": np.nan,
        "IoU_std": np.nan,
        "Dice_mean": np.nan,
        "Dice_std": np.nan,
        "SoftIoU_mean": np.nan,
        "SoftIoU_std": np.nan,
        "SoftDice_mean": np.nan,
        "SoftDice_std": np.nan,
        "SSIM_mean": np.nan,
        "SSIM_std": np.nan,
    }
def compare_models(
    model_configs: list[dict[str, Any]],
    gt_folder: Optional[str | Path] = None,
    threshold: float = 0.65,
    output_dir: str | Path = "model_comparison_results",
    save_excel: bool = True,
    enable_xai: bool = False,          # <-- جدید
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    مقایسه‌ی جامع مدل‌ها با یا بدون Soft Ground Truth پزشکی.

    اگر enable_xai=False باشد یا gt_folder در دسترس نباشد،
    فقط مقایسه‌ی آموزش، Accuracy، زمان و حافظه اجرا می‌شود.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(gt_folder) if gt_folder is not None else None
    gt_available = (
        enable_xai
        and gt_path is not None
        and gt_path.exists()
        and gt_path.is_dir()
        and any(gt_path.glob("*.npy"))
    )

    if gt_available:
        print(f"Soft Ground Truth found: {gt_path.resolve()}", flush=True)
    else:
        print(
            "[INFO] XAI evaluation is disabled or Ground Truth is unavailable. "
            "Only training / efficiency metrics will be reported.",
            flush=True,
        )

    model_summaries: list[dict[str, Any]] = []
    epoch_dfs: list[pd.DataFrame] = []
    heatmap_dfs: list[pd.DataFrame] = []

    for config in model_configs:
        # heatmap_folder فقط وقتی XAI فعال است اجباری است
        required_keys = {"name", "log_path"}
        if enable_xai:
            required_keys.add("heatmap_folder")

        missing = required_keys - set(config)
        if missing:
            raise ValueError(
                f"Model configuration is missing keys: {missing}. "
                f"Current config: {config}"
            )

        model_name = config["name"]

        print("\n" + "#" * 90)
        print(f"Processing model: {model_name}")
        print("#" * 90)

        training_summary = summarize_training_log(
            log_path=config["log_path"],
            model_name=model_name,
        )

        epochs_df = epochs_to_dataframe(
            log_path=config["log_path"],
            model_name=model_name,
        )

        if gt_available:
            per_image_csv = (
                output_dir / f"{_safe_filename(model_name)}_per_image_xai.csv"
            )

            xai_summary, per_image_df = evaluate_heatmaps_against_ground_truth(
                pred_folder=config["heatmap_folder"],
                gt_folder=gt_path,
                threshold=threshold,
                model_name=model_name,
                save_per_image_csv=per_image_csv,
            )
        else:
            xai_summary = _empty_xai_summary(
                model_name=model_name,
                pred_folder=config.get("heatmap_folder"),
                gt_folder=gt_folder,
                threshold=threshold,
            )

            per_image_df = pd.DataFrame(
                columns=[
                    "model",
                    "image_key",
                    "gt_file",
                    "pred_file",
                    "IoU",
                    "Dice",
                    "SoftIoU",
                    "SoftDice",
                    "SSIM",
                ]
            )

            print(
                f"[{model_name}] XAI evaluation skipped.",
                flush=True,
            )

        full_summary = {**training_summary, **xai_summary}

        model_summaries.append(full_summary)
        epoch_dfs.append(epochs_df)
        heatmap_dfs.append(per_image_df)

    model_summary_df = pd.DataFrame(model_summaries)

    all_epochs_df = (
        pd.concat(epoch_dfs, ignore_index=True)
        if epoch_dfs
        else pd.DataFrame()
    )

    all_heatmaps_df = (
        pd.concat(heatmap_dfs, ignore_index=True)
        if heatmap_dfs
        else pd.DataFrame()
    )

    # مرتب‌سازی: اول test_acc (اگر وجود داشته باشد)، بعد best_val_acc
    sort_columns = [
        column
        for column in ["test_acc", "best_val_acc", "SoftDice_mean", "SSIM_mean"]
        if (
            column in model_summary_df.columns
            and model_summary_df[column].notna().any()
        )
    ]

    if sort_columns:
        model_summary_df = model_summary_df.sort_values(
            by=sort_columns,
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # ذخیره فایل‌ها
    # ------------------------------------------------------------------
    summary_csv = output_dir / "models_comparison_summary.csv"
    epochs_csv = output_dir / "models_all_epochs.csv"
    heatmaps_csv = output_dir / "models_xai_per_image.csv"

    model_summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    all_epochs_df.to_csv(epochs_csv, index=False, encoding="utf-8-sig")
    all_heatmaps_df.to_csv(heatmaps_csv, index=False, encoding="utf-8-sig")

    if save_excel:
        excel_path = output_dir / "models_comparison.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            model_summary_df.to_excel(
                writer, sheet_name="Model Summary", index=False
            )
            all_epochs_df.to_excel(
                writer, sheet_name="All Epochs", index=False
            )
            all_heatmaps_df.to_excel(
                writer, sheet_name="XAI Per Image", index=False
            )

        print(f"\nExcel saved to: {excel_path.resolve()}")

    # ------------------------------------------------------------------
    # نمایش خلاصه
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Final model comparison summary")
    print("=" * 90)

    display_columns = [
        column
        for column in [
            "model",
            "best_epoch",
            "best_val_acc",
            "test_acc",          # <-- جدید
            "best_val_loss",
            "test_loss",         # <-- جدید
            "mean_epoch_time_sec",
            "mean_ram_mb",
            "max_gpu_peak_mb",
            "matched_heatmaps",
            "IoU_mean",
            "Dice_mean",
            "SoftIoU_mean",
            "SoftDice_mean",
            "SSIM_mean",
        ]
        if column in model_summary_df.columns
    ]

    print(model_summary_df[display_columns].to_string(index=False))

    print(f"\nCSV summary saved to: {summary_csv.resolve()}")
    print(f"CSV epochs saved to: {epochs_csv.resolve()}")
    print(f"CSV per-image XAI saved to: {heatmaps_csv.resolve()}")

    return model_summary_df, all_epochs_df, all_heatmaps_df


def _safe_filename(name: str) -> str:
    """تبدیل نام مدل به نام فایل امن."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")

# ============================================================
# 7. Model Comparison Plots
# ============================================================

def plot_radar_comparison(
    model_summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plots a multi-criteria radar chart comparing models across balanced metrics.
    Metrics are normalized between 0.1 and 1.0 for fair visual representation.
    Prefers test_acc over best_val_acc when available.
    """
    # Prefer test_acc if it exists and has valid values
    acc_metric = (
        "test_acc"
        if (
            "test_acc" in model_summary_df.columns
            and model_summary_df["test_acc"].notna().any()
        )
        else "best_val_acc"
    )

    metric_candidates = [
        (acc_metric, True),              # Higher is better
        ("IoU_mean", True),              # Higher is better
        ("Dice_mean", True),             # Higher is better
        ("mean_epoch_time_sec", False),  # Lower is better -> Invert
        ("max_gpu_peak_mb", False),      # Lower is better -> Invert
    ]

    available_metrics = [
        (m, higher)
        for m, higher in metric_candidates
        if m in model_summary_df.columns and model_summary_df[m].notna().any()
    ]
    if len(available_metrics) < 3:
        return

    metric_names = [m for m, _ in available_metrics]

    norm_df = model_summary_df[["model"] + metric_names].copy()
    for col, higher_is_better in available_metrics:
        series = norm_df[col]
        min_val = series.min(skipna=True)
        max_val = series.max(skipna=True)
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            norm_df[col] = 1.0
        else:
            if higher_is_better:
                norm_df[col] = 0.1 + 0.9 * ((series - min_val) / (max_val - min_val))
            else:
                norm_df[col] = 0.1 + 0.9 * ((max_val - series) / (max_val - min_val))

    labels = np.array(metric_names)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in norm_df.iterrows():
        values = row[metric_names].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    plt.title("Holistic Multi-Criteria Model Comparison", size=14, pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()

    radar_path = output_dir / "comparison_radar_chart.png"
    plt.savefig(radar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved radar comparison chart to: {radar_path}", flush=True)

def plot_tradeoff_bubble(
    model_summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Accuracy vs. Speed bubble chart with GPU peak memory size.

    Prefers test_acc over best_val_acc when available.
    """
    acc_col = (
        "test_acc"
        if (
            "test_acc" in model_summary_df.columns
            and model_summary_df["test_acc"].notna().any()
        )
        else "best_val_acc"
    )

    req_cols = {"mean_epoch_time_sec", acc_col, "model"}
    if model_summary_df is None or not req_cols.issubset(
        model_summary_df.columns
    ):
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = model_summary_df.dropna(subset=list(req_cols)).copy()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    size_col = "max_gpu_peak_mb" if "max_gpu_peak_mb" in df.columns else None

    sns.scatterplot(
        data=df,
        x="mean_epoch_time_sec",
        y=acc_col,
        hue="model",
        size=size_col,
        sizes=(120, 900),
        palette="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    # ساخت لیبل‌ها و نگهداری آن‌ها در یک لیست
    texts = []
    for _, r in df.iterrows():
        t = ax.text(
            r["mean_epoch_time_sec"],
            r[acc_col],
            str(r["model"]),
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="gray",
                lw=0.5,
                alpha=0.85,
            ),
        )
        texts.append(t)

    # تنظیم خودکار موقعیت لیبل‌ها جهت جلوگیری از همپوشانی
    if  texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                lw=0.7,
                alpha=0.7,
            ),
            expand=(1.2, 1.3),  # فضای مانور دور نقاط
            force_text=(0.5, 0.8),  # نیروی رانش بین متن‌ها
        )

    ylabel = (
        "Test Accuracy" if acc_col == "test_acc" else "Best Validation Accuracy"
    )
    ax.set(
        xlabel="Mean Epoch Time (seconds)",
        ylabel=ylabel,
        title=f"{ylabel} vs. Speed Trade-off (Bubble Size = GPU Peak MB)",
    )
    ax.grid(True, linestyle="--", alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            title="Model / GPU (MB)" if size_col else "Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            framealpha=0.95,
            fontsize=8.5,
        )

    fig.subplots_adjust(right=0.72, left=0.10, bottom=0.12, top=0.90)
    bubble_path = output_dir / "comparison_tradeoff_bubble.png"
    fig.savefig(
        bubble_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved tradeoff bubble chart to: {bubble_path}", flush=True)

def plot_xai_distribution(
    xai_per_image_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plots box and strip plots for per-image XAI metric distributions across models.
    """
    if xai_per_image_df is None or xai_per_image_df.empty:
        return

    xai_metrics = [m for m in ["IoU", "Dice", "SSIM"] if m in xai_per_image_df.columns]
    if not xai_metrics:
        return

    for metric in xai_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=xai_per_image_df,
            x="model",
            y=metric,
            palette="Set2",
            ax=ax,
        )
        sns.stripplot(
            data=xai_per_image_df,
            x="model",
            y=metric,
            color="black",
            alpha=0.3,
            jitter=0.2,
            size=4,
            ax=ax,
        )
        ax.set_title(f"Per-Image {metric} Distribution across Models", fontsize=12)
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        dist_path = output_dir / f"comparison_xai_distribution_{metric.lower()}.png"
        plt.savefig(dist_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved XAI {metric} distribution chart to: {dist_path}", flush=True)


def plot_model_comparison(
    model_summary_df: pd.DataFrame,
    output_dir: str | Path = "model_comparison_results",
    xai_per_image_df: pd.DataFrame | None = None,
) -> None:
    """
    Generates and saves comprehensive comparison charts:
    1. Train / Val / Test Accuracy (grouped bars)
    2. Train / Val / Test Loss (grouped bars)
    3. Mean Epoch Time Bar Chart
    4. Resource (RAM/GPU) Consumption Chart
    5. XAI Heatmap Metrics Bar Chart (if available)
    6. Multi-criteria Radar Chart
    7. Accuracy vs. Speed Trade-off Bubble Chart
    8. Per-image XAI Distribution Boxplots (if xai_per_image_df is provided)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = model_summary_df.copy()
    models = df["model"].tolist()
    x = range(len(models))

    # ------------------------------------------------------------------
    # 1. Train / Val / Test Accuracy (Grouped Bar)
    # ------------------------------------------------------------------
    acc_cols = {
        "best_train_acc": "Train Acc",
        "best_val_acc": "Val Acc",
        "test_acc": "Test Acc",
    }
    available_acc = {
        k: v for k, v in acc_cols.items()
        if k in df.columns and df[k].notna().any()
    }

    if available_acc:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        width = 0.8 / len(available_acc)
        colors = ["#4C72B0", "#55A868", "#C44E52"]

        for i, (col, label) in enumerate(available_acc.items()):
            offset = (i - (len(available_acc) - 1) / 2) * width
            values = df[col].fillna(0)
            bars = ax.bar(
                [xi + offset for xi in x],
                values,
                width=width,
                label=label,
                color=colors[i % len(colors)],
            )
            for bar, val in zip(bars, df[col]):
                if pd.notna(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        ax.set_title("Train / Validation / Test Accuracy Comparison")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_train_val_test_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Train / Val / Test Loss (Grouped Bar)
    # ------------------------------------------------------------------
    loss_cols = {
        "best_train_loss": "Train Loss",
        "best_val_loss": "Val Loss",
        "test_loss": "Test Loss",
    }
    available_loss = {
        k: v for k, v in loss_cols.items()
        if k in df.columns and df[k].notna().any()
    }

    if available_loss:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        width = 0.8 / len(available_loss)
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        max_loss = df[list(available_loss.keys())].max().max()

        for i, (col, label) in enumerate(available_loss.items()):
            offset = (i - (len(available_loss) - 1) / 2) * width
            values = df[col].fillna(0)
            bars = ax.bar(
                [xi + offset for xi in x],
                values,
                width=width,
                label=label,
                color=colors[i % len(colors)],
            )
            for bar, val in zip(bars, df[col]):
                if pd.notna(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02 * max_loss,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        ax.set_title("Train / Validation / Test Loss Comparison")
        ax.set_ylabel("Loss")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_train_val_test_loss.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Mean Epoch Time
    # ------------------------------------------------------------------
    if "mean_epoch_time_sec" in df.columns and df["mean_epoch_time_sec"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(df["model"], df["mean_epoch_time_sec"], color="darkorange")
        ax.set_title("Average Epoch Time Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Epoch Time (seconds)")
        ax.set_xticks(range(len(df["model"])))
        ax.set_xticklabels(df["model"], rotation=25, ha="right")

        ymax = df["mean_epoch_time_sec"].max()
        for bar, value in zip(bars, df["mean_epoch_time_sec"]):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015 * ymax,
                    f"{value:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_epoch_time.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4. RAM & GPU Peak
    # ------------------------------------------------------------------
    resource_columns = [
        col
        for col in ["mean_ram_mb", "max_gpu_peak_mb"]
        if col in df.columns and df[col].notna().any()
    ]
    if resource_columns:
        ax = df.set_index("model")[resource_columns].plot(
            kind="bar",
            figsize=(11, 5),
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title("Memory Consumption Comparison", fontsize=12, pad=12)
        ax.set_xlabel("Model")
        ax.set_ylabel("Memory (MB)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # اضافه کردن لیبل مقادیر روی هر ستون
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.0f",  # یا '%.1f' برای یک رقم اعشار
                padding=3,  # فاصله متن تا بالای ستون
                fontsize=8,
                rotation=0,  # در صورت بلند بودن اعداد می‌توانید 45 بگذارید
            )

        # افزایش 10 درصدی سقف محور Y برای جلوگیری از بریده شدن لیبل‌های بالاترین ستون
        ax.margins(y=0.12)

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_memory_usage.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(ax.get_figure())


    # ------------------------------------------------------------------
    # 5. XAI Metrics (only if meaningful data exists)
    # ------------------------------------------------------------------
    xai_columns = [
        col for col in [
            "IoU_mean", "Dice_mean", "SoftIoU_mean", "SoftDice_mean", "SSIM_mean"
        ]
        if col in df.columns and df[col].notna().any()
    ]
    if xai_columns:
        ax = df.set_index("model")[xai_columns].plot(kind="bar", figsize=(13, 6))
        ax.set_title("XAI Heatmap vs Medical Annotation Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_xai_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(ax.get_figure())

    # ------------------------------------------------------------------
    # 6 & 7. Radar + Trade-off
    # ------------------------------------------------------------------
    plot_radar_comparison(model_summary_df=df, output_dir=output_dir)
    plot_tradeoff_bubble(model_summary_df=df, output_dir=output_dir)

    # ------------------------------------------------------------------
    # 8. XAI per-image distributions
    # ------------------------------------------------------------------
    if xai_per_image_df is not None and not xai_per_image_df.empty:
        plot_xai_distribution(xai_per_image_df=xai_per_image_df, output_dir=output_dir)

    print(
        f"All comparison plots successfully saved in: {output_dir.resolve()}",
        flush=True,
    )