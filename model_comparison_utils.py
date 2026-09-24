from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from adjustText import adjust_text
import torchvision.models as tv_models

# ============================================================
# 0. وزن مدل ها و اطلاعات آن (models parameters)
# ============================================================
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

# ============================================================
# 1. معیارهای مقایسه Heatmap (XAI Metrics)
# ============================================================
def iou(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.65) -> float:
    """IoU دودویی پس از threshold کردن نقشه‌ها."""
    gt_bin = gt >= threshold
    pred_bin = pred >= threshold
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
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
# 2. آماده‌سازی و استانداردسازی Heatmap
# ============================================================
def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """نرمال‌سازی نقشه در بازه [0, 1]."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.squeeze(heatmap)
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D after squeeze. Got shape: {heatmap.shape}")
    min_val = float(heatmap.min())
    max_val = float(heatmap.max())
    heatmap = heatmap - min_val
    if max_val - min_val > 1e-8:
        heatmap = heatmap / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap.astype(np.float32)


def resize_heatmap_if_needed(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """تغییر اندازه نقشه به ابعاد Ground Truth در صورت لزوم."""
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
# 3. استخراج کلید تصویر و کشف فایل‌ها
# ============================================================
def extract_image_key(filename_or_path: str | Path) -> str:
    """استخراج کلید یکتا تصویر با پوشش کامل تعداد آندرلاین‌ها."""
    name = Path(filename_or_path).name
    match = re.search(r"(salmon_(?:dis|fresh)_*\d+)", name, flags=re.IGNORECASE)
    if match:
        raw_key = match.group(1).lower()
        return re.sub(r"_+", "_", raw_key)
    clean_stem = Path(name).stem.replace("_heatmap", "").replace(".npy", "").strip().lower()
    return re.sub(r"_+", "_", clean_stem)


def build_heatmap_file_map(folder: str | Path) -> dict[str, Path]:
    """تهیه دیکشنری نگاشت کلید تصویر به آدرس فایل."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")
    file_map: dict[str, Path] = {}
    for file_path in folder.glob("*.npy"):
        key = extract_image_key(file_path.name)
        file_map[key] = file_path
    return file_map


def auto_discover_seed_logs(
    target_path_or_pattern: Union[str, Path, list],
    root_dir: Optional[Path] = None,
    pattern: str = "*.jsonl",
) -> list[Path]:
    """کشف هوشمند و خودکار تمامی فایل‌های JSONL لاگ‌ها."""
    if root_dir is None:
        root_dir = Path(".")

    if isinstance(target_path_or_pattern, list):
        found_paths = []
        for item in target_path_or_pattern:
            found_paths.extend(auto_discover_seed_logs(item, root_dir, pattern))
        return sorted(list(set(found_paths)))

    p = Path(target_path_or_pattern)
    full_path = p if p.is_absolute() else (root_dir / p)

    if "*" in full_path.name or "?" in full_path.name:
        parent_dir = full_path.parent
        if parent_dir.exists():
            return sorted(list(parent_dir.glob(full_path.name)))
        return []

    if full_path.is_file():
        return [full_path]

    if full_path.is_dir():
        files = list(full_path.glob(pattern))
        if not files:
            files = list(full_path.rglob(pattern))
        return sorted(files)

    return []


# ============================================================
# 4. ارزیابی کامل XAI
# ============================================================
def evaluate_heatmaps_against_ground_truth(
    pred_folder: str | Path,
    gt_folder: str | Path,
    threshold: float = 0.65,
    model_name: Optional[str] = None,
    save_per_image_csv: Optional[str | Path] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    gt_map = build_heatmap_file_map(gt_folder)
    pred_map = build_heatmap_file_map(pred_folder)

    common_keys = sorted(set(gt_map) & set(pred_map))
    if not common_keys:
        print(f"⚠️ [XAI Warning] No matching heatmaps found between {gt_folder.name} and {pred_folder.name}")
        return {}, pd.DataFrame()

    records = []
    for key in common_keys:
        try:
            gt = np.load(gt_map[key])
            pred = np.load(pred_map[key])

            gt = normalize_heatmap(gt)
            pred = normalize_heatmap(pred)
            pred = resize_heatmap_if_needed(pred, gt.shape)

            records.append({
                "image_key": key,
                "model": model_name or pred_folder.name,
                "IoU": iou(gt, pred, threshold),
                "Dice": dice(gt, pred, threshold),
                "SoftIoU": soft_iou(gt, pred),
                "SoftDice": soft_dice(gt, pred),
                "SSIM": float(ssim(gt, pred, data_range=1.0)),
            })
        except Exception as e:
            print(f"⚠️ Failed on {key}: {e}")
            continue

    if not records:
        return {}, pd.DataFrame()

    per_image_df = pd.DataFrame(records)

    summary = {
        "model": model_name or pred_folder.name,
        "num_images": len(per_image_df),
        "IoU_mean": float(per_image_df["IoU"].mean()),
        "IoU_std": float(per_image_df["IoU"].std()),
        "Dice_mean": float(per_image_df["Dice"].mean()),
        "Dice_std": float(per_image_df["Dice"].std()),
        "SoftIoU_mean": float(per_image_df["SoftIoU"].mean()),
        "SoftIoU_std": float(per_image_df["SoftIoU"].std()),
        "SoftDice_mean": float(per_image_df["SoftDice"].mean()),
        "SoftDice_std": float(per_image_df["SoftDice"].std()),
        "SSIM_mean": float(per_image_df["SSIM"].mean()),
        "SSIM_std": float(per_image_df["SSIM"].std()),
        "gt_folder": str(gt_folder),
        "heatmap_folder": str(pred_folder),
    }

    if save_per_image_csv is not None:
        save_path = Path(save_per_image_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        per_image_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return summary, per_image_df


# ============================================================
# 5. خلاصه‌سازی جامع لاگ‌های آموزشی
# ============================================================
def summarize_training_log(log_path: str | Path, model_name: Optional[str] = None) -> list[dict[str, Any]]:
    """
    پیمایش کامل فایل JSONL و استخراج تک‌تک اجراها (حتی اگر چند اجرای K-Fold/Seed در یک فایل باشند)
    """
    log_path = Path(log_path)
    runs = []
    
    current_train_record = None
    current_test_record = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # اگر سطر مربوط به آموزش اپوک‌ها باشد
            if "epochs" in item and isinstance(item["epochs"], list) and item["epochs"]:
                current_train_record = item

            # اگر سطر مربوط به تست نهایی باشد
            if item.get("event") == "final_test":
                current_test_record = item
                
                # وقتی هم آموزش و هم تست این فولد پیدا شد، یک Run کامل تشکیل می‌دهیم
                if current_train_record is not None:
                    hparams = current_train_record.get("hyperparameters", {})
                    # اگر در final_test اطلاعات تکمیلی بود ترجیح داده شود
                    fold_id = item.get("fold") or hparams.get("fold", "1")
                    seed_id = item.get("seed") or hparams.get("seed", "42")
                    
                    epochs = current_train_record.get("epochs", [])
                    best_epoch_data = max(epochs, key=lambda it: it.get("val_acc", float("-inf")))
                    last_epoch_data = epochs[-1]

                    def _mean(k):
                        vals = [float(x[k]) for x in epochs if x.get(k) is not None]
                        return float(np.mean(vals)) if vals else None

                    def _max(k):
                        vals = [float(x[k]) for x in epochs if x.get(k) is not None]
                        return float(np.max(vals)) if vals else None

                    m_name = model_name or item.get("model") or hparams.get("model", log_path.stem)

                    run_summary = {
                        "model": m_name,
                        "seed": str(seed_id),
                        "fold": str(fold_id),
                        "log_path": str(log_path.resolve()),
                        "datetime": current_train_record.get("datetime"),
                        "num_logged_epochs": len(epochs),
                        # نتایج اپوک‌ها
                        "best_epoch": best_epoch_data.get("epoch"),
                        "best_train_loss": best_epoch_data.get("train_loss"),
                        "best_train_acc": best_epoch_data.get("train_acc"),
                        "best_val_loss": best_epoch_data.get("val_loss"),
                        "best_val_acc": best_epoch_data.get("val_acc"),
                        "last_epoch": last_epoch_data.get("epoch"),
                        "last_train_loss": last_epoch_data.get("train_loss"),
                        "last_train_acc": last_epoch_data.get("train_acc"),
                        "last_val_loss": last_epoch_data.get("val_loss"),
                        "last_val_acc": last_epoch_data.get("val_acc"),
                        # زمان و منابع
                        "total_train_time_sec": sum(x.get("time_sec", 0) for x in epochs),
                        "mean_epoch_time_sec": _mean("time_sec"),
                        "mean_ram_mb": _mean("ram_mb"),
                        "max_ram_mb": _max("ram_mb"),
                        "mean_gpu_peak_mb": _mean("gpu_peak_mb"),
                        "max_gpu_peak_mb": _max("gpu_peak_mb"),
                        # نتایج تست نهایی
                        "test_acc": item.get("test_acc"),
                        "test_loss": item.get("test_loss"),
                        "test_size": item.get("test_size"),
                    }

                    # استخراج متریک‌های پرکلاس
                    per_class = item.get("per_class_metrics", {})
                    for cls_name, metrics in per_class.items():
                        run_summary[f"{cls_name}_sens"] = metrics.get("sensitivity") or metrics.get("recall")
                        run_summary[f"{cls_name}_spec"] = metrics.get("specificity")
                        run_summary[f"{cls_name}_prec"] = metrics.get("precision")
                        run_summary[f"{cls_name}_f1"] = metrics.get("f1_score") or metrics.get("f1")

                    # اضافه کردن سایر هایپرپارامترها (منهای seed و fold که خودمان ساختیم)
                    for k, v in hparams.items():
                        if k not in ["seed", "fold", "model"]:
                            run_summary[f"hp_{k}"] = v

                    runs.append(run_summary)
                    # ریست برای فولد بعدی
                    current_train_record = None
                    current_test_record = None

    if not runs:
        raise RuntimeError(f"No valid run pairs (epochs + final_test) found in log: {log_path}")

    return runs


def epochs_to_dataframe(log_path: str | Path, model_name: Optional[str] = None) -> pd.DataFrame:
    log_path = Path(log_path)
    training_record = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if "epochs" in item and isinstance(item["epochs"], list) and item["epochs"]:
                    training_record = item
            except json.JSONDecodeError:
                continue

    if training_record is None:
        return pd.DataFrame()

    df = pd.DataFrame(training_record.get("epochs", []))
    if model_name is None:
        model_name = training_record.get("hyperparameters", {}).get("model", log_path.stem)
    df.insert(0, "model", model_name)
    return df


# ============================================================
# 6. پردازشگر اصلی مقایسه چند سیده (Core Benchmark Aggregator)
# ============================================================
def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")

def compare_models(
    model_configs: list[dict[str, Any]],
    root_dir: Optional[str | Path] = None,
    gt_folder: Optional[str | Path] = None,
    threshold: float = 0.65,
    output_dir: str | Path = "model_comparison_results",
    save_excel: bool = True,
    enable_xai: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    اجرای بنچمارک جامع مقایسه‌ای با تجمیع آماری کامل تمام ستون‌ها،
    محاسبه Macro F1 و نمایش هوشمند مقادیر ثابت بدون ±0.00.
    """
    root_dir = Path(root_dir) if root_dir else Path(".")
    output_dir = Path(output_dir) if Path(output_dir).is_absolute() else (root_dir / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(gt_folder) if gt_folder is not None else None
    if gt_path and not gt_path.is_absolute():
        gt_path = root_dir / gt_path

    raw_runs: list[dict[str, Any]] = []
    epoch_dfs: list[pd.DataFrame] = []
    heatmap_dfs: list[pd.DataFrame] = []

    for cfg in model_configs:
        model_name = cfg["name"]
        target = cfg.get("log_paths") or cfg.get("log_path") or cfg.get("log_pattern")
        if not target:
            target = f"*{_safe_filename(model_name)}*.jsonl"

        discovered_logs = auto_discover_seed_logs(target, root_dir=root_dir)
        if not discovered_logs:
            print(f"⚠️ [SKIP] No JSONL log found for: {model_name}")
            continue

        print(f"📦 Model '{model_name}': Found {len(discovered_logs)} run(s)/seed(s)")

        for log_p in discovered_logs:
            file_runs = summarize_training_log(log_p, model_name=model_name)
            for r in file_runs:
                r["log_file"] = log_p.name
                raw_runs.append(r)

    if not raw_runs:
        raise RuntimeError("❌ No valid training logs were loaded! Please verify your directories.")

    raw_runs_df = pd.DataFrame(raw_runs)

    # **۱. محاسبه Macro F1 به ازای هر اجرا (رویداد تکی)**
    f1_cols = [c for c in raw_runs_df.columns if c.endswith("_f1") and c != "macro_f1"]
    if f1_cols:
        raw_runs_df["macro_f1"] = raw_runs_df[f1_cols].mean(axis=1)

    all_epochs_df = pd.concat(epoch_dfs, ignore_index=True) if epoch_dfs else pd.DataFrame()
    all_heatmaps_df = pd.concat(heatmap_dfs, ignore_index=True) if heatmap_dfs else pd.DataFrame()

    # **۲. شناسایی خودکار تمامی ستون‌های عددی برای میانگین و انحراف معیار**
    exclude_from_agg = {"model", "seed", "fold", "log_path", "log_file", "datetime", "gt_folder", "heatmap_folder"}
    all_numeric_cols = [
        col for col in raw_runs_df.select_dtypes(include=[np.number]).columns
        if col not in exclude_from_agg
    ]

    agg_dict = {c: ["mean", "std"] for c in all_numeric_cols}
    agg_dict["log_file"] = "count"

    stats_df = raw_runs_df.groupby("model").agg(agg_dict)
    stats_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in stats_df.columns]
    stats_df = stats_df.rename(columns={"log_file_count": "runs"}).reset_index()

    # **۳. ساخت جدول تجمیعی مقاله (Paper Summary) با فرمت‌دهی هوشمند**
    paper_table = pd.DataFrame({
        "model": stats_df["model"],
        "runs": stats_df["runs"],
    })

    # پارامترها و حجم مدل
    if "num_params_mean" in stats_df.columns:
        paper_table["num_params"] = stats_df["num_params_mean"].apply(
            lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x > 0 else "N/A"
        )
    if "model_size_mb_mean" in stats_df.columns:
        paper_table["model_size_mb"] = stats_df["model_size_mb_mean"].apply(
            lambda x: f"{x:.2f} MB" if pd.notna(x) and x > 0 else "N/A"
        )

    # تابع محلی جهت قالب‌بندی هوشمند
    def _smart_format(col_name: str, m: float, s: float) -> str:
        if pd.isna(m):
            return "N/A"

        is_zero_std = (pd.isna(s) or abs(s) < 1e-7)
        lower_col = col_name.lower()

        # الف) درصدها (Accuracy, F1, Sensitivity, Specificity, Precision)
        if any(term in lower_col for term in ["acc", "f1", "sens", "spec", "prec", "iou", "dice"]):
            if m <= 1.0:
                m_val, s_val = m * 100, s * 100
            else:
                m_val, s_val = m, s
            
            return f"{m_val:.2f}%" if is_zero_std else f"{m_val:.2f} ± {s_val:.2f}%"

        # ب) زمان (Seconds)
        elif "time" in lower_col or "sec" in lower_col:
            return f"{m:.2f}s" if is_zero_std else f"{m:.2f}s ± {s:.2f}s"

        # ج) حافظه (MB/RAM/GPU)
        elif "mb" in lower_col or "ram" in lower_col or "gpu" in lower_col:
            return f"{m:.1f} MB" if is_zero_std else f"{m:.1f} ± {s:.1f} MB"

        # د) سایر اعداد صحیح یا هایپرپارامترهای ثابت
        else:
            # اگر عدد اعشار ندارد یا بسیار نزدیک به صحیح است
            m_str = f"{int(m)}" if abs(m - round(m)) < 1e-5 else f"{m:.2f}"
            if is_zero_std:
                return m_str
            s_str = f"{int(s)}" if abs(s - round(s)) < 1e-5 else f"{s:.2f}"
            return f"{m_str} ± {s_str}"

    # اعمال فرمت‌دهی هوشمند روی تمام ستون‌های عددی
    for col in all_numeric_cols:
        if col in ["num_params", "trainable_params", "model_size_mb"]:
            continue

        m_col = f"{col}_mean"
        s_col = f"{col}_std"

        if m_col not in stats_df.columns:
            continue

        paper_table[col] = [
            _smart_format(col, m, s)
            for m, s in zip(stats_df[m_col], stats_df[s_col].fillna(0.0))
        ]

    # **۴. ترتیب‌بندی ستون‌ها (انتقال macro_f1 به بعد از آخرین F1-score کلاس)**
    cols = list(paper_table.columns)
    if "macro_f1" in cols:
        cols.remove("macro_f1")
        # پیدا کردن موقعیت آخرین F1 مربوط به کلاس‌ها
        f1_indices = [i for i, c in enumerate(cols) if "_f1" in c]
        insert_pos = (max(f1_indices) + 1) if f1_indices else len(cols)
        cols.insert(insert_pos, "macro_f1")
        paper_table = paper_table[cols]

    # **۵. مرتب‌سازی ردیف‌ها بر اساس بالاترین دقت تست**
    sort_key = "test_acc_mean" if "test_acc_mean" in stats_df.columns else "best_val_acc_mean"
    if sort_key in stats_df.columns:
        s_idx = stats_df.sort_values(by=sort_key, ascending=False).index
        stats_df = stats_df.loc[s_idx].reset_index(drop=True)
        paper_table = paper_table.loc[s_idx].reset_index(drop=True)

    # **۶. ذخیره‌سازی نتایج در فایل‌های CSV و Excel**
    paper_table.to_csv(output_dir / "paper_summary_mean_std.csv", index=False, encoding="utf-8-sig")
    stats_df.to_csv(output_dir / "models_aggregated_stats.csv", index=False, encoding="utf-8-sig")
    raw_runs_df.to_csv(output_dir / "all_runs_raw.csv", index=False, encoding="utf-8-sig")

    if save_excel:
        excel_p = output_dir / "models_multi_seed_comparison.xlsx"
        with pd.ExcelWriter(excel_p, engine="openpyxl") as writer:
            paper_table.to_excel(writer, sheet_name="Paper Summary (Mean±Std)", index=False)
            stats_df.to_excel(writer, sheet_name="Aggregated Stats", index=False)
            raw_runs_df.to_excel(writer, sheet_name="All Runs (Raw)", index=False)
            if not all_epochs_df.empty:
                all_epochs_df.to_excel(writer, sheet_name="All Epochs", index=False)
            if not all_heatmaps_df.empty:
                all_heatmaps_df.to_excel(writer, sheet_name="XAI Heatmaps", index=False)

    return stats_df, paper_table, raw_runs_df, all_epochs_df


# ============================================================
# 7. توابع ترسیم نمودارها (Visualizations)
# ============================================================
def plot_radar_comparison(stats_df: pd.DataFrame, output_dir: Path) -> None:
    acc_col = "test_acc_mean" if "test_acc_mean" in stats_df.columns else "best_val_acc_mean"
    metric_candidates = [
        (acc_col, True),
        ("SoftDice_mean", True),
        ("SSIM_mean", True),
        ("mean_epoch_time_sec_mean", False),
        ("max_gpu_peak_mb_mean", False),
    ]
    avail = [(m, higher) for m, higher in metric_candidates if m in stats_df.columns and stats_df[m].notna().any()]
    if len(avail) < 3:
        return

    metric_names = [m for m, _ in avail]
    clean_labels = [m.replace("_mean", "").replace("_", " ").title() for m in metric_names]

    norm_df = stats_df[["model"] + metric_names].copy()
    for col, higher in avail:
        s = norm_df[col]
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mn == mx:
            norm_df[col] = 1.0
        else:
            norm_df[col] = 0.1 + 0.9 * ((s - mn) / (mx - mn) if higher else (mx - s) / (mx - mn))

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in norm_df.iterrows():
        vals = row[metric_names].tolist() + [row[metric_names].tolist()[0]]
        ax.plot(angles, vals, linewidth=2, label=row["model"])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), clean_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.title("Multi-Criteria Trade-off Analysis (Radar)", size=13, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_bubble(stats_df: pd.DataFrame, output_dir: Path) -> None:
    acc_col = "test_acc_mean" if "test_acc_mean" in stats_df.columns else "best_val_acc_mean"
    time_col = "mean_epoch_time_sec_mean"

    if acc_col not in stats_df.columns or time_col not in stats_df.columns:
        return

    df = stats_df.dropna(subset=[acc_col, time_col]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    size_col = "max_gpu_peak_mb_mean" if "max_gpu_peak_mb_mean" in df.columns else None

    sns.scatterplot(
        data=df, x=time_col, y=acc_col, hue="model",
        size=size_col, sizes=(150, 1000), palette="tab10", alpha=0.85, ax=ax
    )

    texts = [ax.text(r[time_col], r[acc_col], str(r["model"]), fontsize=9) for _, r in df.iterrows()]
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))

    ax.set_xlabel("Mean Epoch Time (seconds)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy vs. Training Speed Trade-off", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_tradeoff_bubble.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    stats_df: pd.DataFrame,
    output_dir: str | Path = "model_comparison_results",
    xai_per_image_df: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = stats_df["model"].astype(str)

    # ------------------------------------------------------------------
    # Helper: رسم bar chart ساده با (یا بدون) error bar
    # ------------------------------------------------------------------
    def _bar_with_optional_error(
        ax,
        values,
        stds=None,
        color="#2b5c8f",
        ylabel="",
        title="",
        value_fmt="{:.1f}",
        unit="",
    ):
        x = np.arange(len(models))
        bars = ax.bar(
            x,
            values,
            yerr=stds if stds is not None else None,
            capsize=5,
            color=color,
            alpha=0.85,
            edgecolor="black",
            width=0.65,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # برچسب روی میله‌ها
        for i, (bar, v) in enumerate(zip(bars, values)):
            height = bar.get_height()
            err = stds[i] if stds is not None else 0
            label = value_fmt.format(v)
            if stds is not None and err > 0:
                label += f"±{value_fmt.format(err)}"
            label += unit
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + err + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ------------------------------------------------------------------
    # 1. Accuracy (با error bar)
    # ------------------------------------------------------------------
    acc_col = "test_acc" if "test_acc" in stats_df.columns else "best_val_acc"
    mean_col = f"{acc_col}_mean"
    std_col = f"{acc_col}_std"

    if mean_col in stats_df.columns:
        means = stats_df[mean_col] * 100
        stds = stats_df[std_col].fillna(0.0) * 100 if std_col in stats_df.columns else None
    else:
        means = stats_df[acc_col] * 100
        stds = None

    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_with_optional_error(
        ax,
        means,
        stds,
        color="#2b5c8f",
        ylabel="Accuracy (%)",
        title="Model Performance (Mean ± Std across Seeds)" if stds is not None else "Model Performance",
        value_fmt="{:.1f}",
        unit="%",
    )
    ax.set_ylim(0, max(110, means.max() * 1.15 if len(means) else 110))
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_accuracy_with_errorbars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Total Training Time
    # ------------------------------------------------------------------
    time_col = "total_train_time_sec"
    mean_col = f"{time_col}_mean"
    std_col = f"{time_col}_std"

    if mean_col in stats_df.columns:
        times = stats_df[mean_col] / 60  # تبدیل به دقیقه
        time_stds = stats_df[std_col].fillna(0.0) / 60 if std_col in stats_df.columns else None
    elif time_col in stats_df.columns:
        times = stats_df[time_col] / 60
        time_stds = None
    else:
        times = None

    if times is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        _bar_with_optional_error(
            ax,
            times,
            time_stds,
            color="#e67e22",
            ylabel="Total Training Time (minutes)",
            title="Training Time Comparison",
            value_fmt="{:.1f}",
            unit=" min",
        )
        ax.set_ylim(0, times.max() * 1.25 if len(times) else 1)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_training_time.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. RAM Usage (Mean & Max)
    # ------------------------------------------------------------------
    has_ram = "mean_ram_mb" in stats_df.columns or "mean_ram_mb_mean" in stats_df.columns
    if has_ram:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mean RAM
        mean_col = "mean_ram_mb_mean" if "mean_ram_mb_mean" in stats_df.columns else "mean_ram_mb"
        std_col = "mean_ram_mb_std"
        ram_mean = stats_df[mean_col]
        ram_std = stats_df[std_col].fillna(0.0) if std_col in stats_df.columns else None

        _bar_with_optional_error(
            axes[0],
            ram_mean,
            ram_std,
            color="#27ae60",
            ylabel="Mean RAM (MB)",
            title="Mean RAM Usage",
            value_fmt="{:.0f}",
            unit=" MB",
        )
        axes[0].set_ylim(0, ram_mean.max() * 1.25)

        # Max RAM
        max_col = "max_ram_mb_mean" if "max_ram_mb_mean" in stats_df.columns else "max_ram_mb"
        std_col = "max_ram_mb_std"
        ram_max = stats_df[max_col]
        ram_max_std = stats_df[std_col].fillna(0.0) if std_col in stats_df.columns else None

        _bar_with_optional_error(
            axes[1],
            ram_max,
            ram_max_std,
            color="#16a085",
            ylabel="Max RAM (MB)",
            title="Peak RAM Usage",
            value_fmt="{:.0f}",
            unit=" MB",
        )
        axes[1].set_ylim(0, ram_max.max() * 1.25)

        plt.tight_layout()
        plt.savefig(output_dir / "comparison_ram_usage.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4. GPU Peak Memory (Mean & Max)
    # ------------------------------------------------------------------
    has_gpu = "mean_gpu_peak_mb" in stats_df.columns or "mean_gpu_peak_mb_mean" in stats_df.columns
    if has_gpu:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mean GPU Peak
        mean_col = "mean_gpu_peak_mb_mean" if "mean_gpu_peak_mb_mean" in stats_df.columns else "mean_gpu_peak_mb"
        std_col = "mean_gpu_peak_mb_std"
        gpu_mean = stats_df[mean_col]
        gpu_std = stats_df[std_col].fillna(0.0) if std_col in stats_df.columns else None

        _bar_with_optional_error(
            axes[0],
            gpu_mean,
            gpu_std,
            color="#8e44ad",
            ylabel="Mean GPU Peak (MB)",
            title="Mean GPU Peak Memory",
            value_fmt="{:.0f}",
            unit=" MB",
        )
        axes[0].set_ylim(0, gpu_mean.max() * 1.25)

        # Max GPU Peak
        max_col = "max_gpu_peak_mb_mean" if "max_gpu_peak_mb_mean" in stats_df.columns else "max_gpu_peak_mb"
        std_col = "max_gpu_peak_mb_std"
        gpu_max = stats_df[max_col]
        gpu_max_std = stats_df[std_col].fillna(0.0) if std_col in stats_df.columns else None

        _bar_with_optional_error(
            axes[1],
            gpu_max,
            gpu_max_std,
            color="#9b59b6",
            ylabel="Max GPU Peak (MB)",
            title="Peak GPU Memory",
            value_fmt="{:.0f}",
            unit=" MB",
        )
        axes[1].set_ylim(0, gpu_max.max() * 1.25)

        plt.tight_layout()
        plt.savefig(output_dir / "comparison_gpu_usage.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Combined Resource Overview (اختیاری اما خیلی مفید)
    # ------------------------------------------------------------------
    if has_ram and has_gpu and times is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Time
        _bar_with_optional_error(
            axes[0],
            times,
            time_stds,
            color="#e67e22",
            ylabel="Time (min)",
            title="Training Time",
            value_fmt="{:.1f}",
        )
        axes[0].set_ylim(0, times.max() * 1.25)

        # Max RAM
        max_ram_col = "max_ram_mb_mean" if "max_ram_mb_mean" in stats_df.columns else "max_ram_mb"
        _bar_with_optional_error(
            axes[1],
            stats_df[max_ram_col],
            None,
            color="#27ae60",
            ylabel="Max RAM (MB)",
            title="Peak RAM",
            value_fmt="{:.0f}",
        )
        axes[1].set_ylim(0, stats_df[max_ram_col].max() * 1.25)

        # Max GPU
        max_gpu_col = "max_gpu_peak_mb_mean" if "max_gpu_peak_mb_mean" in stats_df.columns else "max_gpu_peak_mb"
        _bar_with_optional_error(
            axes[2],
            stats_df[max_gpu_col],
            None,
            color="#8e44ad",
            ylabel="Max GPU (MB)",
            title="Peak GPU Memory",
            value_fmt="{:.0f}",
        )
        axes[2].set_ylim(0, stats_df[max_gpu_col].max() * 1.25)

        fig.suptitle("Resource Usage Overview", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_resources_overview.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Radar & Bubble (همون قبلی‌ها)
    plot_radar_comparison(stats_df, output_dir)
    plot_tradeoff_bubble(stats_df, output_dir)

    print(f"✅ Plots saved to: {output_dir}")