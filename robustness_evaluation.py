"""
Robustness Evaluation Pipeline
ارزیابی دقت مدل‌ها در برابر اغتشاشات تعریف‌شده در augment.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------
# Import from existing project modules
# ----------------------------------------------------------
from common_utils import (
    evaluate,
    evaluate_ensemble,
    get_device,
    get_test_transform,
    load_model_weights,
    set_seed,
)
from augment import PERTURBATIONS, apply_perturbation, make_stable_seed


# ============================================================
# 1. Dataset با اعمال آنلاین اغتشاش
# ============================================================
class PerturbedImageFolder(Dataset):
    """ImageFolder که می‌تواند اغتشاش را به صورت آنلاین اعمال کند."""

    def __init__(
        self,
        root: str | Path,
        transform=None,
        perturbation_name: str | None = None,
        level: Any = None,
        global_seed: int = 42,
    ):
        self.root = Path(root)
        self.transform = transform
        self.perturbation_name = perturbation_name
        self.level = level
        self.global_seed = global_seed

        if not self.root.exists():
            raise FileNotFoundError(f"Test directory not found: {self.root}")

        self.samples: List[Tuple[Path, int]] = []
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not classes:
            raise RuntimeError(f"No class folders found in {self.root}")

        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.classes = classes

        for cls in classes:
            cls_dir = self.root / cls
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"  [Dataset] Loaded {len(self.samples)} images from {len(classes)} classes "
              f"({self.root.name})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {path}")

        if self.perturbation_name is not None:
            relative_path = path.relative_to(self.root)
            local_seed = make_stable_seed(
                image_relative_path=relative_path,
                perturbation_name=self.perturbation_name,
                level=self.level,
                global_seed=self.global_seed,
            )
            rng = np.random.default_rng(local_seed)
            image_bgr = apply_perturbation(
                image_bgr=image_bgr,
                perturbation_name=self.perturbation_name,
                level=self.level,
                rng=rng,
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        if self.transform is not None:
            image_pil = self.transform(image_pil)

        return image_pil, label


# ============================================================
# 2. بارگذاری دینامیک builder مدل‌ها
# ============================================================
def load_builder_from_file(file_path: Path, function_name: str) -> Callable:
    """لود کردن تابع builder از فایل base-model-*.py بدون import معمولی."""
    print(f"  [Builder] Loading {function_name} from {file_path.name} ...")

    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    module_name = f"dynamic_model_{file_path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {file_path.name}")

    print(f"  [Builder] Successfully loaded {function_name}")
    return getattr(module, function_name)


def get_model_builder(model_name: str, root_dir: Path) -> Callable[[int], nn.Module]:
    """برگرداندن تابع builder مربوط به مدل."""
    BUILDER_MAP = {
        "MobileNetV3-Large": (
            root_dir / "mobilenetv3large" / "base-model-mobilenetv3large.py",
            "build_mobilenet_model",
        ),
        "ResNet50": (
            root_dir / "resnet50" / "base-model-resnet50.py",
            "build_resnet50_model",
        ),
        "ResNet18": (
            root_dir / "resnet18" / "base-model-resnet18.py",
            "build_resnet18_model",
        ),
        "ViT-B16": (
            root_dir / "ViTB16" / "base-model-VITB16.py",
            "build_vit_model",
        ),
        "Swin-Tiny": (
            root_dir / "swintiny" / "base-model-swintiny.py",
            "build_swin_model",
        ),
        "ConvNeXt-Tiny": (
            root_dir / "convnexttiny" / "base-model-convnexttiny.py",
            "build_convnext_tiny_model",
        ),
        "EfficientNetB0": (
            root_dir / "efficientnetb0" / "base-model-efficientnet.py",
            "build_efficientnet_b0_model",
        ),
        "DenseNet121": (
            root_dir / "densenet121" / "base-model-densenet.py",
            "build_densenet_model",
        ),
    }

    if model_name not in BUILDER_MAP:
        raise ValueError(
            f"Model '{model_name}' is not registered.\n"
            f"Available: {list(BUILDER_MAP.keys())}"
        )

    file_path, func_name = BUILDER_MAP[model_name]
    return load_builder_from_file(file_path, func_name)


# ============================================================
# 3. خواندن مدل‌ها از YAML
# ============================================================
def load_models_from_config(config_path: Path, root_dir: Path) -> tuple[list[dict], dict]:
    print(f"\n[Config] Loading configuration from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("No models defined in the YAML config.")

    print(f"[Config] Found {len(models_cfg)} model(s) in config.")

    models_to_eval = []

    for m in models_cfg:
        name = m.get("name")
        pattern = m.get("weight_pattern")

        if not name or not pattern:
            print(f"  ⚠️  Skipping invalid entry: {m}")
            continue

        print(f"\n[Config] Processing model: {name}")
        print(f"  weight_pattern = {pattern}")

        weight_paths = sorted(root_dir.glob(pattern))
        if not weight_paths:
            print(f"  ⚠️  No weight files found for '{name}' → skipped")
            continue

        print(f"  ✓ Found {len(weight_paths)} weight file(s):")
        for p in weight_paths:
            print(f"      - {p.relative_to(root_dir)}")

        try:
            builder_fn = get_model_builder(name, root_dir)
        except Exception as e:
            print(f"  ⚠️  Failed to load builder for '{name}': {e}")
            continue

        models_to_eval.append({
            "name": name,
            "builder": builder_fn,
            "weight_paths": weight_paths,
        })

    return models_to_eval, cfg


# ============================================================
# 4. ارزیابی Robustness
# ============================================================
def format_level(level: Any) -> str:
    if isinstance(level, dict):
        return f"k{level['kernel_size']}_a{int(level['angle'])}"
    if isinstance(level, float):
        return f"{level:.4f}".rstrip("0").rstrip(".")
    return str(level)


def evaluate_robustness(
    model_builder_fn,
    weight_paths: List[Path],
    clean_test_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    device: torch.device | None = None,
    global_seed: int = 42,
    num_workers: int = 2,
) -> pd.DataFrame:
    if device is None:
        device = get_device()

    print(f"\n[Eval] Device          : {device}")
    print(f"[Eval] Image size      : {img_size}")
    print(f"[Eval] Batch size      : {batch_size}")
    print(f"[Eval] Num weight files: {len(weight_paths)}")
    print(f"[Eval] Clean test dir  : {clean_test_dir}")

    transform = get_test_transform(img_size)
    criterion = nn.CrossEntropyLoss()
    results = []

    # ---------- Clean baseline ----------
    print("\n" + "-" * 50)
    print("[Eval] >>> Evaluating CLEAN test set")
    clean_ds = PerturbedImageFolder(
        root=clean_test_dir,
        transform=transform,
        perturbation_name=None,
        global_seed=global_seed,
    )
    clean_loader = DataLoader(
        clean_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if len(weight_paths) > 1:
        print(f"[Eval] Using Ensemble of {len(weight_paths)} folds")
        loss, acc, labels, preds = evaluate_ensemble(
            model_builder_fn, weight_paths, clean_loader, criterion, device
        )
    else:
        print("[Eval] Using single model")
        model = model_builder_fn(num_classes=len(clean_ds.classes)).to(device)
        load_model_weights(model, weight_paths[0], device)
        model.eval()
        loss, acc, labels, preds = evaluate(model, clean_loader, criterion, device)

    f1 = f1_score(labels, preds, average="macro")
    results.append({
        "perturbation": "clean",
        "level": "none",
        "severity_idx": 0,
        "accuracy": acc,
        "f1_macro": f1,
        "loss": loss,
        "num_samples": len(labels),
    })
    print(f"[Eval] CLEAN → Acc={acc:.4f} | F1={f1:.4f} | Loss={loss:.4f}")

    # ---------- All perturbations ----------
    total_pert = sum(len(levels) for levels in PERTURBATIONS.values())
    print(f"\n[Eval] Starting {total_pert} perturbation evaluations...")

    for pert_name, levels in PERTURBATIONS.items():
        for severity_idx, level in enumerate(levels, start=1):
            level_str = format_level(level)

            print("\n" + "-" * 50)
            print(f"[Eval] >>> {pert_name} | level={level_str} (severity {severity_idx}/{len(levels)})")

            pert_ds = PerturbedImageFolder(
                root=clean_test_dir,
                transform=transform,
                perturbation_name=pert_name,
                level=level,
                global_seed=global_seed,
            )
            pert_loader = DataLoader(
                pert_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

            if len(weight_paths) > 1:
                loss, acc, labels, preds = evaluate_ensemble(
                    model_builder_fn, weight_paths, pert_loader, criterion, device
                )
            else:
                model = model_builder_fn(num_classes=len(pert_ds.classes)).to(device)
                load_model_weights(model, weight_paths[0], device)
                model.eval()
                loss, acc, labels, preds = evaluate(model, pert_loader, criterion, device)

            f1 = f1_score(labels, preds, average="macro")
            results.append({
                "perturbation": pert_name,
                "level": level_str,
                "severity_idx": severity_idx,
                "accuracy": acc,
                "f1_macro": f1,
                "loss": loss,
                "num_samples": len(labels),
            })
            print(f"[Eval] {pert_name} ({level_str}) → Acc={acc:.4f} | F1={f1:.4f} | Loss={loss:.4f}")

    df = pd.DataFrame(results)
    print(f"\n[Eval] Finished all evaluations. Total rows: {len(df)}")
    return df


# ============================================================
# 5. محاسبات و رسم نمودار
# ============================================================
def compute_relative_robustness(df: pd.DataFrame) -> pd.DataFrame:
    clean_acc = df.loc[df["perturbation"] == "clean", "accuracy"].values[0]
    df = df.copy()
    df["acc_drop"] = clean_acc - df["accuracy"]
    df["relative_robustness"] = df["accuracy"] / clean_acc
    return df


def plot_robustness_results(df: pd.DataFrame, model_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df_pert = df[df["perturbation"] != "clean"].copy()

    if df_pert.empty:
        print(f"  [Plot] No perturbation results to plot for {model_name}")
        return

    # Bar plot
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_pert, x="perturbation", y="accuracy", hue="level")
    clean_acc = df.loc[df["perturbation"] == "clean", "accuracy"].values[0]
    plt.axhline(y=clean_acc, color="red", linestyle="--", label=f"Clean Acc ({clean_acc:.3f})")
    plt.title(f"Robustness – {model_name}")
    plt.xticks(rotation=30, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    bar_path = output_dir / f"{model_name}_robustness_acc.png"
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved bar chart → {bar_path}")

    # Heatmap
    try:
        pivot = df_pert.pivot_table(index="perturbation", columns="level", values="accuracy")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn")
        plt.title(f"Accuracy Heatmap – {model_name}")
        plt.tight_layout()
        heat_path = output_dir / f"{model_name}_robustness_heatmap.png"
        plt.savefig(heat_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] Saved heatmap  → {heat_path}")
    except Exception as e:
        print(f"  [Plot] Heatmap skipped: {e}")


# ============================================================
# 6. main
# ============================================================
def main():
    print("=" * 70)
    print("🚀  ROBUSTNESS EVALUATION PIPELINE STARTED")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Robustness Evaluation")
    parser.add_argument("--config", type=Path, default=Path("robustness_config.yaml"),
                        help="Path to robustness YAML config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        root_dir = Path(__file__).resolve().parent
        print(f"[Main] Project root : {root_dir}")

        config_path = args.config if args.config.is_absolute() else root_dir / args.config
        print(f"[Main] Config path  : {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load models from YAML
        models_to_eval, cfg = load_models_from_config(config_path, root_dir)

        if not models_to_eval:
            raise RuntimeError(
                "No valid models found.\n"
                "→ Check 'weight_pattern' in YAML and make sure .pth files exist."
            )

        print(f"\n[Main] Successfully prepared {len(models_to_eval)} model(s) for evaluation.")

        # Output directory
        output_dir = args.output_dir or Path(cfg.get("output_dir", "robustness_results"))
        if not output_dir.is_absolute():
            output_dir = root_dir / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Main] Output dir   : {output_dir}")

        # Seed & other settings
        global_seed = args.seed if args.seed is not None else cfg.get("global_seed", 42)
        set_seed(global_seed)
        print(f"[Main] Global seed  : {global_seed}")

        clean_test_dir = Path(cfg.get("clean_test_dir", "datasets/SalmonScan_Split/test"))
        if not clean_test_dir.is_absolute():
            clean_test_dir = root_dir / clean_test_dir
        print(f"[Main] Clean test   : {clean_test_dir}")

        if not clean_test_dir.exists():
            raise FileNotFoundError(f"Clean test directory does not exist: {clean_test_dir}")

        img_size = cfg.get("img_size", 224)
        batch_size = cfg.get("batch_size", 32)
        num_workers = cfg.get("num_workers", 2)
        device = get_device()

        print(f"[Main] Device       : {device}")
        print(f"[Main] img_size={img_size} | batch_size={batch_size} | num_workers={num_workers}")
        print("=" * 70)

        all_results = []

        for i, m in enumerate(models_to_eval, start=1):
            print(f"\n\n{'#' * 70}")
            print(f"#  MODEL {i}/{len(models_to_eval)}: {m['name']}")
            print(f"{'#' * 70}")

            try:
                df = evaluate_robustness(
                    model_builder_fn=m["builder"],
                    weight_paths=m["weight_paths"],
                    clean_test_dir=clean_test_dir,
                    img_size=img_size,
                    batch_size=batch_size,
                    device=device,
                    global_seed=global_seed,
                    num_workers=num_workers,
                )
                df["model"] = m["name"]
                df = compute_relative_robustness(df)
                all_results.append(df)

                model_out = output_dir / m["name"].replace(" ", "_")
                model_out.mkdir(parents=True, exist_ok=True)

                csv_path = model_out / "robustness_metrics.csv"
                df.to_csv(csv_path, index=False)
                print(f"[Main] Saved metrics → {csv_path}")

                plot_robustness_results(df, m["name"], model_out)

            except Exception as e:
                print(f"\n❌ ERROR while evaluating {m['name']}:")
                traceback.print_exc()
                print(f"Skipping {m['name']} and continuing...\n")
                continue

        if not all_results:
            raise RuntimeError("No model was successfully evaluated.")

        final_df = pd.concat(all_results, ignore_index=True)
        final_csv = output_dir / "all_models_robustness.csv"
        final_xlsx = output_dir / "all_models_robustness.xlsx"
        final_df.to_csv(final_csv, index=False)
        final_df.to_excel(final_xlsx, index=False)

        print("\n" + "=" * 70)
        print("✅  ROBUSTNESS EVALUATION FINISHED SUCCESSFULLY")
        print(f"📁  Results saved to: {output_dir.resolve()}")
        print(f"   - {final_csv.name}")
        print(f"   - {final_xlsx.name}")
        print("=" * 70)

    except Exception as e:
        print("\n" + "!" * 70)
        print("❌  FATAL ERROR – Pipeline stopped")
        print("!" * 70)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()