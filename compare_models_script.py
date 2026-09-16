from pathlib import Path
from IPython.display import display
import yaml
from model_comparison_utils import (
    compare_models,
    plot_model_comparison,
    get_model_weights_table,
    auto_discover_seed_logs,
)

# ============================================================
# 1. تنظیمات مسیرها و بارگذاری کانفیگ
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = ROOT_DIR / "model_comparison_config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH.resolve()}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# مدیریت مسیر خروجی (پشتیبانی از مسیر نسبی یا مطلق)
output_cfg = config.get("output_dir", "model_comparison_results")
OUTPUT_DIR = Path(output_cfg) if Path(output_cfg).is_absolute() else (ROOT_DIR / output_cfg)

GT_FOLDER = config.get("gt_folder")
THRESHOLD = float(config.get("threshold", 0.65))
SAVE_EXCEL = bool(config.get("save_excel", True))
ENABLE_XAI = bool(config.get("enable_xai", False))
MODELS = config.get("models", [])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🚀 STARTING BENCHMARK & MULTI-SEED EVALUATION PIPELINE")
print("=" * 80)
print(f"Root Directory   : {ROOT_DIR.resolve()}")
print(f"Output Directory : {OUTPUT_DIR.resolve()}")
print(f"XAI Evaluation   : {'ENABLED' if ENABLE_XAI else 'DISABLED'}")
print(f"Models Count     : {len(MODELS)}")
print("=" * 80)

# ============================================================
# 2. اعتبارسنجی هوشمند فایل‌ها و سیدها (Auto-Discovery Validation)
# ============================================================
VALID_MODELS = []
for m in MODELS:
    model_name = m.get("name", "Unknown")
    target_pattern = m.get("log_paths") or m.get("log_path") or m.get("log_pattern")
    
    # کشف خودکار لاگ‌های موجود
    found_logs = auto_discover_seed_logs(target_pattern, root_dir=ROOT_DIR)
    
    if not found_logs:
        print(f"⚠️ Warning: No training log (.jsonl) found for '{model_name}' with pattern: {target_pattern}")
        continue

    if ENABLE_XAI:
        heatmap_dir_raw = m.get("heatmap_folder", "")
        heatmap_dir = Path(heatmap_dir_raw) if Path(heatmap_dir_raw).is_absolute() else (ROOT_DIR / heatmap_dir_raw)
        if not heatmap_dir.exists():
            print(f"⚠️ Warning: Heatmap folder missing for '{model_name}': {heatmap_dir}")
            continue

    print(f"✔ '{model_name}': Found {len(found_logs)} run/seed file(s)")
    VALID_MODELS.append(m)

print(f"\n✅ Valid models ready for benchmark: {len(VALID_MODELS)} / {len(MODELS)}")

if not VALID_MODELS:
    raise RuntimeError("❌ No valid models with training logs were found. Please check your YAML configuration.")

# ============================================================
# 3. اجرای مقایسه چندسیده و تجمیع آماری (Benchmark Execution)
# ============================================================
stats_df, paper_table, raw_runs_df, all_epochs_df = compare_models(
    model_configs=VALID_MODELS,
    root_dir=ROOT_DIR,
    gt_folder=GT_FOLDER if ENABLE_XAI else None,
    threshold=THRESHOLD,
    output_dir=OUTPUT_DIR,
    save_excel=SAVE_EXCEL,
    enable_xai=ENABLE_XAI,
)

# ============================================================
# 4. استخراج متادیتای وزن‌ها و پارامترها (Weights Metadata)
# ============================================================
weights_df = get_model_weights_table(
    models_config=VALID_MODELS,
    output_dir=OUTPUT_DIR,
    save_excel=SAVE_EXCEL,
)

# ============================================================
# 5. رسم تمامی نمودارها (Visualizations & Error-Bars)
# ============================================================
plot_model_comparison(
    stats_df=stats_df,
    output_dir=OUTPUT_DIR,
)

# ============================================================
# 6. نمایش خروجی نهایی جدول ژورنالی (Paper Summary)
# ============================================================
print("\n" + "=" * 90)
print("🏆 FINAL BENCHMARK SUMMARY (MEAN ± STD FOR PAPER / REPORT):")
print("=" * 90)

try:
    # نمایش زیبا در Jupyter Notebook
    display(paper_table)
except Exception:
    # نمایش در کنسول متنی استاندارد
    print(paper_table.to_string(index=False))

print("\n" + "=" * 90)
print(f"🎉 Benchmark completed successfully!")
print(f"📁 Reports, Excel multi-sheet workbook, and charts saved to:\n➡️ {OUTPUT_DIR.resolve()}")
print("=" * 90)
