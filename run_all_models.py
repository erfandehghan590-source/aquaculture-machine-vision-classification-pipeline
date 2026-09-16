import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_TO_RUN = [
    "base-model-convnexttiny.py",
    "base-model-densenet.py",
    "base-model-efficientnet.py",
    "base-model-mobilenetv3large.py",
    "base-model-resnet18.py",
    "base-model-resnet50.py",
    "base-model-swintiny.py",
    "base-model-VITB16.py",
    "base-model-YOLOv8.py"
]

script_counter = 0
print(f"SCRIPTS_TO_RUN list contains the following {len(SCRIPTS_TO_RUN)} .py files:")
for script_name in SCRIPTS_TO_RUN:
    script_counter += 1
    print(f" [{script_counter}] - {script_name}")

PROJECT_ROOT = Path(__file__).resolve().parent

def find_script(script_name: str, root_dir: Path):
    """جستجو در تمام زیرپوشه‌ها برای پیدا کردن مسیر فایل"""
    matches = list(root_dir.rglob(script_name))
    return matches[0] if matches else None


def run_models():
    accept = input("Are all models in the list? Type '1' for YES and '0' for NO. ")
    delay_on = input("Do you want to add any delays between running each script? It's recommended when you're training. Type '1' for YES and '0' for NO. ")

    delay_seconds = 0
    if delay_on == "1":
        while True:
            try:
                delay_input = input("Enter the delay time in seconds (e.g. 30, 60, 120): ").strip()
                delay_seconds = float(delay_input)
                if delay_seconds < 0:
                    print("Delay cannot be negative. Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a valid number (e.g. 30).")

    if accept == "1":
        print(f"Searching for scripts inside: {PROJECT_ROOT}\n" + "=" * 60)
        total_scripts = len(SCRIPTS_TO_RUN)
        print(f"Starting execution of {total_scripts} model(s)...\n" + "=" * 50)
        start_total_time = time.time()
        successful = []
        failed = []

        for idx, script in enumerate(SCRIPTS_TO_RUN, 1):
            script_path = find_script(script, PROJECT_ROOT)

            if script_path is None or not script_path.exists():
                print(f"warning: [{idx}/{total_scripts}] File not found: {script}")
                failed.append((script, "File not found"))
                continue

            print(f"\n[{idx}/{total_scripts}] Running: {script} ...")
            start_time = time.time()

            try:
                subprocess.run(
                    [sys.executable, str(script_path)],
                    check=True
                )
                elapsed_time = time.time() - start_time
                print(f"[{idx}/{total_scripts}] Finished {script} successfully in {elapsed_time:.2f}s")
                successful.append(script)

            except subprocess.CalledProcessError as e:
                elapsed_time = time.time() - start_time
                print(f"error: [{idx}/{total_scripts}] Error while running {script} (Exit Code: {e.returncode})")
                failed.append((script, f"Exit code {e.returncode}"))

            except KeyboardInterrupt:
                print("\nProcess interrupted by user (Ctrl+C). Stopping execution.")
                break

            # اعمال واقعی تاخیر (فقط بین اسکریپت‌ها، نه بعد از آخرین)
            if delay_on == "1" and idx < total_scripts and delay_seconds > 0:
                print(f"\nWaiting {delay_seconds} seconds before the next script...")
                time.sleep(delay_seconds)

        total_time = time.time() - start_total_time
        print("\n" + "=" * 50)
        print(f"All tasks completed in {total_time / 60:.2f} minutes.")
        print(f"Successful ({len(successful)}): {', '.join(successful) if successful else 'None'}")
        if failed:
            print(f"Failed ({len(failed)}):")
            for f_name, reason in failed:
                print(f"   - {f_name}: {reason}")
    elif accept == "0":
        print("Run the script again after editing the SCRIPTS_TO_RUN")
        print("App closed.")
    else:
        print("Input not recognized. Please enter 0 or 1 in the input.")
        print("App closed.")

if __name__ == "__main__":
    run_models()
