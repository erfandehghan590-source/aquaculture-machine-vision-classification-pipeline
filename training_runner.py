from pathlib import Path
from collections import defaultdict
import copy, glob, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from common_utils import (
    set_seed, get_train_transform, get_test_transform, SampleListDataset,
    _extract_parent_id, train_one_epoch, evaluate, evaluate_ensemble,
    collect_hyperparameters, log_training_run, log_test_metrics,
    get_ram_usage_mb, load_model_weights, save_gradcams_for_predicted_infected, plot_learning_curves,
)

from proto_specified_utils import(
    train_one_epoch_proto,
    compute_dataset_prototypes,
    evaluate_proto,
    save_gradcams_for_predicted_infected_proto,
    )


def run_single_seed(
    seed: int,
    config: dict,
    model_builder_fn,
    model_name: str,
    model_tag: str,
    script_dir: Path,
    device: torch.device,
    is_first_seed: bool = True,
):
    """TRAIN: Parent-Level Stratified K-Fold. EVAL: Ensemble + optional Grad-CAM."""

    print(
        f"\n{'=' * 60}\n  STARTING RUN | Seed {seed} | {model_name}\n{'=' * 60}\n",
        flush=True,
    )
    set_seed(seed)

    # Configuration
    mode = config["MODE"]
    data_root = script_dir.parent / config["DATA_ROOT_SPLITED"]
    data_subset = config["DATA_SUBSET"]
    img_size = config["IMG_SIZE"]
    batch_size = config["BATCH_SIZE"]
    num_epochs = config["NUM_EPOCHS"]
    learning_rate = float(config["LEARNING_RATE"])
    weight_decay = float(config["WEIGHT_DECAY"])
    test_ratio = config["TEST_RATIO"]
    num_classes = config["NUM_CLASSES"]
    infected_class_name = config["INFECTED_CLASS_NAME"]
    xai_enable = config.get("XAI_ENABLE", False)
    n_splits = config.get("N_SPLITS", 5)
    pin_memory = device.type == "cuda"

    # Directories
    logs_dir = script_dir / "logs"
    cm_dir = script_dir / "confusion_matrices"
    lc_dir = script_dir / "learning_curves"
    gradcam_dir = script_dir / "gradcams"

    for d in (logs_dir, cm_dir, lc_dir, gradcam_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_log_path = logs_dir / f"{model_tag}_{data_subset.lower()}_all.jsonl"
    gradcam_output_dir = gradcam_dir / f"{model_tag}_seed{seed}_{data_subset.lower()}_infected"

    # Datasets
    train_dir, test_dir = data_root / "train", data_root / "test"
    train_tf, eval_tf = get_train_transform(img_size), get_test_transform(img_size)

    train_base = datasets.ImageFolder(str(train_dir), transform=None)
    test_base = datasets.ImageFolder(str(test_dir), transform=eval_tf)
    test_dataset = Subset(test_base, range(len(test_base)))

    class_names = train_base.classes
    if infected_class_name not in class_names:
        raise ValueError(f"Class '{infected_class_name}' not found in {class_names}")

    infected_class_idx = class_names.index(infected_class_name)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=2,
    )

    # ============================================================
    # TRAIN
    # ============================================================
    if mode == "train":
        print(f"🔄 Performing {n_splits}-Fold Parent-Level Stratified Cross-Validation...")

        parent_to_samples, parent_to_label = defaultdict(list), {}

        for path, label in train_base.samples:
            pid = _extract_parent_id(Path(path).name)
            parent_to_samples[pid].append((path, label))
            parent_to_label[pid] = label

        unique_parents = sorted(parent_to_samples)
        parent_labels = [parent_to_label[pid] for pid in unique_parents]

        print(f"Total unique parents in Train: {len(unique_parents)}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(unique_parents, parent_labels), 1
        ):
            print(
                f"\n{'-' * 50}\n"
                f"  FOLD {fold_idx}/{n_splits}\n"
                f"{'-' * 50}",
                flush=True,
            )

            train_parents = {unique_parents[i] for i in train_idx}
            val_parents = {unique_parents[i] for i in val_idx}

            train_samples = [
                sample
                for pid in train_parents
                for sample in parent_to_samples[pid]
            ]

            val_samples = [
                (path, label)
                for pid in val_parents
                for path, label in parent_to_samples[pid]
                if "__" not in Path(path).stem
            ]

            print(
                f"  Train images: {len(train_samples)} | "
                f"Val images (raw only): {len(val_samples)}"
            )

            train_dataset = SampleListDataset(train_samples, transform=train_tf)
            val_dataset = SampleListDataset(val_samples, transform=eval_tf)

            g = torch.Generator().manual_seed(seed + fold_idx)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                generator=g,
                num_workers=2,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=2,
            )

            fold_test_loader = DataLoader(
                test_base,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=2,
            )

            model = model_builder_fn(num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_loss = float("inf")

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            epoch_times, epoch_ram_mb, epoch_gpu_peak_mb = [], [], []

            for epoch in range(num_epochs):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()

                epoch_start = time.perf_counter()

                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )

                val_loss, val_acc, _, _ = evaluate(
                    model, val_loader, criterion, device
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    gpu_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                else:
                    gpu_peak_mb = 0.0

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                epoch_times.append(time.perf_counter() - epoch_start)
                epoch_ram_mb.append(get_ram_usage_mb())
                epoch_gpu_peak_mb.append(gpu_peak_mb)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            lc_path = (lc_dir/ f"{model_tag}_seed{seed}_fold{fold_idx}_lc_{data_subset.lower()}.png")
            plot_learning_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accs=train_accs,
                val_accs=val_accs,
                MODEL_NAME=f"{model_name} - Seed {seed} Fold {fold_idx}",
                LC_PATH=lc_path,
            )

            model.load_state_dict(best_model_wts)

            model_save_path_fold = (script_dir/ f"{model_tag}_seed{seed}_fold{fold_idx}_{data_subset.lower()}.pth")
            torch.save(model.state_dict(), model_save_path_fold)

            test_loss, test_acc, y_true, y_pred = evaluate(
                model, fold_test_loader, criterion, device
            )

            hyperparams = collect_hyperparameters(
                seed=seed,
                img_size=img_size,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                val_ratio=len(val_parents) / len(unique_parents),
                test_ratio=test_ratio,
                model=model_name,
                model_tag=model_tag,
                data_subset=data_subset,
                device=str(device),
                fold=fold_idx,
                n_splits=n_splits,
                train_size=len(train_samples),
                val_size=len(val_samples),
                weight_decay=weight_decay,
                optimizer="AdamW",
                pretrained="ImageNet",
                mode="kfold",
            )

            log_training_run(
                hyperparameters=hyperparams,
                train_accs=train_accs,
                val_accs=val_accs,
                train_losses=train_losses,
                val_losses=val_losses,
                log_path=run_log_path,
                epoch_times=epoch_times,
                epoch_ram_mb=epoch_ram_mb,
                epoch_gpu_peak_mb=epoch_gpu_peak_mb,
            )

            log_test_metrics(
                log_path=run_log_path,
                test_loss=test_loss,
                test_acc=test_acc,
                model_name=model_name,
                model_tag=model_tag,
                data_subset=data_subset,
                test_size=len(test_base),
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                extra_info={
                    "seed": seed,
                    "fold": fold_idx,
                    "n_splits": n_splits,
                    "mode": "kfold",
                },
            )

            cm_path_fold = (cm_dir/ f"{model_tag}_seed{seed}_fold{fold_idx}_cm_{data_subset.lower()}.png")

            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names,
            ).plot(cmap="Blues", values_format="d")

            plt.title(f"CM (Seed {seed} Fold {fold_idx}) - {model_name}")
            plt.tight_layout()
            plt.savefig(cm_path_fold, dpi=300, bbox_inches="tight")
            plt.close()

            fold_results.append({
                "fold": fold_idx,
                "test_acc": test_acc,
                "test_loss": test_loss,
            })

            print(
                f"  Fold {fold_idx} → Test Acc: {test_acc * 100:.2f}% "
                f"| Test Loss: {test_loss:.4f}"
            )

        accs = [r["test_acc"] for r in fold_results]
        losses = [r["test_loss"] for r in fold_results]

        print(
            f"\n{'=' * 60}\n"
            f"  K-FOLD SUMMARY | {model_name} | Seed {seed} | {n_splits}-Fold\n"
            f"{'=' * 60}"
        )

        for r in fold_results:
            print(
                f"  Fold {r['fold']}: "
                f"Acc = {r['test_acc'] * 100:.2f}% | "
                f"Loss = {r['test_loss']:.4f}"
            )

        mean_acc, mean_loss = np.mean(accs), np.mean(losses)

        print(
            f"{'-' * 60}\n"
            f"  Mean Acc  : {mean_acc * 100:.2f}% ± {np.std(accs) * 100:.2f}%\n"
            f"  Mean Loss : {mean_loss:.4f} ± {np.std(losses):.4f}\n"
            f"{'=' * 60}\n"
        )

        return mean_acc, mean_loss

    # ============================================================
    # EVAL
    # ============================================================
    if mode == "eval":
        print(f"🔍 [EVAL MODE] Searching weights for Ensemble (seed={seed}) ...")

        pattern = (
            script_dir
            / f"{model_tag}_seed{seed}_fold*_{data_subset.lower()}.pth"
        )
        fold_paths = sorted(Path(p) for p in glob.glob(str(pattern)))

        if not fold_paths:
            fallback = (
                script_dir
                / f"{model_tag}_seed{seed}_{data_subset.lower()}.pth"
            )

            if not fallback.exists():
                raise FileNotFoundError(
                    f"No weights found for seed {seed}. "
                    f"Looked for:\n - {pattern}\n - {fallback}"
                )

            print(
                f"[WARNING] No folds found. "
                f"Falling back to single model: {fallback.name}"
            )
            fold_paths = [fallback]
        else:
            print(f"[OK] Found {len(fold_paths)} folds for Ensemble Evaluation:")
            for p in fold_paths:
                print(f"   - {p.name}")

        test_loss, test_acc, y_true, y_pred = evaluate_ensemble(
            model_builder_fn=model_builder_fn,
            weight_paths=fold_paths,
            loader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            num_classes=num_classes,
        )

        log_test_metrics(
            log_path=run_log_path,
            test_loss=test_loss,
            test_acc=test_acc,
            model_name=f"{model_name} (Ensemble)",
            model_tag=model_tag,
            data_subset=data_subset,
            test_size=len(test_dataset),
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            extra_info={
                "seed": seed,
                "mode": mode,
                "ensemble_folds_count": len(fold_paths),
                "ensemble_files": [p.name for p in fold_paths],
            },
        )

        cm_path_ensemble = (
            cm_dir
            / f"{model_tag}_seed{seed}_cm_{data_subset.lower()}_ensemble.png"
        )

        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        ).plot(cmap="Blues", values_format="d")

        plt.title(
            f"Ensemble CM (Seed {seed}) - {data_subset} - {model_name}"
        )
        plt.tight_layout()
        plt.savefig(cm_path_ensemble, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] Ensemble Confusion matrix saved to: {cm_path_ensemble}")

        if xai_enable and is_first_seed:
            print("\nGenerating Grad-CAMs (using fold 1 model)...")

            model_for_cam = model_builder_fn(num_classes).to(device)

            load_model_weights(
                model=model_for_cam,
                model_save_path=fold_paths[0],
                device=device,
            )

            save_gradcams_for_predicted_infected(
                mode=xai_enable,
                model=model_for_cam,
                test_dataset=test_dataset,
                full_dataset=test_base,
                class_names=class_names,
                infected_class_idx=infected_class_idx,
                output_dir=gradcam_output_dir / f"seed{seed}_ensemble_cam",
                device=device,
                architecture=model_tag,
                img_size=img_size,
                clear_previous=True,
            )

        return test_acc, test_loss

    raise ValueError(f"Unknown MODE: {mode}. Use 'train' or 'eval'.")

def run_single_seed_proto(
    seed: int,
    config: dict,
    model_builder_fn,
    model_name: str,
    model_tag: str,
    script_dir: Path,
    device: torch.device,
    target_layers_fn,
    reshape_transform=None,
    is_first_seed: bool = True,
):
    """TRAIN/EVAL pipeline for ProtoNet with Parent-Level Stratified K-Fold."""

    print(
        f"\n{'=' * 60}\n"
        f"  STARTING PROTONET RUN | Seed {seed} | {model_name}\n"
        f"{'=' * 60}\n",
        flush=True,
    )

    set_seed(seed)

    mode = config["MODE"]
    data_root = script_dir.parent / config["DATA_ROOT_SPLITED"]
    data_subset = config["DATA_SUBSET"]
    subset_name = data_subset.lower()

    img_size = config["IMG_SIZE"]
    batch_size = config["BATCH_SIZE"]
    num_epochs = config["NUM_EPOCHS"]
    learning_rate = float(config["LEARNING_RATE"])
    weight_decay = float(config["WEIGHT_DECAY"])
    n_splits = config.get("N_SPLITS", 5)
    test_ratio = config["TEST_RATIO"]

    num_classes = config["NUM_CLASSES"]
    infected_class_name = config["INFECTED_CLASS_NAME"]
    embedding_dim = config["EMBEDDING_DIM"]
    xai_enable = config.get("XAI_ENABLE", False)
    pin_memory = device.type == "cuda"

    logs_dir = script_dir / "logs"
    cm_dir = script_dir / "confusion_matrices"
    lc_dir = script_dir / "learning_curves"
    gradcam_dir = script_dir / "gradcams"

    for d in (logs_dir, cm_dir, lc_dir, gradcam_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_log_path = logs_dir / f"{model_tag}_{subset_name}_all.jsonl"

    # ============================================================
    # DATA
    # ============================================================
    train_tf = get_train_transform(img_size)
    eval_tf = get_test_transform(img_size)

    train_base_dataset = datasets.ImageFolder(
        str(data_root / "train"),
        transform=None,
    )

    test_base_dataset = datasets.ImageFolder(
        str(data_root / "test"),
        transform=eval_tf,
    )

    test_dataset = Subset(
        test_base_dataset,
        range(len(test_base_dataset)),
    )

    class_names = train_base_dataset.classes
    infected_class_idx = class_names.index(infected_class_name)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=2,
    )

    # Parent grouping
    parent_to_samples = defaultdict(list)
    parent_to_label = {}

    for path, label in train_base_dataset.samples:
        parent_id = _extract_parent_id(Path(path).name)
        parent_to_samples[parent_id].append((path, label))
        parent_to_label[parent_id] = label

    unique_parents = list(parent_to_samples)
    parent_labels = [parent_to_label[p] for p in unique_parents]

    # ============================================================
    # TRAIN
    # ============================================================
    if mode == "train":

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(unique_parents, parent_labels),
            start=1,
        ):
            print(
                f"\n{'=' * 60}\n"
                f"  FOLD {fold_idx}/{n_splits} | Seed {seed}\n"
                f"{'=' * 60}",
                flush=True,
            )

            train_parents = {
                unique_parents[i] for i in train_idx
            }
            val_parents = {
                unique_parents[i] for i in val_idx
            }

            train_samples = [
                sample
                for p in train_parents
                for sample in parent_to_samples[p]
            ]

            val_samples = [
                (path, label)
                for p in val_parents
                for path, label in parent_to_samples[p]
                if "__" not in Path(path).stem
            ]

            train_dataset = SampleListDataset(
                train_samples,
                transform=train_tf,
            )
            train_eval_dataset = SampleListDataset(
                train_samples,
                transform=eval_tf,
            )
            val_dataset = SampleListDataset(
                val_samples,
                transform=eval_tf,
            )

            generator = torch.Generator().manual_seed(
                seed + fold_idx
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                generator=generator,
                num_workers=2,
            )

            train_eval_loader = DataLoader(
                train_eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=2,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=2,
            )

            model = model_builder_fn(
                embedding_dim=embedding_dim
            ).to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_loss = float("inf")
            best_prototypes = None

            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            epoch_times, epoch_ram_mb, epoch_gpu_peak_mb = [], [], []

            for epoch in range(num_epochs):

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()

                start = time.perf_counter()

                train_loss, train_acc = train_one_epoch_proto(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    num_classes=num_classes,
                )

                current_prototypes = compute_dataset_prototypes(
                    model=model,
                    loader=train_eval_loader,
                    device=device,
                    num_classes=num_classes,
                )

                val_loss, val_acc, _, _ = evaluate_proto(
                    model=model,
                    loader=val_loader,
                    prototypes=current_prototypes,
                    device=device,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    gpu_peak_mb = (
                        torch.cuda.max_memory_allocated(device)
                        / (1024 ** 2)
                    )
                else:
                    gpu_peak_mb = 0.0

                epoch_times.append(time.perf_counter() - start)
                epoch_ram_mb.append(get_ram_usage_mb())
                epoch_gpu_peak_mb.append(gpu_peak_mb)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                print(
                    f"Fold {fold_idx}/{n_splits} | "
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc * 100:.2f}%",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(
                        model.state_dict()
                    )
                    best_prototypes = current_prototypes.detach().clone()

            model.load_state_dict(best_model_wts)
            prototypes = best_prototypes

            model_path = (
                script_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"{subset_name}_salmonscan.pth"
            )
            proto_path = (
                script_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"{subset_name}_prototypes.pt"
            )
            lc_path = (
                lc_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"lc_{subset_name}.png"
            )
            cm_path = (
                cm_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"cm_{subset_name}.png"
            )

            torch.save(model.state_dict(), model_path)
            torch.save(prototypes, proto_path)

            plot_learning_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accs=train_accs,
                val_accs=val_accs,
                MODEL_NAME=f"{model_name} - Seed {seed} - Fold {fold_idx}",
                LC_PATH=lc_path,
            )

            hyperparams = collect_hyperparameters(
                seed=seed,
                img_size=img_size,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                val_ratio=None,
                test_ratio=test_ratio,
                model=model_name,
                model_tag=model_tag,
                data_subset=data_subset,
                device=str(device),
                train_size=len(train_dataset),
                val_size=len(val_dataset),
                fold=fold_idx,
                n_splits=n_splits,
                train_parent_count=len(train_parents),
                val_parent_count=len(val_parents),
                weight_decay=weight_decay,
                optimizer="AdamW",
                pretrained="ImageNet",
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                mode="protonet",
            )

            log_training_run(
                hyperparameters=hyperparams,
                train_accs=train_accs,
                val_accs=val_accs,
                train_losses=train_losses,
                val_losses=val_losses,
                log_path=run_log_path,
                epoch_times=epoch_times,
                epoch_ram_mb=epoch_ram_mb,
                epoch_gpu_peak_mb=epoch_gpu_peak_mb,
            )

            test_loss, test_acc, y_true, y_pred = evaluate_proto(
                model=model,
                loader=test_loader,
                prototypes=prototypes,
                device=device,
            )

            log_test_metrics(
                log_path=run_log_path,
                test_loss=test_loss,
                test_acc=test_acc,
                model_name=model_name,
                model_tag=model_tag,
                data_subset=data_subset,
                test_size=len(test_dataset),
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                extra_info={
                    "seed": seed,
                    "fold": fold_idx,
                    "n_splits": n_splits,
                    "mode": mode,
                    "architecture": "ProtoNet",
                    "embedding_dim": embedding_dim,
                },
            )

            cm = confusion_matrix(y_true, y_pred)

            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names,
            ).plot(
                cmap="Blues",
                values_format="d",
            )

            plt.title(
                f"CM (Seed {seed} - Fold {fold_idx}) - "
                f"{data_subset} - {model_name}"
            )
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300, bbox_inches="tight")
            plt.close()

            fold_results.append(
                (test_acc, test_loss)
            )

            if xai_enable and is_first_seed and fold_idx == 1:

                gradcam_output_dir = (
                    gradcam_dir
                    / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                      f"{subset_name}_infected"
                )
                gradcam_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                save_gradcams_for_predicted_infected_proto(
                    mode=xai_enable,
                    model=model,
                    prototypes=prototypes,
                    test_dataset=test_dataset,
                    full_dataset=test_base_dataset,
                    class_names=class_names,
                    infected_class_idx=infected_class_idx,
                    output_dir=gradcam_output_dir,
                    device=device,
                    target_layers=target_layers_fn(model),
                    reshape_transform=reshape_transform,
                    img_size=img_size,
                    clear_previous=True,
                )

        mean_acc = np.mean([x[0] for x in fold_results])
        mean_loss = np.mean([x[1] for x in fold_results])

        print(
            f"\nK-FOLD RESULT | {model_name} | Seed {seed}\n"
            f"Accuracy: {mean_acc * 100:.2f}% | "
            f"Loss: {mean_loss:.4f}",
            flush=True,
        )

        return mean_acc, mean_loss

    # ============================================================
    # EVAL
    # ============================================================
    elif mode == "eval":

        fold_results = []

        for fold_idx in range(1, n_splits + 1):

            model_path = (
                script_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"{subset_name}_salmonscan.pth"
            )
            proto_path = (
                script_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"{subset_name}_prototypes.pt"
            )

            model = model_builder_fn(
                embedding_dim=embedding_dim
            ).to(device)

            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=device,
                    weights_only=True,
                )
            )

            prototypes = torch.load(
                proto_path,
                map_location=device,
                weights_only=True,
            )

            test_loss, test_acc, y_true, y_pred = evaluate_proto(
                model=model,
                loader=test_loader,
                prototypes=prototypes,
                device=device,
            )

            log_test_metrics(
                log_path=run_log_path,
                test_loss=test_loss,
                test_acc=test_acc,
                model_name=model_name,
                model_tag=model_tag,
                data_subset=data_subset,
                test_size=len(test_dataset),
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                extra_info={
                    "seed": seed,
                    "fold": fold_idx,
                    "n_splits": n_splits,
                    "mode": mode,
                    "architecture": "ProtoNet",
                    "embedding_dim": embedding_dim,
                },
            )

            cm = confusion_matrix(y_true, y_pred)

            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names,
            ).plot(
                cmap="Blues",
                values_format="d",
            )

            plt.title(
                f"CM (Seed {seed} - Fold {fold_idx}) - "
                f"{data_subset} - {model_name}"
            )
            plt.tight_layout()
            plt.savefig(
                cm_dir
                / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                  f"cm_{subset_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            fold_results.append(
                (test_acc, test_loss)
            )

            if xai_enable and is_first_seed and fold_idx == 1:

                gradcam_output_dir = (
                    gradcam_dir
                    / f"{model_tag}_seed{seed}_fold{fold_idx}_"
                      f"{subset_name}_infected"
                )
                gradcam_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                save_gradcams_for_predicted_infected_proto(
                    mode=xai_enable,
                    model=model,
                    prototypes=prototypes,
                    test_dataset=test_dataset,
                    full_dataset=test_base_dataset,
                    class_names=class_names,
                    infected_class_idx=infected_class_idx,
                    output_dir=gradcam_output_dir,
                    device=device,
                    target_layers=target_layers_fn(model),
                    reshape_transform=reshape_transform,
                    img_size=img_size,
                    clear_previous=True,
                )

        mean_acc = np.mean([x[0] for x in fold_results])
        mean_loss = np.mean([x[1] for x in fold_results])

        print(
            f"\nK-FOLD EVAL | {model_name} | Seed {seed}\n"
            f"Accuracy: {mean_acc * 100:.2f}% | "
            f"Loss: {mean_loss:.4f}",
            flush=True,
        )

        return mean_acc, mean_loss

    raise ValueError(
        f"Unknown MODE: {mode}. Use 'train' or 'eval'."
    )