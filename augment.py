from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

PERTURBATIONS: dict[str, list[Any]] = {
    "gaussian_blur": [1.0, 2.0, 3.0],
    "gaussian_noise": [0.0025, 0.01, 0.025],
    "brightness_reduction": [0.85, 0.70, 0.55],
    "contrast_reduction": [0.85, 0.70, 0.55],
    "haze": [0.15, 0.30, 0.45],
    "motion_blur": [
        {"kernel_size": 7, "angle": 0},
        {"kernel_size": 7, "angle": 45},
        {"kernel_size": 7, "angle": 90},
        {"kernel_size": 15, "angle": 0},
        {"kernel_size": 15, "angle": 45},
        {"kernel_size": 15, "angle": 90},
        {"kernel_size": 23, "angle": 0},
        {"kernel_size": 23, "angle": 45},
        {"kernel_size": 23, "angle": 90},
    ],
    "rotation": [-15, -10, -5, 0, 5, 10, 15],
    "mirror": ["x", "y"],
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not config.get("DATA_ROOT_PRIMARY"):
        raise ValueError(f"`DATA_ROOT_PRIMARY` در فایل تنظیمات معتبر نیست: {config_path}")

    return config


def clear_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def make_stable_seed(
    image_relative_path: Path,
    perturbation_name: str,
    level: Any,
    global_seed: int,
) -> int:
    key = f"{global_seed}|{image_relative_path.as_posix()}|{perturbation_name}|{level}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return str(value).replace("-", "neg").replace(".", "p")


def gaussian_kernel_size(sigma: float) -> int:
    return 2 * math.ceil(3 * sigma) + 1


def apply_gaussian_blur(image_bgr: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = gaussian_kernel_size(sigma)
    return cv2.GaussianBlur(
        image_bgr,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def apply_gaussian_noise(
    image_bgr: np.ndarray,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    image_float = image_bgr.astype(np.float32) / 255.0
    noise = rng.normal(
        loc=0.0,
        scale=math.sqrt(variance),
        size=image_float.shape,
    ).astype(np.float32)

    noisy_image = np.clip(image_float + noise, 0.0, 1.0)
    return (noisy_image * 255.0).round().astype(np.uint8)


def apply_brightness_reduction(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast_reduction(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    image_float = image_bgr.astype(np.float32)
    mean_intensity = image_float.mean(axis=(0, 1), keepdims=True)
    reduced_contrast = mean_intensity + factor * (image_float - mean_intensity)
    return np.clip(reduced_contrast, 0, 255).astype(np.uint8)


def apply_haze(
    image_bgr: np.ndarray,
    intensity: float,
    atmospheric_light_bgr: tuple[int, int, int] = (220, 235, 245),
) -> np.ndarray:
    image_float = image_bgr.astype(np.float32)
    atmospheric_light = np.array(atmospheric_light_bgr, dtype=np.float32).reshape(1, 1, 3)

    hazy_image = (1.0 - intensity) * image_float + intensity * atmospheric_light
    return np.clip(hazy_image, 0, 255).astype(np.uint8)


def make_motion_blur_kernel(kernel_size: int, angle: float) -> np.ndarray:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("`kernel_size` برای Motion Blur باید یک عدد فرد و حداقل 3 باشد.")

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    cv2.line(
        kernel,
        pt1=(0, center),
        pt2=(kernel_size - 1, center),
        color=1.0,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center=(center, center),
        angle=angle,
        scale=1.0,
    )

    kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        dsize=(kernel_size, kernel_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        raise ValueError("ساخت Motion Blur kernel ناموفق بود.")

    return kernel / kernel_sum


def apply_motion_blur(
    image_bgr: np.ndarray,
    kernel_size: int,
    angle: float,
) -> np.ndarray:
    kernel = make_motion_blur_kernel(kernel_size=kernel_size, angle=angle)

    return cv2.filter2D(
        image_bgr,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )


def apply_rotation(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    center = (width / 2.0, height / 2.0)

    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)

    return cv2.warpAffine(
        image_bgr,
        rotation_matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_mirror(image_bgr: np.ndarray, axis: str) -> np.ndarray:
    if axis == "x":
        return cv2.flip(image_bgr, 0)

    if axis == "y":
        return cv2.flip(image_bgr, 1)

    raise ValueError("محور mirror باید `x` یا `y` باشد.")


def apply_perturbation(
    image_bgr: np.ndarray,
    perturbation_name: str,
    level: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    if perturbation_name == "gaussian_blur":
        return apply_gaussian_blur(image_bgr, sigma=float(level))

    if perturbation_name == "gaussian_noise":
        return apply_gaussian_noise(image_bgr, variance=float(level), rng=rng)

    if perturbation_name == "brightness_reduction":
        return apply_brightness_reduction(image_bgr, factor=float(level))

    if perturbation_name == "contrast_reduction":
        return apply_contrast_reduction(image_bgr, factor=float(level))

    if perturbation_name == "haze":
        return apply_haze(image_bgr, intensity=float(level))

    if perturbation_name == "motion_blur":
        if not isinstance(level, dict):
            raise ValueError("تنظیمات Motion Blur باید یک دیکشنری باشند.")

        return apply_motion_blur(
            image_bgr,
            kernel_size=int(level["kernel_size"]),
            angle=float(level["angle"]),
        )

    if perturbation_name == "rotation":
        return apply_rotation(image_bgr, angle=float(level))

    if perturbation_name == "mirror":
        return apply_mirror(image_bgr, axis=str(level))

    raise ValueError(f"اغتشاش پشتیبانی‌نشده: {perturbation_name}")


def make_output_stem(
    source_stem: str,
    perturbation_name: str,
    level: Any,
) -> str:
    if perturbation_name == "gaussian_blur":
        return f"{source_stem}__gaussian_blur__sigma_{format_number(level)}"

    if perturbation_name == "gaussian_noise":
        return f"{source_stem}__gaussian_noise__var_{format_number(level)}"

    if perturbation_name == "brightness_reduction":
        return f"{source_stem}__brightness_reduction__factor_{format_number(level)}"

    if perturbation_name == "contrast_reduction":
        return f"{source_stem}__contrast_reduction__factor_{format_number(level)}"

    if perturbation_name == "haze":
        return f"{source_stem}__haze__intensity_{format_number(level)}"

    if perturbation_name == "motion_blur":
        return (
            f"{source_stem}__motion_blur"
            f"__kernel_{level['kernel_size']}"
            f"__angle_{format_number(level['angle'])}deg"
        )

    if perturbation_name == "rotation":
        return f"{source_stem}__rotation__angle_{format_number(level)}deg"

    if perturbation_name == "mirror":
        return f"{source_stem}__mirror__axis_{level}"

    raise ValueError(f"نام‌گذاری برای اغتشاش پشتیبانی‌نشده: {perturbation_name}")


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    global_seed: int,
    keep_original: bool,
) -> tuple[int, int]:
    created_count = 0
    skipped_count = 0

    image_paths = sorted(
        image_path
        for image_path in raw_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"هیچ تصویر معتبری در مسیر زیر پیدا نشد:\n{raw_dir}")

    for image_path in image_paths:
        relative_path = image_path.relative_to(raw_dir)
        output_class_dir = output_dir / relative_path.parent
        output_class_dir.mkdir(parents=True, exist_ok=True)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Skipped unreadable image: {image_path}")
            skipped_count += 1
            continue

        if keep_original:
            original_output_path = output_class_dir / image_path.name
            if cv2.imwrite(str(original_output_path), image_bgr):
                created_count += 1
            else:
                print(f"Skipped unwritable output: {original_output_path}")
                skipped_count += 1

        for perturbation_name, levels in PERTURBATIONS.items():
            for level in levels:
                local_seed = make_stable_seed(
                    image_relative_path=relative_path,
                    perturbation_name=perturbation_name,
                    level=level,
                    global_seed=global_seed,
                )
                rng = np.random.default_rng(local_seed)

                try:
                    perturbed_image = apply_perturbation(
                        image_bgr=image_bgr,
                        perturbation_name=perturbation_name,
                        level=level,
                        rng=rng,
                    )

                    output_stem = make_output_stem(
                        source_stem=image_path.stem,
                        perturbation_name=perturbation_name,
                        level=level,
                    )
                    output_path = output_class_dir / f"{output_stem}.jpg"

                    if cv2.imwrite(
                        str(output_path),
                        perturbed_image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    ):
                        created_count += 1
                    else:
                        print(f"Skipped unwritable output: {output_path}")
                        skipped_count += 1

                except (ValueError, KeyError, cv2.error) as error:
                    print(f"Skipped {image_path} | {perturbation_name} | {level}: {error}")
                    skipped_count += 1

    return created_count, skipped_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate augmented SalmonScan images from <DATA_ROOT_PRIMARY>/Raw."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("MVconfig.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global deterministic seed.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Also copy each clean image into the output directory.",
    )

    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(f"Ignoring notebook arguments: {unknown_args}")

    return args


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)

    DATA_ROOT_PRIMARY = Path(config["DATA_ROOT_PRIMARY"]).expanduser()
    raw_dir = DATA_ROOT_PRIMARY / "Raw"
    output_dir = DATA_ROOT_PRIMARY / "My augment"

    if not raw_dir.exists():
        raise FileNotFoundError(f"پوشه Raw پیدا نشد:\n{raw_dir}")

    print(f"Input directory:  {raw_dir}")
    print(f"Output directory: {output_dir}")
    print("Clearing previous output files...")

    clear_output_directory(output_dir)

    created_count, skipped_count = process_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        global_seed=args.seed,
        keep_original=args.keep_original,
    )

    augmentations_per_image = sum(len(levels) for levels in PERTURBATIONS.values())

    print("\nCompleted successfully.")
    print(f"Augmentations per source image: {augmentations_per_image}")
    print(f"Created files: {created_count}")
    print(f"Skipped files: {skipped_count}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
