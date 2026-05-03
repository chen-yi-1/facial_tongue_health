import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageEnhance


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def split(paths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = int(len(paths) * val_ratio)
    return paths[n_val:], paths[:n_val]


def copy_many(paths: Iterable[Path], dst_dir: Path, prefix: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in paths:
        ext = p.suffix.lower()
        dst = dst_dir / f"{prefix}_{n:06d}{ext}"
        shutil.copy2(p, dst)
        n += 1
    return n


def take_sample(paths: List[Path], k: int, seed: int) -> List[Path]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    if k >= len(paths):
        return list(paths)
    return rng.sample(paths, k)


def augment_image(src: Path, dst: Path, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.open(src).convert("RGB")

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR, expand=False)

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=92)


def fill_to_target_with_augmentation(src_train: List[Path], dst_dir: Path, target_n: int, prefix: str, seed: int) -> None:
    current = len(list_images(dst_dir))
    if current >= target_n:
        return
    if not src_train:
        raise ValueError("训练集原图为空，无法增强补齐。")
    rng = random.Random(seed)
    i = current
    while i < target_n:
        src = rng.choice(src_train)
        dst = dst_dir / f"{prefix}_aug_{i:06d}.jpg"
        augment_image(src, dst, seed=seed + i)
        i += 1


def main():
    parser = argparse.ArgumentParser(description="将 Kaggle 舌象数据整理为 data/tongue 三分类目录（含增强补齐）")
    parser.add_argument("--kaggle_root", required=True, help="datasets_raw/kaggle 路径")
    parser.add_argument("--out_data_root", required=True, help="项目 data 路径（例如 facial_tongue_health/data）")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--healthy_n", type=int, default=1000)
    parser.add_argument("--subhealthy_n", type=int, default=600)
    parser.add_argument("--unhealthy_n", type=int, default=600)
    args = parser.parse_args()

    kaggle_root = Path(args.kaggle_root).resolve()
    out_root = Path(args.out_data_root).resolve() / "tongue"

    # 数据源组合（都来自 Kaggle 自动下载结果）：
    # - tongue-images：Healthy / Pre Diabetes / Diabetes（有三类，但规模较小）
    # - tongue-coating：Coated_front / Coated_Middle / Coated-back（规模较小，仅作补充）
    # 训练集不足部分用增强补齐到指定规模，验证集保持真实样本以便评估。
    tongue_images_root = kaggle_root / "tongue_images" / "data Dr fiiuzzy"
    healthy_a = list_images(tongue_images_root / "Healthy")
    subhealthy_a = list_images(tongue_images_root / "Pre Diabetes")
    unhealthy_a = list_images(tongue_images_root / "Diabetes")

    coating_root = kaggle_root / "tongue_tongue_coating"
    healthy_b = list_images(coating_root / "Coated_front")
    subhealthy_b = list_images(coating_root / "Coated_Middle")
    unhealthy_b = list_images(coating_root / "Coated-back")

    healthy_pool = healthy_a + healthy_b
    subhealthy_pool = subhealthy_a + subhealthy_b
    unhealthy_pool = unhealthy_a + unhealthy_b
    if not (healthy_pool and subhealthy_pool and unhealthy_pool):
        raise FileNotFoundError("舌象源数据不足，请确认已下载 tongue_images 与 tongue_tongue_coating。")

    healthy_imgs = take_sample(healthy_pool, args.healthy_n, seed=args.seed)
    subhealthy_imgs = take_sample(subhealthy_pool, args.subhealthy_n, seed=args.seed + 1)
    unhealthy_imgs = take_sample(unhealthy_pool, args.unhealthy_n, seed=args.seed + 2)

    healthy_tr, healthy_val = split(healthy_imgs, val_ratio=args.val_ratio, seed=args.seed)
    sub_tr, sub_val = split(subhealthy_imgs, val_ratio=args.val_ratio, seed=args.seed)
    un_tr, un_val = split(unhealthy_imgs, val_ratio=args.val_ratio, seed=args.seed)

    copy_many(healthy_tr, out_root / "train" / "healthy", "healthy")
    copy_many(healthy_val, out_root / "val" / "healthy", "healthy")
    copy_many(sub_tr, out_root / "train" / "subhealthy", "subhealthy")
    copy_many(sub_val, out_root / "val" / "subhealthy", "subhealthy")
    copy_many(un_tr, out_root / "train" / "unhealthy", "unhealthy")
    copy_many(un_val, out_root / "val" / "unhealthy", "unhealthy")

    # 补齐训练集
    fill_to_target_with_augmentation(
        src_train=healthy_tr,
        dst_dir=out_root / "train" / "healthy",
        target_n=int(args.healthy_n * (1 - args.val_ratio)),
        prefix="healthy",
        seed=args.seed + 10_000,
    )
    fill_to_target_with_augmentation(
        src_train=sub_tr,
        dst_dir=out_root / "train" / "subhealthy",
        target_n=int(args.subhealthy_n * (1 - args.val_ratio)),
        prefix="subhealthy",
        seed=args.seed + 20_000,
    )
    fill_to_target_with_augmentation(
        src_train=un_tr,
        dst_dir=out_root / "train" / "unhealthy",
        target_n=int(args.unhealthy_n * (1 - args.val_ratio)),
        prefix="unhealthy",
        seed=args.seed + 30_000,
    )

    print("OK. 输出：", out_root)
    print("Counts:")
    print("  healthy:", len(healthy_imgs), "subhealthy:", len(subhealthy_imgs), "unhealthy:", len(unhealthy_imgs))


if __name__ == "__main__":
    main()

