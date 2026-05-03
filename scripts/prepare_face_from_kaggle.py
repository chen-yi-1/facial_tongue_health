import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


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


def main():
    parser = argparse.ArgumentParser(description="将 Kaggle 下载的面部数据整理为 data/face 三分类目录")
    parser.add_argument("--kaggle_root", required=True, help="datasets_raw/kaggle 路径")
    parser.add_argument("--out_data_root", required=True, help="项目 data 路径（例如 facial_tongue_health/data）")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--healthy_n", type=int, default=1000, help="从 FFHQ 抽取健康样本数量")
    parser.add_argument("--subhealthy_n", type=int, default=600, help="亚健康样本数量（从痤疮数据集中抽取）")
    parser.add_argument("--unhealthy_n", type=int, default=600, help="不健康样本数量（从皮肤病数据集中抽取）")
    args = parser.parse_args()

    kaggle_root = Path(args.kaggle_root).resolve()
    out_root = Path(args.out_data_root).resolve() / "face"

    # 1) Healthy: FFHQ thumbnails 128x128 (70k)
    ffhq_dir = kaggle_root / "face_ffhq" / "thumbnails128x128"
    ffhq_imgs = list_images(ffhq_dir)
    if not ffhq_imgs:
        raise FileNotFoundError(f"未找到 FFHQ 图片：{ffhq_dir}")
    healthy_imgs = take_sample(ffhq_imgs, args.healthy_n, seed=args.seed)

    # 2) Subhealthy: acne-dataset (单文件夹 Acne)
    acne_dir = kaggle_root / "face_acne" / "Acne"
    acne_imgs = list_images(acne_dir)
    if not acne_imgs:
        raise FileNotFoundError(f"未找到 Acne 图片：{acne_dir}")
    subhealthy_imgs = take_sample(acne_imgs, args.subhealthy_n, seed=args.seed + 1)

    # 3) Unhealthy: face-skin-disease 的多个类别合并
    skin_train = kaggle_root / "face_skin_disease" / "DATA" / "train"
    skin_test = kaggle_root / "face_skin_disease" / "DATA" / "testing"
    skin_imgs = list_images(skin_train) + list_images(skin_test)
    if not skin_imgs:
        raise FileNotFoundError(f"未找到 face-skin-disease 图片：{skin_train} / {skin_test}")
    unhealthy_imgs = take_sample(skin_imgs, args.unhealthy_n, seed=args.seed + 2)

    # Split train/val within each class
    healthy_tr, healthy_val = split(healthy_imgs, val_ratio=args.val_ratio, seed=args.seed)
    sub_tr, sub_val = split(subhealthy_imgs, val_ratio=args.val_ratio, seed=args.seed)
    un_tr, un_val = split(unhealthy_imgs, val_ratio=args.val_ratio, seed=args.seed)

    # Copy
    copy_many(healthy_tr, out_root / "train" / "healthy", "ffhq")
    copy_many(healthy_val, out_root / "val" / "healthy", "ffhq")
    copy_many(sub_tr, out_root / "train" / "subhealthy", "acne")
    copy_many(sub_val, out_root / "val" / "subhealthy", "acne")
    copy_many(un_tr, out_root / "train" / "unhealthy", "skin")
    copy_many(un_val, out_root / "val" / "unhealthy", "skin")

    print("OK. 输出：", out_root)
    print("Counts:")
    print("  healthy:", len(healthy_imgs), "subhealthy:", len(subhealthy_imgs), "unhealthy:", len(unhealthy_imgs))


if __name__ == "__main__":
    main()

