import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    safe_mkdir(dst.parent)
    shutil.copy2(src, dst)


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if is_image(p)]


def split_paths(paths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = int(len(paths) * val_ratio)
    val = paths[:n_val]
    train = paths[n_val:]
    return train, val


def build_mapping_from_args(mapping_args: List[str]) -> Dict[str, str]:
    """
    解析 --map 传入的映射： source_label=target_label
    例如：
      --map mild= subhealthy
      --map severe=unhealthy
    """
    mapping: Dict[str, str] = {}
    for item in mapping_args:
        if "=" not in item:
            raise ValueError(f"映射格式错误：{item}，应为 source=target")
        src, tgt = item.split("=", 1)
        src = src.strip()
        tgt = tgt.strip()
        if tgt not in {"healthy", "subhealthy", "unhealthy"}:
            raise ValueError(f"target_label 必须是 healthy/subhealthy/unhealthy，当前：{tgt}")
        mapping[src] = tgt
    return mapping


def prepare_from_class_dirs(
    raw_root: Path,
    out_root: Path,
    modality: str,
    mapping: Dict[str, str],
    val_ratio: float,
    seed: int,
    max_per_class: int | None,
) -> None:
    """
    raw_root 目录结构假设为：
      raw_root/<source_label>/*.jpg
      raw_root/<source_label>/**/*.jpg  (允许嵌套)

    mapping 负责把 source_label 映射为 target_label（healthy/subhealthy/unhealthy）
    输出结构为：
      out_root/<modality>/train/<target_label>/
      out_root/<modality>/val/<target_label>/
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root 不存在：{raw_root}")

    for src_label_dir in [p for p in raw_root.iterdir() if p.is_dir()]:
        src_label = src_label_dir.name
        if src_label not in mapping:
            continue

        target_label = mapping[src_label]
        images = list_images(src_label_dir)
        if not images:
            continue

        if max_per_class is not None:
            images = images[:max_per_class]

        train_paths, val_paths = split_paths(images, val_ratio=val_ratio, seed=seed)

        for split_name, split_paths_list in [("train", train_paths), ("val", val_paths)]:
            for i, src_img in enumerate(split_paths_list):
                ext = src_img.suffix.lower()
                dst = out_root / modality / split_name / target_label / f"{src_label}_{i:06d}{ext}"
                copy_file(src_img, dst)


def main():
    parser = argparse.ArgumentParser(description="将公开数据集整理为项目 data/ 目录结构（train/val + 三分类映射）")
    parser.add_argument("--raw", required=True, help="原始数据根目录（内部按类别文件夹组织）")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"), help="输出 data 目录")
    parser.add_argument("--modality", required=True, choices=["face", "tongue"], help="数据模态：face 或 tongue")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=123, help="随机种子，默认 123")
    parser.add_argument("--max_per_class", type=int, default=0, help="每个 source 类最多取多少张；0 表示不限制")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="类别映射：source_label=healthy|subhealthy|unhealthy，可重复传入",
    )
    args = parser.parse_args()

    mapping = build_mapping_from_args(args.map)
    if not mapping:
        raise ValueError("必须至少提供一个 --map source=target 映射")

    raw_root = Path(args.raw).resolve()
    out_root = Path(args.out).resolve()
    max_per_class = None if args.max_per_class == 0 else args.max_per_class

    prepare_from_class_dirs(
        raw_root=raw_root,
        out_root=out_root,
        modality=args.modality,
        mapping=mapping,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=max_per_class,
    )

    print("完成。输出目录：", out_root / args.modality)
    print("提示：请检查 train/val 内三类样本数是否均衡，并在论文中说明映射与划分策略。")


if __name__ == "__main__":
    main()

