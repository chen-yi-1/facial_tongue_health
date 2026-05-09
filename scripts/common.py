import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def split_paths(paths: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = int(len(paths) * val_ratio)
    return paths[n_val:], paths[:n_val]


def copy_many(paths: Iterable[Path], dst_dir: Path, prefix: str) -> int:
    ensure_dir(dst_dir)
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
