import argparse
import os
import zipfile
from pathlib import Path


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser(description="使用 Python 解压 zip（支持更多压缩方法）")
    parser.add_argument("--zip", required=True, help="zip 文件路径")
    parser.add_argument("--out", required=True, help="解压输出目录")
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    out_dir = Path(args.out).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    unzip(zip_path, out_dir)
    print("OK:", zip_path, "->", out_dir)


if __name__ == "__main__":
    main()

