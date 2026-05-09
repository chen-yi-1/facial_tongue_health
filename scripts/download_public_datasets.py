import argparse
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


DRYAD_TMC_TONGUE_ZIP_URL = "https://datadryad.org/downloads/file_stream/4540656"


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def kaggle_cmd(args: list[str]) -> list[str]:
    """
    Windows 上有时找不到 `kaggle.exe`（WinError 2）。
    使用 `python -c "from kaggle.cli import main; main()"` 可避免 PATH/entrypoint 问题。
    """
    code = "from kaggle.cli import main; main()"
    return [sys.executable, "-c", code, *args]


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    print("Downloading:", url)
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        },
    )
    with urlopen(req) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def kaggle_ready() -> bool:
    # Kaggle CLI looks for %USERPROFILE%\.kaggle\kaggle.json on Windows
    home = Path.home()
    return (home / ".kaggle" / "kaggle.json").exists()


def main():
    parser = argparse.ArgumentParser(description="下载公开数据集（Dryad + Kaggle）到 datasets_raw/")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "datasets_raw"), help="原始数据保存根目录")
    parser.add_argument("--skip_kaggle", action="store_true", help="只下载 Dryad，不下载 Kaggle")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dl_dir = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    dl_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dryad: TMC-Tongue
    # 注意：Dryad 站点在部分网络环境下会启用 Anubis（浏览器 JS 验证）导致脚本下载拿到“Validating...”页面。
    # 这种情况下请改为浏览器手动下载，或使用 Kaggle 的舌象数据集替代（本脚本支持 Kaggle 自动下载）。
    tmc_zip = dl_dir / "tmc_tongue_shezhen_datasets1.zip"
    try:
        download(DRYAD_TMC_TONGUE_ZIP_URL, tmc_zip)
        # 简单判断是否下载到了验证页
        if tmc_zip.exists() and tmc_zip.stat().st_size < 1024 * 1024:
            print(
                "提示：Dryad 可能返回了验证页（文件过小）。\n"
                "请用浏览器打开 Dryad 页面手动下载 zip，或改用 Kaggle 舌象数据集。"
            )
        else:
            print("Dryad downloaded:", tmc_zip)
    except Exception as e:
        print("Dryad 下载失败（可能被站点验证拦截）：", repr(e))
        print("你可以改用 Kaggle 舌象数据集（见下方 kaggle_targets）。")

    # 2) Kaggle datasets (optional)
    if args.skip_kaggle:
        print("Skip Kaggle downloads.")
        return

    if not kaggle_ready():
        print(
            "未检测到 Kaggle API 凭证：%USERPROFILE%\\.kaggle\\kaggle.json\n"
            "请到 Kaggle 账号页面创建 API Token，并将 kaggle.json 放到该路径后重试。\n"
            "（这不是申请权限，只是登录凭证）"
        )
        return

    # 你可按论文需要替换为更合适的数据集 slug
    # 示例：FFHQ（健康/正常面部候选）、acne、face-skin-disease（非健康候选）
    kaggle_targets = [
        # 舌象（公开、无需申请，但需 Kaggle token）
        ("jyotidabas/tongue-coating", "tongue_tongue_coating"),
        # 面部（健康/正常候选）
        ("greatgamedota/ffhq-face-data-set", "face_ffhq"),
        # 面部（非健康候选：痤疮/皮肤问题）
        ("nayanchaure/acne-dataset", "face_acne"),
        ("amellia/face-skin-disease", "face_skin_disease"),
    ]

    for slug, subdir in kaggle_targets:
        out_dir = root / "kaggle" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        print("Kaggle downloading:", slug, "->", out_dir)
        run(kaggle_cmd(["datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"]))


if __name__ == "__main__":
    main()

