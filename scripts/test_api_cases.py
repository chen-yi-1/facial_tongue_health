import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
API_URL = "http://127.0.0.1:5000/predict"


def pick_one(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(pattern)
    return files[0]


def post_predict(face_path, tongue_path, timeout=120):
    files = {}
    handles = []
    try:
        if face_path is not None:
            f = open(face_path, "rb")
            handles.append(f)
            files["face_image"] = f
        if tongue_path is not None:
            f = open(tongue_path, "rb")
            handles.append(f)
            files["tongue_image"] = f
        r = requests.post(API_URL, files=files, timeout=timeout)
        return r.status_code, r.text
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass


def ensure_server_up() -> None:
    # 简单探测：发一个缺文件请求应返回 400
    code, _ = post_predict(None, None, timeout=20)
    if code not in (400, 405):
        raise RuntimeError("后端服务可能未启动或端口异常，请先运行 app/app.py")


def main():
    ensure_server_up()

    report = {"api_url": API_URL, "cases": []}

    # Case A: 三类各跑一次（用 val 集做样本来源）
    for cls in ["healthy", "subhealthy", "unhealthy"]:
        face = pick_one(os.path.join(DATA_DIR, "face", "val", cls, "*"))
        tongue = pick_one(os.path.join(DATA_DIR, "tongue", "val", cls, "*"))
        t0 = time.time()
        code, text = post_predict(face, tongue)
        dt = time.time() - t0
        report["cases"].append(
            {
                "name": f"predict_{cls}",
                "status_code": code,
                "latency_s": round(dt, 3),
                "face": os.path.basename(face),
                "tongue": os.path.basename(tongue),
                "response": text[:2000],
            }
        )

    # Case B: 缺文件
    face_ok = pick_one(os.path.join(DATA_DIR, "face", "val", "healthy", "*"))
    tongue_ok = pick_one(os.path.join(DATA_DIR, "tongue", "val", "healthy", "*"))
    report["cases"].append({"name": "missing_tongue", "status_code": post_predict(face_ok, None)[0]})
    report["cases"].append({"name": "missing_face", "status_code": post_predict(None, tongue_ok)[0]})

    # Case C: 非图片文件
    tmp_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    bad_path = os.path.join(tmp_dir, "not_image.txt")
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("not an image")
    report["cases"].append({"name": "invalid_face_file", "status_code": post_predict(bad_path, tongue_ok)[0]})

    # Case D: 连续并发 10 次（健康样本），检查是否有 500
    def one_call():
        return post_predict(face_ok, tongue_ok)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(one_call) for _ in range(10)]
        for fu in as_completed(futs):
            results.append(fu.result())
    dt = time.time() - t0
    report["cases"].append(
        {
            "name": "concurrency_10",
            "latency_s_total": round(dt, 3),
            "status_codes": [c for c, _ in results],
            "any_500": any(c >= 500 for c, _ in results),
        }
    )

    out_path = os.path.join(BASE_DIR, "test_report_api.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("OK. report saved to:", out_path)
    for c in report["cases"]:
        print(c["name"], "->", c.get("status_code", ""), c.get("any_500", ""))


if __name__ == "__main__":
    main()

