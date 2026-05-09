import glob
import os

import requests


def pick_one(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(pattern)
    return files[0]


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    face = pick_one(os.path.join(base, "data", "face", "val", "healthy", "*"))
    tongue = pick_one(os.path.join(base, "data", "tongue", "val", "healthy", "*"))
    print("face:", face)
    print("tongue:", tongue)

    with open(face, "rb") as f1, open(tongue, "rb") as f2:
        files = {"face_image": f1, "tongue_image": f2}
        r = requests.post("http://127.0.0.1:5000/predict", files=files, timeout=120)

    print("status:", r.status_code)
    print(r.text)


if __name__ == "__main__":
    main()

