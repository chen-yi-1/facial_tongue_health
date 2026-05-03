import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_v3

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

CLASS_NAMES = ["healthy", "subhealthy", "unhealthy"]
NUM_CLASSES = 3
IMAGE_SIZE = (224, 224)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(dir_path: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.iterdir() if p.is_file() and p.name.lower().endswith(exts)]
    files.sort()
    return files


def decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def make_single_dataset(root: Path, subset: str, batch_size: int, seed: int) -> Tuple[tf.data.Dataset, List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for y, cls in enumerate(CLASS_NAMES):
        cls_dir = root / subset / cls
        imgs = list_images(cls_dir)
        paths.extend([str(p) for p in imgs])
        labels.extend([y] * len(imgs))

    if not paths:
        raise FileNotFoundError(f"未找到数据：{root}/{subset}")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def _map(p, y):
        x = decode_and_resize(p)
        return x, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels


def make_paired_dataset(face_root: Path, tongue_root: Path, subset: str, batch_size: int, seed: int) -> Tuple[tf.data.Dataset, List[int]]:
    face_paths: List[str] = []
    tongue_paths: List[str] = []
    labels: List[int] = []

    for y, cls in enumerate(CLASS_NAMES):
        f = list_images(face_root / subset / cls)
        t = list_images(tongue_root / subset / cls)
        n = min(len(f), len(t))
        if n == 0:
            continue
        face_paths.extend([str(p) for p in f[:n]])
        tongue_paths.extend([str(p) for p in t[:n]])
        labels.extend([y] * n)

    if not face_paths:
        raise FileNotFoundError("未找到可配对的 face/tongue 数据")

    ds = tf.data.Dataset.from_tensor_slices((face_paths, tongue_paths, labels))
    ds = ds.shuffle(buffer_size=len(face_paths), seed=seed, reshuffle_each_iteration=True)

    def _map(fp, tp, y):
        fx = decode_and_resize(fp)
        tx = decode_and_resize(tp)
        return (fx, tx), y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels


def build_single_mobilenet(name_prefix: str, trainable_backbone: bool, backbone_version: str = "v2") -> Model:
    backbone_version = backbone_version.lower()
    if backbone_version == "v2":
        base = MobileNetV2(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        pre_fn = preprocess_input
        base_name = "mobilenet_v2"
        model_suffix = "v2"
    elif backbone_version == "v3":
        base = MobileNetV3Small(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        pre_fn = preprocess_input_v3
        base_name = "mobilenet_v3_small"
        model_suffix = "v3"
    else:
        raise ValueError(f"unsupported backbone_version: {backbone_version}")

    base.trainable = trainable_backbone
    base._name = f"{name_prefix}_{base_name}"
    for layer in base.layers:
        layer._name = f"{name_prefix}_{layer.name}"

    inp = layers.Input(shape=IMAGE_SIZE + (3,), name=f"{name_prefix}_input")
    x = pre_fn(inp)
    x = base(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return Model(inputs=inp, outputs=out, name=f"{name_prefix}_single_mobilenet_{model_suffix}")


def build_multimodal(trainable_backbone: bool, backbone_version: str = "v2") -> Model:
    # 用与项目一致的结构，但不依赖其它模块，保证脚本自包含可复现
    backbone_version = backbone_version.lower()

    def backbone(prefix: str):
        if backbone_version == "v2":
            base = MobileNetV2(
                input_shape=IMAGE_SIZE + (3,),
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            pre_fn = preprocess_input
            base_name = "mobilenet_v2"
            model_suffix = "v2"
        elif backbone_version == "v3":
            base = MobileNetV3Small(
                input_shape=IMAGE_SIZE + (3,),
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            pre_fn = preprocess_input_v3
            base_name = "mobilenet_v3_small"
            model_suffix = "v3"
        else:
            raise ValueError(f"unsupported backbone_version: {backbone_version}")

        base.trainable = trainable_backbone
        base._name = f"{prefix}_{base_name}"
        for layer in base.layers:
            layer._name = f"{prefix}_{layer.name}"
        inp = layers.Input(shape=IMAGE_SIZE + (3,), name=f"{prefix}_input")
        x = pre_fn(inp)
        x = base(x)
        return inp, x, model_suffix

    face_in, face_feat, model_suffix = backbone("face")
    tongue_in, tongue_feat, _ = backbone("tongue")
    merged = layers.Concatenate(name="feature_concat")([face_feat, tongue_feat])
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return Model(inputs=[face_in, tongue_in], outputs=out, name=f"multimodal_mobilenet_{model_suffix}")


def compile_model(m: Model, lr: float) -> Model:
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def plot_history(hist: Dict[str, List[float]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=30)
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def eval_model(m: Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for x, y in ds:
        probs = m.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def run_one(
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    trainable_backbone: bool,
    backbone_version: str,
    seed: int,
    out_dir: Path,
) -> Dict:
    if mode == "face":
        train_ds, _ = make_single_dataset(DATA_DIR / "face", "train", batch_size, seed)
        val_ds, _ = make_single_dataset(DATA_DIR / "face", "val", batch_size, seed + 1)
        model = compile_model(build_single_mobilenet("face", trainable_backbone, backbone_version=backbone_version), lr)
    elif mode == "tongue":
        train_ds, _ = make_single_dataset(DATA_DIR / "tongue", "train", batch_size, seed)
        val_ds, _ = make_single_dataset(DATA_DIR / "tongue", "val", batch_size, seed + 1)
        model = compile_model(build_single_mobilenet("tongue", trainable_backbone, backbone_version=backbone_version), lr)
    elif mode == "multimodal":
        train_ds, _ = make_paired_dataset(DATA_DIR / "face", DATA_DIR / "tongue", "train", batch_size, seed)
        val_ds, _ = make_paired_dataset(DATA_DIR / "face", DATA_DIR / "tongue", "val", batch_size, seed + 1)
        model = compile_model(build_multimodal(trainable_backbone, backbone_version=backbone_version), lr)
    else:
        raise ValueError(mode)

    # 手动算 val 指标，用于历史曲线绘图（避免你环境里 Keras validation 触发奇怪结构错误）
    hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for ep in range(epochs):
        h = model.fit(train_ds, epochs=1, verbose=1)
        metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        hist["loss"].append(float(h.history["loss"][-1]))
        hist["accuracy"].append(float(h.history["accuracy"][-1]))
        hist["val_loss"].append(float(metrics["loss"]))
        hist["val_accuracy"].append(float(metrics["accuracy"]))
        print(f"[{mode}] epoch={ep+1}/{epochs} val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f}")

    # 预测 + 指标
    y_true, y_pred = eval_model(model, val_ds)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)

    ensure_dir(out_dir)
    plot_history(hist, f"{mode} loss/acc", out_dir / f"{mode}_curve.png")
    plot_confusion(cm, f"{mode} confusion", out_dir / f"{mode}_confusion.png")
    (out_dir / f"{mode}_classification_report.txt").write_text(report_txt, encoding="utf-8")
    (out_dir / f"{mode}_metrics.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "trainable_backbone": trainable_backbone,
                "val_macro_precision": float(pr),
                "val_macro_recall": float(rc),
                "val_macro_f1": float(f1),
                "val_confusion_matrix": cm.tolist(),
                "history": hist,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "mode": mode,
        "val_macro_precision": float(pr),
        "val_macro_recall": float(rc),
        "val_macro_f1": float(f1),
        "val_accuracy_last": float(hist["val_accuracy"][-1]),
        "out_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="跑同一 backbone 版本的 face/tongue/multimodal 对比实验，并输出曲线、混淆矩阵、PRF")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trainable_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backbone_version", type=str, default="v3", choices=["v2", "v3"])
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = REPORTS_DIR / f"exp_{ts}_{args.backbone_version}"
    ensure_dir(out_dir)

    summary = []
    for mode in ["face", "tongue", "multimodal"]:
        summary.append(
            run_one(
                mode=mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                trainable_backbone=args.trainable_backbone,
                backbone_version=args.backbone_version,
                seed=args.seed,
                out_dir=out_dir,
            )
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("==> DONE. Results in:", out_dir)
    for row in summary:
        print(row["mode"], "val_acc_last=", f'{row["val_accuracy_last"]:.4f}', "macro_f1=", f'{row["val_macro_f1"]:.4f}')


if __name__ == "__main__":
    main()

