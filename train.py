import os
import math
from typing import List, Tuple

import tensorflow as tf

from models.mobilenet_multimodal import (
    build_multimodal_mobilenet_by_version,
    compile_model,
    IMAGE_SIZE,
    NUM_CLASSES,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FACE_DIR = os.path.join(DATA_DIR, "face")
TONGUE_DIR = os.path.join(DATA_DIR, "tongue")
BATCH_SIZE = 16
EPOCHS = int(os.environ.get("EPOCHS", "10"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v2").strip().lower()
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, f"multimodal_mobilenet_{MODEL_VERSION}_savedmodel")
WEIGHTS_PATH = os.path.join(MODEL_DIR, f"multimodal_mobilenet_{MODEL_VERSION}_weights.h5")

CLASS_NAMES = ["healthy", "subhealthy", "unhealthy"]

def _list_images(dir_path: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.exists(dir_path):
        return []
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and name.lower().endswith(exts):
            files.append(p)
    files.sort()
    return files


def _decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def create_paired_dataset(
    face_root: str,
    tongue_root: str,
    subset: str,
    batch_size: int,
    seed: int = 123,
) -> Tuple[tf.data.Dataset, int]:
    """
    更稳妥的多模态配对方式：按类别分别取两路样本，取 min 数量后按索引配对。
    注意：这不代表“同一人/同一时刻”的真实配对，只是用于多模态融合训练的可复现实验设定。
    """
    face_subset = os.path.join(face_root, subset)
    tongue_subset = os.path.join(tongue_root, subset)

    face_paths: List[str] = []
    tongue_paths: List[str] = []
    labels: List[int] = []

    for label, cls in enumerate(CLASS_NAMES):
        f_cls = _list_images(os.path.join(face_subset, cls))
        t_cls = _list_images(os.path.join(tongue_subset, cls))
        n = min(len(f_cls), len(t_cls))
        if n == 0:
            continue
        face_paths.extend(f_cls[:n])
        tongue_paths.extend(t_cls[:n])
        labels.extend([label] * n)

    if not face_paths:
        raise FileNotFoundError("未找到可配对的面部/舌象数据，请检查 data 目录。")

    ds = tf.data.Dataset.from_tensor_slices((face_paths, tongue_paths, labels))
    ds = ds.shuffle(buffer_size=len(face_paths), seed=seed, reshuffle_each_iteration=True)

    def _map(face_p, tongue_p, y):
        face_img = _decode_and_resize(face_p)
        tongue_img = _decode_and_resize(tongue_p)
        # 用 tuple 对齐模型的两个输入，避免 dict 适配器在部分版本下异常
        return ((face_img, tongue_img), y)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps = math.ceil(len(face_paths) / batch_size)
    return ds, steps


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if MODEL_VERSION not in {"v2", "v3"}:
        raise ValueError(f"MODEL_VERSION 仅支持 v2/v3，当前为: {MODEL_VERSION}")

    print(f"==> 构建多模态 MobileNet 模型，版本: {MODEL_VERSION} ...")
    model = build_multimodal_mobilenet_by_version(
        version=MODEL_VERSION,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        alpha=1.0,
        backbone_trainable=False,
    )
    model = compile_model(model, learning_rate=1e-4)
    model.summary()

    print("==> 加载训练集与验证集...")
    train_ds, train_steps = create_paired_dataset(FACE_DIR, TONGUE_DIR, subset="train", batch_size=BATCH_SIZE)
    val_ds, val_steps = create_paired_dataset(FACE_DIR, TONGUE_DIR, subset="val", batch_size=BATCH_SIZE, seed=456)
    print(f"==> 验证集 steps: {val_steps}")
    print("==> 预跑一次验证集 evaluate（用于排查数据/结构问题）...")
    model.evaluate(val_ds, verbose=1)

    class ManualValidation(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            metrics = self.model.evaluate(val_ds, verbose=0, return_dict=True)
            # Keras 的 ModelCheckpoint/EarlyStopping 习惯使用 val_* 键
            for k, v in metrics.items():
                logs[f"val_{k}"] = v
            msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k.startswith("val_"))
            print(f"\n[manual val] epoch={epoch + 1} {msg}")

    # 说明：在部分 TF/Keras 版本中，ModelCheckpoint 保存 .h5 可能触发结构打包异常。
    # 这里先使用手动验证回调，训练完成后再显式 model.save()。
    callbacks = [ManualValidation()]

    print("==> 开始训练...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("==> 训练完成，保存模型...")
    # 在部分 TF/Keras 版本中，直接 model.save() 可能触发结构打包异常；
    # 优先导出 SavedModel，并额外保存一份权重文件便于复现。
    tf.saved_model.save(model, MODEL_PATH)
    model.save_weights(WEIGHTS_PATH)
    print(f"==> SavedModel 已保存到: {MODEL_PATH}")
    print(f"==> Weights 已保存到: {WEIGHTS_PATH}")
    return history


if __name__ == "__main__":
    main()

