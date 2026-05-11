import os
import sys
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "multimodal_mobilenet_savedmodel")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# 确保从 app/ 目录运行时也能导入项目模块
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.mobilenet_multimodal import IMAGE_SIZE  # noqa: E402
from app.advice_rules import get_rule_advice  # noqa: E402
from app.advice_ai import generate_ai_advice  # noqa: E402

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


def load_multimodal_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}，请先运行 train.py 完成训练。")
    # 优先尝试按 Keras 模型加载；如果该目录是 tf.saved_model.save 导出的（无 Keras metadata），则降级用 tf.saved_model.load
    try:
        return load_model(MODEL_PATH)
    except Exception:
        loaded = tf.saved_model.load(MODEL_PATH)
        infer = loaded.signatures.get("serving_default")
        if infer is None:
            raise ValueError("SavedModel 缺少 serving_default 签名，无法推理。")
        return infer


model = None
CLASS_NAMES = ["健康", "亚健康", "不健康"]


def preprocess_image(path):
    img = load_img(path, target_size=IMAGE_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_multimodal_model()

    face_file = request.files.get("face_image")
    tongue_file = request.files.get("tongue_image")

    if not face_file or not tongue_file:
        return jsonify({"success": False, "msg": "请同时上传面部图片和舌象图片"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    face_path = os.path.join(UPLOAD_DIR, f"face_{timestamp}.jpg")
    tongue_path = os.path.join(UPLOAD_DIR, f"tongue_{timestamp}.jpg")
    face_file.save(face_path)
    tongue_file.save(tongue_path)

    # 将上传后的图片地址返回给前端，便于在网页上回显
    face_image_url = url_for("uploaded_file", filename=os.path.basename(face_path))
    tongue_image_url = url_for("uploaded_file", filename=os.path.basename(tongue_path))

    try:
        face_arr = preprocess_image(face_path)
        tongue_arr = preprocess_image(tongue_path)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"success": False, "msg": "图片解析失败：请上传有效的图片文件（jpg/png 等）"}), 400

    # 兼容两种加载方式：
    # 1) Keras Model：model.predict((face_arr, tongue_arr))
    # 2) SavedModel signature：infer(face_input=..., tongue_input=...)
    if hasattr(model, "predict"):
        preds = model.predict((face_arr, tongue_arr))
        prob = preds[0]
    else:
        out = model(
            face_input=tf.convert_to_tensor(face_arr, dtype=tf.float32),
            tongue_input=tf.convert_to_tensor(tongue_arr, dtype=tf.float32),
        )
        # out 是 dict，取第一个输出张量
        prob = next(iter(out.values())).numpy()[0]
    idx = int(np.argmax(prob))
    label = CLASS_NAMES[idx]

    result = {
        "success": True,
        "label": label,
        "face_image_url": face_image_url,
        "tongue_image_url": tongue_image_url,
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, prob)},
        "advice": {
            "rule": get_rule_advice(label),
        },
    }
    return jsonify(result)


@app.route("/advice/ai", methods=["POST"])
def advice_ai():
    data = request.get_json(force=True)
    label = data.get("label", "")
    probabilities = data.get("probabilities", {})

    if not label:
        return jsonify({"success": False, "msg": "缺少 label 参数"}), 400

    ai_result = generate_ai_advice(label, probabilities)
    if "error" in ai_result:
        return jsonify({"success": False, "msg": ai_result["error"]}), 500

    return jsonify({
        "success": True,
        "advice": {"ai": ai_result},
    })


@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    # 防止路径穿越：只取文件名
    safe_name = os.path.basename(filename)
    return send_from_directory(UPLOAD_DIR, safe_name)


if __name__ == "__main__":
    # 开发环境直接运行
    app.run(host="0.0.0.0", port=5000, debug=True)

