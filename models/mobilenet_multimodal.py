import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_v3


NUM_CLASSES = 3  # 健康 / 亚健康 / 不健康
IMAGE_SIZE = (224, 224)

def build_backbone(input_shape=IMAGE_SIZE + (3,), trainable=False, alpha=1.0, name_prefix=""):
    """
    构建单路 MobileNetV2，并通过强制重命名模型对象及子层来避开命名冲突。
    """
    # 1. 创建模型
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
        pooling="avg",
    )
    
    # 【关键】：修改模型顶层名称，防止内部 Model 对象重名
    base._name = f"{name_prefix}_mobilenet_v2"
    
    # 【关键】：遍历所有子层，修改子层名称。Keras 必须要求所有层的 name 唯一
    for layer in base.layers:
        layer._name = f"{name_prefix}_{layer.name}"
    
    base.trainable = trainable
    
    # 2. 定义输入
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    
    # 3. 预处理
    x = preprocess_input(inputs)
    
    # 4. 连接
    x = base(x)
    
    return inputs, x

def build_multimodal_mobilenet(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, alpha=1.0, backbone_trainable=False):
    """
    双分支多模态 MobileNet 模型：
    - 一路输入面部图片
    - 一路输入舌象图片
    - 特征拼接后进行全连接分类
    """
    face_input, face_feat = build_backbone(
        input_shape=image_size + (3,),
        trainable=backbone_trainable,
        alpha=alpha,
        name_prefix="face",
    )
    tongue_input, tongue_feat = build_backbone(
        input_shape=image_size + (3,),
        trainable=backbone_trainable,
        alpha=alpha,
        name_prefix="tongue",
    )

    merged = layers.Concatenate(name="feature_concat")([face_feat, tongue_feat])
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=[face_input, tongue_input], outputs=outputs, name="multimodal_mobilenet_v2")
    return model


def build_backbone_v3(input_shape=IMAGE_SIZE + (3,), trainable=False, alpha=1.0, name_prefix=""):
    """
    构建单路 MobileNetV3Small，并显式重命名所有子层避免双分支重名。
    """
    base = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
        pooling="avg",
    )
    base._name = f"{name_prefix}_mobilenet_v3_small"
    for layer in base.layers:
        layer._name = f"{name_prefix}_{layer.name}"

    base.trainable = trainable
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    x = preprocess_input_v3(inputs)
    x = base(x)
    return inputs, x


def build_multimodal_mobilenet_v3(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, alpha=1.0, backbone_trainable=False):
    """
    双分支多模态 MobileNetV3Small 模型，便于与 v2 做论文对比实验。
    """
    face_input, face_feat = build_backbone_v3(
        input_shape=image_size + (3,),
        trainable=backbone_trainable,
        alpha=alpha,
        name_prefix="face",
    )
    tongue_input, tongue_feat = build_backbone_v3(
        input_shape=image_size + (3,),
        trainable=backbone_trainable,
        alpha=alpha,
        name_prefix="tongue",
    )

    merged = layers.Concatenate(name="feature_concat")([face_feat, tongue_feat])
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=[face_input, tongue_input], outputs=outputs, name="multimodal_mobilenet_v3_small")
    return model


def build_multimodal_mobilenet_by_version(
    version="v2",
    num_classes=NUM_CLASSES,
    image_size=IMAGE_SIZE,
    alpha=1.0,
    backbone_trainable=False,
):
    """
    统一入口：version 可选 v2 / v3。
    """
    version = str(version).lower()
    if version == "v2":
        return build_multimodal_mobilenet(
            num_classes=num_classes,
            image_size=image_size,
            alpha=alpha,
            backbone_trainable=backbone_trainable,
        )
    if version == "v3":
        return build_multimodal_mobilenet_v3(
            num_classes=num_classes,
            image_size=image_size,
            alpha=alpha,
            backbone_trainable=backbone_trainable,
        )
    raise ValueError(f"不支持的模型版本: {version}，仅支持 v2/v3")


def compile_model(model: Model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

