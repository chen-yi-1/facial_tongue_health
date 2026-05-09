# 面部与舌象健康特征识别系统

本项目用于支撑毕业论文《基于 MobileNet 与 Flask 的面部与舌象健康特征识别系统》实验与系统实现。

## 一、项目目标

- 利用 **Python + TensorFlow/Keras** 实现基于 MobileNet 的轻量化健康特征分类模型；
- 支持 **面部图片 + 舌象图片** 双通道输入，对“健康 / 亚健康 / 不健康”进行分类；
- 通过 **Flask Web** 搭建简易测试系统，支持图片上传与诊断结果可视化；
- 为论文中的实验部分（模型结构、训练过程、指标对比、系统实现）提供可复现代码。

## 二、目录结构（建议）

```text
facial_tongue_health/
  data/
    face/
      train/healthy|subhealthy|unhealthy/
      val/healthy|subhealthy|unhealthy/
    tongue/
      train/healthy|subhealthy|unhealthy/
      val/healthy|subhealthy|unhealthy/
  models/
    mobilenet_multimodal.py        # 模型结构与构建函数
  saved_models/
    multimodal_mobilenet.h5       # 训练好的模型（训练后生成）
  app/
    app.py                        # Flask 后端
    templates/
      index.html                  # 前端页面
    static/
      css/style.css
      js/main.js
  train.py                        # 训练脚本（调用 models/mobilenet_multimodal）
  requirements.txt                # 依赖环境
  README.md
```

## 三、环境要求

- Python 3.10+
- 建议使用 `virtualenv` 或 Anaconda 创建虚拟环境
- 需要安装的核心库见 `requirements.txt`

## 四、基本使用流程

1. 准备数据集：按照 `data/` 下示例目录放置面部与舌象图片，并按 `healthy/subhealthy/unhealthy` 三类标注。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 训练模型（示例）：

```bash
python train.py
```

论文对比实验（`v2` vs `v3`）可通过环境变量切换：

```bash
# MobileNetV2（默认）
set MODEL_VERSION=v2
python train.py

# MobileNetV3Small
set MODEL_VERSION=v3
python train.py
```

模型会分别保存到 `saved_models/multimodal_mobilenet_v2_*` 与 `saved_models/multimodal_mobilenet_v3_*`，避免互相覆盖。

4. 启动 Web 服务：

```bash
cd app
python app.py
```

5. 浏览器访问 `http://127.0.0.1:5000` 进行测试。

## 五、数据整理（公开数据集落地到 data/）

如果你从 Kaggle / GitHub / Dryad 下载的数据集目录结构与本项目不一致，可使用脚本一键整理并划分训练/验证集：

```bash
python scripts/prepare_dataset.py --raw D:\datasets\face_skin_raw --modality face ^
  --map normal=healthy ^
  --map mild=subhealthy ^
  --map severe=unhealthy
```

舌象同理：

```bash
python scripts/prepare_dataset.py --raw D:\datasets\tongue_raw --modality tongue ^
  --map thin_white=healthy ^
  --map light_yellow=subhealthy ^
  --map thick_greasy=unhealthy
```

三分类映射规则与论文可用描述模板见：`docs/label_mapping_template.md`。

### 面部与舌象（Kaggle）一键整理

面部（FFHQ + acne + face-skin-disease）：

```bash
python scripts/prepare_face_from_kaggle.py --kaggle_root D:\clone\workspace\datasets_raw\kaggle --out_data_root D:\clone\workspace\facial_tongue_health\data --healthy_n 1000 --subhealthy_n 600 --unhealthy_n 600
```

舌象（tongue-images + tongue-coating，训练集不足部分使用增强补齐到目标规模）：

```bash
python scripts/prepare_tongue_from_kaggle.py --kaggle_root D:\clone\workspace\datasets_raw\kaggle --out_data_root D:\clone\workspace\facial_tongue_health\data --healthy_n 1000 --subhealthy_n 600 --unhealthy_n 600
```

## 六、公开数据下载（推荐：Dryad 舌象 + Kaggle 面部）

### 1）下载 Dryad 舌象（无需登录）

Dryad 的 TMC‑Tongue 数据集下载链接在网页中可直接获取。本项目提供脚本下载到 `datasets_raw/downloads/`：

```bash
python scripts/download_public_datasets.py --skip_kaggle
```

注意：Dryad 在部分网络环境下会启用浏览器验证，导致脚本下载到的是“Validating...”页面（文件很小）。遇到这种情况请改为浏览器手动下载，或直接使用 Kaggle 的舌象数据集（见下一节）。

### 2）下载 Kaggle 面部（需要 kaggle.json）

Kaggle 不需要申请权限，但需要一次性配置 API Token：

- Kaggle 网站右上角账号设置页创建 Token，得到 `kaggle.json`
- 放到 Windows 路径：`%USERPROFILE%\.kaggle\kaggle.json`

然后在 `scripts/download_public_datasets.py` 内取消注释你需要的 `kaggle_targets`，再运行：

```bash
python scripts/download_public_datasets.py
```

## 七、与论文写作的对应关系

- **选题依据与意义**：对应 README 中的项目目标与背景描述，可在论文绪论部分扩展。
- **研究内容与方法**：对应 `train.py`、`models/mobilenet_multimodal.py`、`app/` 中的代码实现。
- **实验设计与结果分析**：通过多次训练记录准确率、混淆矩阵等，并在论文中给出对比图表。

