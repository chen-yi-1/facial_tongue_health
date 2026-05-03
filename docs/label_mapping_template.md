# 三分类映射模板（健康 / 亚健康 / 不健康）

本文件用于帮助你把公开数据集中“多类别标签”合并为论文所需的三分类标签，并给出论文可直接使用的表述模板。

> 重要：公开数据集通常不直接提供“亚健康”标签，因此需要你在论文中明确“亚健康”的定义与构建规则（例如轻度异常、疲劳相关外观特征、问卷辅助等），并在实验部分说明可能的偏差来源。

## 一、三分类标签定义（论文可用文字）

- **健康（healthy）**：面部/皮肤外观无明显异常（如严重炎症、明显红斑/皮疹等），舌象表现接近正常（舌质淡红、苔薄白或少苔），图像质量清晰、光照正常。
- **亚健康（subhealthy）**：介于健康与不健康之间的轻度异常状态，主要表现为轻度皮肤问题或疲劳特征（如轻度暗沉、轻微痘痘/泛红、轻度黑眼圈等）或舌苔轻度偏厚、略黄等；该类样本的构建可采用“轻度异常规则 + 人工复核”方式。
- **不健康（unhealthy）**：存在较明显的皮肤或面部异常特征（如中重度痤疮、显著红斑/炎症、明显皮疹等）或明显舌象异常（如苔厚腻、苔黄/黑、舌面异常明显等）。

## 二、公开数据集到三分类的合并示例（你可按实际数据集调整）

### 1）面部皮肤问题数据集（如 acne / face-skin-disease）

常见源标签（示例）到三分类（建议）：

| 源标签（source_label） | 建议三分类（target） | 说明 |
|---|---|---|
| normal / clear / healthy | healthy | 作为健康样本 |
| mild_acne / mild_redness | subhealthy | 轻度异常，建议人工抽检确认 |
| moderate_acne / severe_acne | unhealthy | 明显异常 |
| rosacea / dermatitis / eczema（若面部为主） | unhealthy 或 subhealthy | 视严重程度与标注而定 |

对应数据整理脚本的映射参数示例：

```bash
python scripts/prepare_dataset.py --raw D:\datasets\face_skin_raw --modality face ^
  --map normal=healthy ^
  --map mild_acne=subhealthy ^
  --map severe_acne=unhealthy
```

### 2）通用人脸数据集（如 FFHQ / selfie 数据集）

这类数据一般没有健康标签，建议作为 **healthy 候选池**，并执行以下筛选策略：

- **自动筛除**：过度模糊、遮挡严重、非正脸、分辨率过低。
- **人工复核**：随机抽样人工检查是否存在明显皮肤异常；若存在则剔除或转入 subhealthy/unhealthy 候选。

论文中写法建议：
- “选取通用人脸数据集中图像质量较高且无明显面部异常特征的样本作为健康类数据，并进行人工抽样复核。”

### 3）舌象数据集（Tongue Coating / TMC‑Tongue 等）

舌象数据常见多类别可能包括：薄白苔、厚白苔、黄苔、黑苔、剥苔、少苔等（不同数据集命名不同）。

建议合并策略（示例）：

| 舌苔/舌象源标签（source_label） | 建议三分类（target） | 说明 |
|---|---|---|
| thin_white / normal | healthy | 接近正常 |
| slightly_thick / light_yellow | subhealthy | 轻度异常，作为亚健康候选 |
| thick_greasy / yellow / black / severe_abnormal | unhealthy | 明显异常 |

脚本映射参数示例：

```bash
python scripts/prepare_dataset.py --raw D:\datasets\tongue_raw --modality tongue ^
  --map thin_white=healthy ^
  --map light_yellow=subhealthy ^
  --map thick_greasy=unhealthy
```

## 三、论文中必须写清楚的点（建议写法）

- **数据来源与许可**：说明数据来自公开数据集（Kaggle/GitHub/Dryad），用于学术研究，并遵循数据集原许可协议。
- **标注与合并规则**：说明从多类别到三分类的合并原则（表格+文字）。
- **亚健康构建方法**：说明亚健康类采用“轻度异常规则 + 人工复核/抽检”的策略，以及其不确定性。
- **数据划分策略**：训练集/验证集比例、随机种子、是否按人划分（若存在同一人多图，建议按人划分以避免泄漏）。

