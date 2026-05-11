## 第五章 系统设计与实现

> **使用说明：** 本章正文约 8500 字，配图 10 张 + 表 1 张。`[图5-X: ...]` 为图片插入标注，请用 Word 插入→图片替换。PlantUML 图见同目录 `.puml` 文件，先渲染为 PNG 再插入。`[表5-1: ...]` 为表格插入标注。

---

### 5.1 系统总体架构设计

第四章通过对比实验验证了 MobileNet 双分支多模态融合模型在三分类健康状态判别任务中的有效性。为使该模型能够在实际场景中被便捷地使用，本章基于 Flask Web 框架设计并实现了一个完整的原型系统，将训练好的深度学习模型封装为可在线访问的推理服务，并提供直观的前端交互界面。

**开发与运行环境**

系统的开发与运行环境配置如下：操作系统为 Windows 11（64 位），编程语言为 Python 3.10，深度学习框架采用 TensorFlow 2.10.0（配套 CUDA 11.2 与 cuDNN 8.1.0 实现 GPU 加速），Web 框架采用 Flask 3.0.2。前端部分采用原生 HTML5、CSS3 与 JavaScript 实现，不依赖任何前端框架，以保持项目的轻量化与低耦合特性。

**分层架构**

系统采用四层分层架构设计，自底向上依次为：

- **数据存储层：** 负责管理用户上传的面部与舌象图片文件，以及训练好的 TensorFlow 模型权重文件。上传图片通过时间戳命名存储于 `uploads/` 目录，模型文件以 SavedModel 格式存放于 `saved_models/` 目录。
- **算法推理层：** 封装深度学习模型的加载、图像预处理与推理逻辑，提供统一的 `predict()` 接口供上层调用。该层兼容 Keras H5 模型与 TensorFlow SavedModel 两种格式，确保在不同 TF 版本下的可用性。
- **Flask 应用层：** 作为系统的核心调度中枢，负责接收 HTTP 请求、路由分发、调用推理层获取分类结果，并整合规则引擎与 AI 建议模块生成健康建议。应用层还承担文件验证、异常处理和响应组装等职责。
- **前端展示层：** 基于 HTML/CSS/JavaScript 构建的响应式 Web 页面，提供图片上传、结果可视化和健康建议展示等功能。采用 Fetch API 与后端进行异步数据交互，避免页面刷新。

**技术选型理由**

本章选择 Flask 而非 Django 等重量级框架，主要基于以下考量：（1）系统功能聚焦于单一模型的推理服务，接口数量少（4 个路由），无需 ORM、用户认证等重型组件；（2）Flask 的轻量化特性降低了部署门槛，与 MobileNet 模型"边缘部署"的目标一致；（3）Flask 的路由装饰器机制使代码结构清晰，便于论文展示和后续扩展。

前端未引入 React 或 Vue 等框架，原因在于本系统的前端交互逻辑相对简单，核心功能仅为表单提交与结果渲染。原生 JS 实现可消除构建工具链依赖，减少项目体积，同时避免框架版本迭代带来的维护负担。

[图5-1: UML 组件图 — 系统组件架构图。展示了 Browser、Flask Router、Model Inference (TensorFlow)、Rule Advice Engine、DeepSeek API 五个组件及其依赖关系。使用 PlantUML 绘制，代码见 components_diagram.puml]

**核心调用关系**

系统运行期间的两条核心调用链路如下：

1. **/predict 主流：** 用户在浏览器选择面/舌图片 → 点击"开始分析" → Flask 接收 POST 请求 → 依次对面部和舌象图片执行预处理（Resize 224×224、归一化至 [0,1]、扩维为 batch）→ 调用 TensorFlow 多输入模型推理 → 取 Softmax 输出最大值对应的类别标签 → 调用规则引擎 `get_rule_advice()` 获取对应维度的标准化建议 → 组装 JSON 响应返回前端 → 前端渲染分类结果、概率柱状图与标准建议卡片。

2. **/advice/ai 辅路：** 前端在渲染标准建议后，异步发起 POST 请求 → Flask 调用 DeepSeek Chat API，携带 System Prompt（设定健康顾问角色与 5 维度输出格式约束）和 User Prompt（用户检测结果数据）→ DeepSeek 返回 JSON 格式的 5 维度个性化建议 → Flask 校验维度完整性后返回前端 → 用户点击"AI 建议"标签即可查看。

上述两条链路的设计遵循"先同步获取核心结果、再异步增强体验"的策略：标准建议由本地规则引擎提供，零延迟即时展示；AI 建议依赖外部 API，异步加载且不影响核心功能可用性。

---

### 5.2 后端服务设计与实现

#### 5.2.1 Flask 路由与 API 设计

系统后端基于 Flask 框架构建，共定义了四个路由端点，分别承担页面渲染、模型预测、AI 建议生成和静态文件服务职责。各路由的详细信息汇总如表 5-1 所示。

[表5-1: API 接口汇总表]

| 路由 | 方法 | 入参 | 返回 | 说明 |
|------|------|------|------|------|
| / | GET | 无 | text/html | 返回前端页面（index.html） |
| /predict | POST | FormData: face_image(jpg/png) + tongue_image(jpg/png) | JSON: {success, label, probabilities, advice:{rule}} | 多模态健康状态预测 + 规则建议 |
| /advice/ai | POST | JSON: {label, probabilities} | JSON: {success, advice:{ai}} | AI 个性化生活建议 |
| /uploads/<filename> | GET | 无 | image/jpeg | 安全文件回显（已做路径穿越防护） |

**表5-1 API 接口汇总表**

各接口的设计遵循 RESTful 风格，使用 HTTP 状态码表达请求结果：200 表示成功，400 表示客户端参数错误（如图片缺失或格式非法），500 表示服务端异常（如 AI 服务不可用）。

`/predict` 接口是系统的核心入口，采用 `multipart/form-data` 编码接收图片文件，原因在于图片数据量较大，JSON 编码会导致 Base64 膨胀约 33%，而 FormData 可保持二进制传输效率。接口返回的 JSON 结构如下：

```json
{
  "success": true,
  "label": "亚健康",
  "face_image_url": "/uploads/face_20260511_143025.jpg",
  "tongue_image_url": "/uploads/tongue_20260511_143025.jpg",
  "probabilities": {"健康": 0.12, "亚健康": 0.73, "不健康": 0.15},
  "advice": {
    "rule": {
      "饮食调理": "均衡营养，增加深色蔬菜...",
      "作息建议": "尽量在23点前入睡...",
      "运动指导": "每周3-5次中等强度有氧运动...",
      "中医调理": "根据体质辨证调养...",
      "心理调节": "学会科学减压..."
    }
  }
}
```

其中 `advice.rule` 字段由规则引擎根据预测标签即时生成，使单次请求即可获得完整的"判定+建议"结果，减少前端请求次数。

`/advice/ai` 接口采用独立的 POST 路由，入参为 JSON 格式的 `label` 和 `probabilities`，而非重复上传图片文件。这样设计的原因是 AI 建议生成仅依赖预测结果，与原始图像无关，分离接口可以避免冗余传输。此外，独立路由使得 AI 服务的故障不会影响核心的 `/predict` 预测功能，实现了故障隔离。

#### 5.2.2 模型推理模块

模型推理模块负责将训练好的深度学习模型封装为可供 Web 应用调用的推理服务。其核心挑战在于兼容不同 TensorFlow 版本的模型导出格式。

**双格式兼容加载**

在第四章实验中，训练脚本将模型导出为两个副本：TensorFlow SavedModel 格式和 Keras H5 格式。SavedModel 是 TF 官方推荐的通用格式，但某些 TF 版本在 `tf.saved_model.save()` 后不保留 Keras metadata，导致 `tf.keras.models.load_model()` 加载失败。为此，推理模块实现了分级尝试策略：

1. 优先调用 `tf.keras.models.load_model()` 加载，若成功则获得 Keras Model 实例，可直接使用 `model.predict((face_arr, tongue_arr))` 进行双输入推理。
2. 若加载失败，则降级使用 `tf.saved_model.load()` 加载 SavedModel，通过 `signatures["serving_default"]` 获取推理函数，调用方式变为 `infer(face_input=tensor, tongue_input=tensor)`，输出为字典形式，需通过 `next(iter(out.values()))` 提取张量。

两种加载路径被封装在 `load_multimodal_model()` 函数中，模块仅在首次接收到 `/predict` 请求时执行懒加载，随后缓存在全局变量 `model` 中，后续请求直接复用，避免了重复加载开销。

**图像预处理流水线**

前端上传的图片为任意尺寸的 RGB 图像，需要经过统一的预处理流水线才能输入模型：

1. **尺寸归一化：** 使用 Keras 的 `load_img()` 将图片缩放至 224×224 像素，与训练时的输入尺寸一致。
2. **数值归一化：** 使用 `img_to_array()` 将像素值从 [0,255] 映射至 [0,1] 浮点区间。
3. **维度扩展：** 使用 NumPy 的 `np.expand_dims(arr, axis=0)` 在第 0 轴添加 batch 维度，将单张图片的 shape 从 (224, 224, 3) 扩展为 (1, 224, 224, 3)。

面部与舌象两张图片分别经过上述处理，得到两个独立的 batch-1 张量，随后一对一并传入模型实现端到端推理。

**输出解析**

模型的输出为形状 (1, 3) 的 Softmax 概率向量，三个值分别对应"健康""亚健康""不健康"的概率。模块通过 `np.argmax(prob)` 获取最大概率索引，映射到 `CLASS_NAMES = ["健康", "亚健康", "不健康"]` 得到类别标签，同时将三个概率值转为 Python float 类型以便 JSON 序列化。

#### 5.2.3 生活建议规则引擎

在获得健康状态判定结果后，系统向用户提供基于权威资料的标准健康建议，帮助用户理解判定结果并获取初步的自我调理方向。规则引擎的实现文件为 `app/advice_rules.py`。

**规则设计**

规则的编制参考了以下权威资料：国家卫健委《中国居民膳食指南（2022）》、国家中医药管理局《中国公民中医养生保健素养》、中华中医药学会《亚健康中医临床指南》、国家卫健委《心理健康素养十条》。

每条规则覆盖五个生活维度：饮食调理、作息建议、运动指导、中医调理、心理调节。三个健康类别（健康、亚健康、不健康）各有一套独立建议，构成 3×5=15 条规则的矩阵。

"健康"类别的建议侧重维持现有状态、顺应四季养生，内容较为积极；"亚健康"类别的建议聚焦于调整不良习惯、进行体质辨调，强调可操作性；"不健康"类别的建议以"及时就医"为前提，在此基础上提供辅助性的生活调养指导，避免用户因依赖系统建议而延误诊疗。

**引擎实现**

规则数据以 Python 嵌套字典结构 `ADVICE_RULES` 组织，外层键为类别标签，内层键为维度名称，每条建议文本约 80-150 字，表述具体可操作。对外接口为 `get_rule_advice(label: str) -> dict[str, str]` 函数，接受类别标签参数并返回对应维度的建议字典。函数对输入标签执行严格的 KeyError 检查，若传入未知标签（如通过 URL 手动篡改请求参数），则立即抛出异常并提示合法取值范围，防止因数据异常导致前端展示错误。

由于规则建议基于本地静态数据生成，响应时间为毫秒级，因此被设计为 `/predict` 接口的同步返回字段，用户无需额外等待即可即刻获得标准建议。

#### 5.2.4 AI 智能建议模块

规则引擎虽然响应快速且内容权威，但其建议是预定义的静态文本，无法根据用户的个体概率分布进行差异化调整。为进一步提升建议的针对性和个性化程度，系统引入了基于 DeepSeek 大语言模型的 AI 智能建议模块，实现文件为 `app/advice_ai.py`。

**API 调用设计**

模块调用 DeepSeek Chat API（端点：`https://api.deepseek.com/v1/chat/completions`），模型选用 `deepseek-chat`。每次请求携带以下参数：temperature 设为 0.7 以保证生成多样性，max_tokens 设为 1024 以控制响应长度，timeout 设为 30 秒以防止长时间阻塞。

API Key 通过环境变量 `DEEPSEEK_API_KEY` 注入，模块在启动时从项目根目录的 `.env` 文件中手动解析加载（避免引入 python-dotenv 等第三方依赖），该文件已被 `.gitignore` 排除在版本控制之外，防止 API Key 泄露。

**Prompt 工程设计**

Prompt 是决定 AI 输出质量的关键。模块采用 System Prompt + User Prompt 的双层结构：

- **System Prompt（系统级指令）：** 设定 AI 角色为"基于中医养生和现代医学的健康顾问"，明确要求从五个维度（饮食调理、作息建议、运动指导、中医调理、心理调节）给出建议，每条约 30-80 字且具体可操作，禁止空泛表述。最关键的是，要求以纯 JSON 格式返回，不允许附加任何解释性文字，确保后端可直接 `json.loads()` 解析。
- **User Prompt（用户级输入）：** 动态注入用户的检测结果数据，包括综合判定标签和各维度概率值，格式为 JSON 字符串，使 AI 能够根据个体差异给出针对性建议。

**错误处理与容错**

AI 服务属于外部依赖，存在网络波动、API 配额耗尽、返回格式异常等不可控因素。模块设计了多层容错机制：

1. API Key 缺失时，直接返回错误提示"未配置 DEEPSEEK_API_KEY 环境变量"，不发起网络请求。
2. 网络请求异常（超时、连接失败、HTTP 错误状态码）由 `requests.RequestException` 统一捕获，返回友好的错误消息。
3. AI 返回内容无法解析为 JSON 时，返回"AI 返回格式异常"提示。
4. AI 返回的 JSON 缺少某个维度时，自动补全为针对该维度的兜底建议文本。

这些容错措施确保了 AI 建议模块的故障不会影响系统其他功能的正常运行。前端在 `/advice/ai` 失败时，仅禁用 AI 标签页并显示提示信息，标准建议和预测结果的使用不受影响。

[图5-2: UML 顺序图 — /predict 接口完整调用时序：Browser → Flask Router → Model Inference → Rule Advice Engine → Response → DOM Rendering。PlantUML 代码见 predict_sequence.puml]

[图5-3: UML 顺序图 — /advice/ai 接口完整调用时序：Browser → Flask Router → AI Advice Module → DeepSeek API → JSON Parse & Validate → Response → Tab Rendering。PlantUML 代码见 ai_advice_sequence.puml]

---
