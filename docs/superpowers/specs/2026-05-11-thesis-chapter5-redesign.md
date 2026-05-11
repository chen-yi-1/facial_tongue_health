# 毕业论文第五章重写设计文档

## 背景

今天新增的功能（生活建议规则引擎 + DeepSeek AI 建议模块 + 前端左右分栏布局 + Tab 切换等）未在原论文第五章中体现。原第五章仅描述了最基础的 Flask 原型（约 1200 字），需要大幅重写以完整展示系统设计。

## 章节结构（修订版）

### 5.1 系统总体架构设计（约 1500 字）

**内容：**
- 承上启下：从第四章实验模型过渡到工程实现
- 开发环境说明（Windows 11 / Python 3.10 / TensorFlow 2.10 / Flask 3.0）
- 分层架构：前端展示层 → Flask 应用层 → 算法推理层 → 数据存储层
- 技术选型理由：Flask 轻量零配置、原生 JS 无框架依赖、MobileNet 模型轻量化契合边缘部署
- 核心调用关系：/predict 和 /advice/ai 两条主链路概述

**配图：**
- 图5-1：UML 组件图 — Browser(HTML/CSS/JS) ↔ Flask Routes ↔ Model Inference(TensorFlow) ↔ Advice System(Rule Engine + DeepSeek API)

---

### 5.2 后端服务设计与实现（约 2500 字）

#### 5.2.1 Flask 路由与 API 设计
- 全部路由表：`GET /`、`POST /predict`、`POST /advice/ai`、`GET /uploads/<filename>`
- 请求/响应 JSON 格式示例
- 表5-1：API 接口汇总表（路由/方法/入参/返回/说明）

#### 5.2.2 模型推理模块
- 双输入模型兼容加载（Keras Model vs tf.saved_model）
- 图片预处理流水线（Resize 224×224 → 归一化 [0,1] → 扩维 batch → 双输入推理）
- 输出 Softmax 概率解析与标签映射

#### 5.2.3 生活建议规则引擎
- 数据来源：国家卫健委《中国居民膳食指南（2022）》、国家中医药管理局《中国公民中医养生保健素养》、中华中医药学会《亚健康中医临床指南》
- 3 类 × 5 维度（饮食调理、作息建议、运动指导、中医调理、心理调节）共 15 条规则
- 防御式编程：KeyError 即时报错

#### 5.2.4 AI 智能建议模块
- DeepSeek API 调用流程（POST → Authorization Bearer → Chat Completions）
- System Prompt 设计策略（角色设定 + 维度约束 + JSON 格式要求）
- 错误处理：API Key 缺失返回提示 → 网络超时 30s → JSON 解析异常兜底
- `.env` 环境变量加载机制（无 python-dotenv 依赖的轻量实现）

**配图：**
- 图5-2：UML 顺序图 — /predict 完整时序（Browser → Flask → Load Model → Preprocess(face, tongue) → Model.predict() → get_rule_advice() → JSON → Render）
- 图5-3：UML 顺序图 — /advice/ai 完整时序（Browser → Flask → DeepSeek API → JSON Parse → Validate → Tab Render）

---

### 5.3 前端界面设计与实现（约 2500 字）

#### 5.3.1 页面布局设计
- 左右分栏 CSS Grid 布局：左侧上传面板 sticky 固定，右侧结果面板独立滚动
- CSS 自定义属性设计系统（颜色/阴影/圆角/动画曲线集中管理）
- 响应式断点：≥860px 双栏 ↔ <860px 单栏纵向
- 医疗健康风格：Teal 主色调 + 渐变 + 噪点纹理 + Noto Serif/Sans SC 字体

#### 5.3.2 图片上传与预览
- 拖拽上传 + 点击上传双模式（dragenter/dragover/drop 事件）
- ObjectURL 预览机制与内存回收（revokeObjectURL）
- 表单提交前文件校验

#### 5.3.3 结果可视化
- 概率柱状图递进动画（CSS transition + setTimeout 错峰触发）
- 分类标签居中高亮展示
- 概率项依次出现的进场动画

#### 5.3.4 生活建议双 Tab 切换
- "标准建议"/"AI 建议" 双标签：规则建议随 /predict 同步渲染，AI 建议异步 fetch
- AI Tab 三种状态：加载中（旋转图标 + disabled）、加载成功（内容渲染 + enabled）、加载失败（title 提示）
- 5 维度卡片网格布局（2 列 + 最后单卡片 span 2 处理奇数）
- Tab 切换 CSS 状态管理（active 类 + hidden 类）

**配图：**
- 图5-4：UML 活动图 — 用户完整操作流程（选择照片 → 点击分析 → 进度反馈 → 结果+规则建议渲染 → 点击AI Tab → AI加载 → 成功展示/失败提示）
- 图5-5：页面布局区域划分示意图（标注左右分栏各功能区域）
- 图5-6：系统截图 — 图片上传与预览状态

---

### 5.4 系统功能展示（约 1500 字）

按三种健康判定结果各展示一组完整截图，配合简要文字说明，展示结果面板 + 建议面板的完整效果。

**配图：**
- 图5-7：系统截图 — "健康"判定 + 标准建议面板
- 图5-8：系统截图 — "亚健康"判定 + AI 建议面板
- 图5-9：系统截图 — "不健康"判定 + 建议面板
- 图5-10：系统截图 — 响应式窄屏/移动端布局

---

## 图号总表（10 张图 + 1 张表）

| 编号 | 类型 | 内容 | 位置 |
|------|------|------|------|
| 5-1 | UML 组件图 | 系统组件架构 | 5.1 |
| 5-2 | UML 顺序图 | /predict 接口调用时序 | 5.2 |
| 5-3 | UML 顺序图 | /advice/ai 接口调用时序 | 5.2 |
| 表5-1 | 表格 | API 接口汇总（路由/方法/入参/返回） | 5.2 |
| 5-4 | UML 活动图 | 用户完整操作流程（含AI分支） | 5.3 |
| 5-5 | 示意图 | 页面布局区域划分 | 5.3 |
| 5-6 | 截图 | 图片上传与预览状态 | 5.3 |
| 5-7 | 截图 | 健康判定 + 标准建议面板 | 5.4 |
| 5-8 | 截图 | 亚健康判定 + AI建议面板 | 5.4 |
| 5-9 | 截图 | 不健康判定 + 建议面板 | 5.4 |
| 5-10 | 截图 | 响应式窄屏/移动端效果 | 5.4 |

## 图示绘制说明

- UML 图建议使用 PlantUML（VS Code 插件，文本即代码，方便版本管理）或 draw.io
- 组件图的参与者：Browser、Flask Router、Model Inference、Rule Advice、DeepSeek API
- 顺序图需标注生命线、激活框、同步/异步调用区分
- 活动图需包含用户操作泳道 + 系统处理泳道，以及 AI 加载分支（成功/失败）
- 截图需在 Flask 服务运行状态下截取，确保 UI 完整、数据真实
