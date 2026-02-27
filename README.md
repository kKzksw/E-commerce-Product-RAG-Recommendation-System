# E-commerce Product Recommendation System

基于 **Agentic RAG + Streamlit** 的手机推荐系统。支持自然语言输入，自动识别“推荐”或“对比”意图，并结合结构化参数与评论证据给出结果。

## 功能概览

- 自然语言路由：自动识别 `RECOMMEND` / `COMPARE`
- 多源检索融合：
  - 结构化检索（预算、品牌、型号、规格偏好）
  - 语义评论检索（句向量 + 相似度）
  - 描述文本相关性评分
- 对比模式：输出胜出机型、关键差异与评分拆解
- 评论洞察：自动提取优缺点、常见投诉与情感摘要
- Streamlit 可视化页面：Top3 卡片、技术细节、证据片段

## 技术栈

- Python
- Streamlit
- Pandas
- EdenAI（文本生成 + 向量）

## 项目结构

```text
.
├── streamlit_app.py            # 主入口（Web UI）
├── requirements.txt            # 依赖
├── data/
│   ├── mobile_reviews.csv      # 原始评论数据
│   └── sentence_embeddings.pkl # 句向量缓存（首次运行可自动生成）
└── src/
    ├── agent/
    │   ├── router.py           # 查询意图与约束解析
    │   ├── compare_tool.py     # 对比评分与胜出逻辑
    │   └── explainer.py        # 结果解释生成
    ├── retriever/
    │   ├── structured.py       # 结构化检索
    │   ├── multi_source.py     # 多源融合检索
    │   ├── review_insights.py  # 评论洞察总结
    │   └── evidence.py         # 证据片段提取
    └── utils/
        ├── data.py             # 数据加载与句向量预计算
        ├── edenai.py           # EdenAI API 封装
        └── evaluation.py       # 路由/解释评估工具
```

## 快速开始

### 1. 创建并激活虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
EDENAI_API_KEY=your_edenai_api_key
```

### 4. 启动应用

```bash
streamlit run streamlit_app.py
```

启动后在浏览器打开本地地址（通常是 `http://localhost:8501`）。

## 使用示例

- 推荐：`Recommend a phone under $600 with good battery life`
- 对比：`Compare iPhone 14 vs Galaxy S24`
- 指定偏好：`I want a latest Google phone with strong camera`

## 数据说明

`data/mobile_reviews.csv` 主要字段包括：

- 商品信息：`brand`, `model`, `price_usd`
- 综合评分：`rating`
- 维度评分：`battery_life_rating`, `camera_rating`, `performance_rating`, `design_rating`, `display_rating`
- 评论信息：`review_text`, `review_date`, `source`

## 评估（可选）

运行内置评估脚本：

```bash
python -m src.utils.evaluation
```

## 常见问题

- 报错 `Missing EDENAI_API_KEY`：检查 `.env` 是否存在且 key 有效。
- 首次启动较慢：系统会预计算评论句向量并缓存到 `data/sentence_embeddings.pkl`。
- 语义检索不可用：会自动回退到关键词检索，但效果可能稍弱。

## 注意事项

- 请勿将真实 API Key 提交到 Git 仓库。
- 如果要强制重建向量缓存，可在页面侧边栏点击 `Recompute embeddings`。
