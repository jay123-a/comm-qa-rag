# 通信知识库 RAG 问答系统

结合通信专业背景，用 RAG（检索增强生成）构建的通信知识问答系统。

与微调项目 [comm-qa-lora](https://github.com/jay123-a/comm-qa-lora) 形成互补：**训练记不住的知识，用检索来兜底**。

## 项目亮点

- **RAG 全流程跑通**：检索（ChromaDB 向量检索）→ 增强（拼提示词）→ 生成（通义千问），三步闭环。
- **解决微调痛点**：香农公式这类精确知识，0.5B 微调模型记不住，RAG 直接检索出来，一次答对 `C = B·log₂(1+SNR)`。
- **工程化封装**：FastAPI 接口 + 网页对话前端，不只是命令行 demo。
- **本地向量模型**：BGE 中文向量模型（bge-small-zh，约 24MB），CPU 可跑，向量化零成本。

## 技术栈

| 部件 | 技术 |
|---|---|
| 向量库 | ChromaDB |
| 向量化 | BGE（bge-small-zh-v1.5） |
| 大模型 | 通义千问 qwen-flash（OpenAI 兼容接口） |
| 后端 | FastAPI + uvicorn |
| 前端 | 原生 HTML / CSS / JS |

## 项目结构

```
.
├── app.py                # FastAPI 服务（后端接口）
├── static/index.html     # 前端对话页
├── 1_构建知识库.py        # 500 条 QA 向量化入 ChromaDB
├── 2_检索测试.py          # 纯检索验证（不调大模型）
├── 3_检索问答.py          # 命令行版完整问答
├── config.py             # API key（已 gitignore，需自己填）
└── data/comm_qa_500.json # 500 条通信 QA 数据
```

## 快速开始

```bash
# 1. 装依赖
pip install chromadb sentence-transformers fastapi uvicorn openai

# 2. 填 API key：复制 config.example 思路，在 config.py 里填你的通义千问 key
#    （去 https://bailian.console.aliyun.com 申请）

# 3. 构建知识库（首次运行，约 1 分钟）
python 1_构建知识库.py

# 4. 启动服务
python app.py

# 5. 浏览器打开 http://127.0.0.1:8000 开始提问
```

## 接口一览

| 接口 | 说明 |
|---|---|
| `GET /` | 前端对话页 |
| `GET /health` | 健康检查 |
| `POST /chat` | 传 `{"question": "..."}`，返回 `{"answer": "...", "sources": [...]}` |

## RAG 原理（一句话）

让大模型"开卷考试"：先检索知识库找最相关的片段（R），拼进提示词（A），再让大模型生成（G）。

## 与微调项目的对比

| | LoRA 微调（comm-qa-lora） | RAG（本项目） |
|---|---|---|
| 知识放哪 | 改模型参数（背进脑子） | 外部知识库（翻参考书） |
| 要不要训练 | 要 | 不要，零训练 |
| 精确公式 | 0.5B 记不住香农公式 | 检索出来，100% 准 |
| 知识更新 | 重新训练 | 直接加资料 |

## 局限与展望

- 当前知识库只含 500 条通信 QA，后续可扩展多格式文档（PDF/Word）解析入库。
- 检索目前是纯向量检索，可升级混合检索（向量 + BM25）+ 重排（rerank）提升效果。
- 可加 Agent 能力（工具调用、多轮任务），进一步贴合生产场景。
