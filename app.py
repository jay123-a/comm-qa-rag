# -*- coding: utf-8 -*-
"""
app.py —— RAG 问答的 FastAPI 服务

把命令行脚本 3_检索问答.py 封装成 HTTP 接口，让前端网页能通过网络调用。

接口一览：
  GET  /         → 返回前端对话页面
  GET  /health   → 健康检查（看服务是否活着、知识库多少条）
  POST /chat     → 传 {"question": "..."}，返回 {"answer": "...", "sources": [...]}

启动（在 D:\rag 目录下执行）：
  D:/dsH/dl-env/Scripts/python.exe -m uvicorn app:app --reload --port 8000

启动后浏览器打开 http://127.0.0.1:8000 即可看到对话页。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config import QWEN_API_KEY, MODEL_NAME


def load_model():
    """加载 embedding 模型：优先本地缓存，失败再联网下载"""
    try:
        return SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
    except Exception:
        return SentenceTransformer("BAAI/bge-small-zh-v1.5")


# ── 启动时加载一次（重要：不要每次请求都重新加载，否则又慢又费显存）──
embed_model = load_model()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("comm_kb")
llm = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

app = FastAPI(title="通信知识库 RAG 问答")


class ChatRequest(BaseModel):
    """POST /chat 的请求体格式"""
    question: str


def build_prompt(question: str, docs: list) -> str:
    """A — 增强：把检索到的资料拼进提示词（和命令行版一致）"""
    ref = "\n\n".join(f"【资料{i}】\n{d}" for i, d in enumerate(docs, start=1))
    return (
        "你是一名通信领域的专业助教。请根据下面给出的【参考资料】回答用户问题。\n"
        "要求：\n"
        "1. 只依据参考资料回答，资料中没有的内容，诚实地说「资料中未提及」，不要编造。\n"
        "2. 回答准确、简洁、专业。\n\n"
        f"{ref}\n\n"
        f"用户问题：{question}\n"
        "请回答："
    )


@app.get("/")
def index():
    """返回前端对话页面"""
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    """健康检查：返回服务状态和知识库规模"""
    return {"status": "ok", "knowledge_base_size": collection.count()}


@app.post("/chat")
def chat(req: ChatRequest):
    """核心接口：接收问题，跑完整 RAG，返回回答 + 检索来源"""
    question = req.question.strip()
    if not question:
        return {"answer": "问题不能为空", "sources": []}

    # R：检索 top-3
    qv = embed_model.encode([question], normalize_embeddings=True)
    result = collection.query(query_embeddings=qv.tolist(), n_results=3)
    docs = result["documents"][0]

    # A：拼提示词
    prompt = build_prompt(question, docs)

    # G：调大模型生成
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    # 来源只取"问题"部分（每段的第一行），方便前端展示
    sources = [d.split("\n")[0] for d in docs]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
