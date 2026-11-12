# -*- coding: utf-8 -*-
"""
1_构建知识库.py —— RAG 第一步：把知识变成向量，存进向量库

本脚本做的事（对应 R-A-G 里的 R 的前半段准备工作）：
    读 500 条通信 QA → 每条拼成一段文本 → 用 BGE 模型转成向量 → 存进 ChromaDB

运行：在 D:\rag 目录下执行
    D:/dsH/dl-env/Scripts/python.exe 1_构建知识库.py
"""
import os
# 镜像：加速 HuggingFace 模型下载（相当于"保险丝"，本地有模型就不走网络）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import chromadb
from sentence_transformers import SentenceTransformer


def load_qa(path: str):
    """读入 JSON 数据集，返回 list[dict]"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ── 1. 读数据 ──────────────────────────────────────────────
    qa_list = load_qa("data/comm_qa_500.json")
    print(f"读入 {len(qa_list)} 条 QA")

    # ── 2. 加载 embedding 模型（把文字变成向量的"翻译器"）─────────
    #    bge-small-zh：中文向量模型，约 24MB，CPU 就能跑，无需 API key
    #    优先读本地缓存（稳定秒开），没有缓存时自动联网下载（走镜像）
    try:
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
    except Exception:
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # ── 3. 每条 QA 拼成一段文本 ─────────────────────────────────
    #    把"问题 + 答案"放一起存，好处：
    #      · 用户问的是问题，向量能匹配到"问题"部分
    #      · 答案也一起存进库里，检索命中后直接取出来拼 prompt
    texts, metadatas, ids = [], [], []
    for i, qa in enumerate(qa_list):
        q = qa["instruction"].strip()
        a = qa["output"].strip()
        texts.append(f"问题：{q}\n答案：{a}")
        metadatas.append({"question": q, "answer": a})
        ids.append(f"qa_{i}")

    # ── 4. 向量化（全脚本唯一稍慢的一步，500 条 CPU 约几十秒）────
    #    normalize=True：把向量长度归一化成 1，这样后面用余弦相似度算距离才准
    print("正在向量化 500 条数据，请稍候...")
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # ── 5. 存进 ChromaDB（向量库）───────────────────────────────
    #    PersistentClient：数据落盘到 chroma_db/ 目录，关掉程序也不丢
    client = chromadb.PersistentClient(path="chroma_db")
    #    get_or_create_collection：有就复用，没有就新建（幂等，重复跑不报错）
    collection = client.get_or_create_collection(
        name="comm_kb",
        metadata={"hnsw:space": "cosine"},  # 距离用"余弦相似度"
    )
    #    add 一次写入：ids=每条的唯一编号，embeddings=向量，documents=原文，metadatas=附加字段
    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"构建完成：知识库共 {collection.count()} 条")


if __name__ == "__main__":
    main()
