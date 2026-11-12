# -*- coding: utf-8 -*-
"""
2_检索测试.py —— RAG 第二步（前半）：只做"检索"，先验证知识库建对了没有

特点：不调大模型 API，纯本地，零成本。跑通后你能直观看到：
    输入一个问题 → 输出知识库里最相关的几条 QA（按相似度排序）

运行：在 D:\rag 目录下执行
    D:/dsH/dl-env/Scripts/python.exe 2_检索测试.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer


def main():
    # ── 1. 加载 embedding 模型 ─────────────────────────────────
    #    重点：必须和建库时用同一个模型，否则"翻译口径"不一致，
    #    问句向量和库里的向量就不在同一套坐标里，检索会全乱套
    #    优先读本地缓存（稳定秒开），没有缓存时自动联网下载
    try:
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
    except Exception:
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # ── 2. 打开已建好的知识库 ─────────────────────────────────
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("comm_kb")
    print(f"知识库已打开，共 {collection.count()} 条")

    # ── 3. 循环提问，实时检索 ─────────────────────────────────
    print("（提示：试试问 5G 编码 / 香农公式 / OFDM 循环前缀 这些通信题）")
    while True:
        query = input("\n请输入问题（直接回车退出）：").strip()
        if not query:
            break

        # 问句也要向量化（和建库时一样的处理）
        qv = model.encode([query], normalize_embeddings=True)

        # 核心一行：把问句向量丢给向量库，让它返回最相似的 n 条
        result = collection.query(
            query_embeddings=qv.tolist(),
            n_results=3,          # 取 top-3
        )

        # result 里有三个东西：
        #   documents  -> 命中的原文（问题+答案）
        #   distances  -> 余弦距离，越小 = 越相似
        #   metadatas  -> 我们存的 question/answer 字段
        docs = result["documents"][0]
        dists = result["distances"][0]
        print(f"\n最相关的 {len(docs)} 条资料：")
        for j, (doc, dist) in enumerate(zip(docs, dists), start=1):
            print(f"\n—— 第 {j} 名（余弦距离 {dist:.4f}）——")
            print(doc)


if __name__ == "__main__":
    main()
