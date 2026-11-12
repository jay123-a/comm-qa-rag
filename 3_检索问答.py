# -*- coding: utf-8 -*-
"""
3_检索问答.py —— RAG 完整流程：检索(R) + 增强(A) + 生成(G) 串起来

三步对应图里那三个环节：
    R 检索：问句转向量 → 向量库找 top-3 相似片段
    A 增强：把片段拼进提示词，当作"参考资料"喂给大模型
    G 生成：调 DeepSeek API，让它"根据参考资料回答"

运行：在 D:\rag 目录下执行
    D:/dsH/dl-env/Scripts/python.exe 3_检索问答.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import QWEN_API_KEY, MODEL_NAME


def load_model():
    """加载 embedding 模型：优先本地缓存，失败再联网下载"""
    try:
        return SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
    except Exception:
        return SentenceTransformer("BAAI/bge-small-zh-v1.5")


def build_prompt(question: str, docs: list) -> str:
    """
    A — 增强（Augmented）：把检索到的资料拼进提示词。

    这是 RAG 里"Prompt 工程"的核心，一个好提示词要包含：
      1. 角色设定：你是谁
      2. 行为约束：只依据资料、不许编造（防幻觉）
      3. 资料注入：检索到的原文
      4. 明确问题
    """
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


def main():
    # 先检查 key 有没有填
    if not QWEN_API_KEY or QWEN_API_KEY.startswith("sk-在这里填"):
        print("请先在 config.py 里填你的通义千问 API Key（去 bailian.console.aliyun.com 申请）。")
        return

    # ── 1. 加载向量模型 + 知识库（和前面脚本一样）─────────────
    model = load_model()
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("comm_kb")

    # ── 2. 初始化通义千问客户端（OpenAI 兼容接口）──────────────
    #    换 DeepSeek/智谱/本地 vLLM，只需要改 base_url 和 model 名，代码其余不变
    llm = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # ── 3. 循环问答 ──────────────────────────────────────────
    print("RAG 问答已就绪（输入问题开始，直接回车退出）")
    while True:
        question = input("\n请输入问题：").strip()
        if not question:
            break

        # R：检索 top-3
        qv = model.encode([question], normalize_embeddings=True)
        result = collection.query(query_embeddings=qv.tolist(), n_results=3)
        docs = result["documents"][0]

        # A：拼提示词
        prompt = build_prompt(question, docs)

        # G：调大模型生成
        resp = llm.chat.completions.create(
            model=MODEL_NAME,   # 默认 qwen-flash，可在 config.py 改 qwen-plus/qwen-max
            messages=[
                {"role": "system", "content": "你是一名通信领域专业助教。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,   # 温度越低越"照资料说话"，越高越爱自由发挥
        )
        answer = resp.choices[0].message.content

        print("\n" + "=" * 50)
        print("【回答】")
        print(answer)
        print("\n【本次检索到的资料】")
        for i, d in enumerate(docs, start=1):
            print(f"  [{i}] {d.split(chr(10))[0]}")


if __name__ == "__main__":
    main()
