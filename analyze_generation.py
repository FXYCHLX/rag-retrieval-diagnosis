# -*- coding: utf-8 -*-
"""诊断：为什么 RAG 打不过直答"""
import json
import os
from retrieval import load_chunks, index_text, BM25

WORK = os.path.dirname(os.path.abspath(__file__))
chunks = load_chunks(os.path.join(WORK, "chunks_final.jsonl"))
retriever = BM25([index_text(c) for c in chunks])


def hit(chunk, key):
    k = key.lower()
    return k in chunk["text"].lower() or k in chunk["heading"].lower()


data = json.load(open(os.path.join(WORK, "experiment_generation.json"), encoding="utf-8"))
rows = data["rows"]

quad = {"both": 0, "rag_only": 0, "base_only": 0, "neither": 0}
print("%-3s %-6s %-6s %-5s %-7s  %s" % ("#", "直答", "RAG", "R@3", "RAG弃答", "问题"))
print("-" * 108)
ceil_hit = 0
for i, r in enumerate(rows, 1):
    b = "✓" if r["baseline_correct"] else "✗"
    g = "✓" if r["rag_correct"] else "✗"
    if r["baseline_correct"] and r["rag_correct"]:
        quad["both"] += 1
    elif r["rag_correct"]:
        quad["rag_only"] += 1
    elif r["baseline_correct"]:
        quad["base_only"] += 1
    else:
        quad["neither"] += 1
    # 重算 top-3 里有没有答案（RAG 的理论上限）
    idxs = [i2 for i2, _ in retriever.search(r["question"], topk=3)]
    ceil = any(hit(chunks[j], r["answer_key"]) for j in idxs)
    ceil_hit += ceil
    print("%-3d %-6s %-6s %-5s %-7s  %s" % (
        i, b, g, "有" if ceil else "无",
        "是" if r["rag_notfound"] else "",
        r["question"][:52]))

n = len(rows)
print("-" * 108)
print("四象限：都对 %d ｜ 只有 RAG 对 %d ｜ 只有直答对 %d ｜ 都错 %d" %
      (quad["both"], quad["rag_only"], quad["base_only"], quad["neither"]))
print("检索上限（Top-3 含答案）= %d/%d = %.2f  ← RAG 的理论天花板" % (ceil_hit, n, ceil_hit / n))
print("RAG 实际 = %.2f，与上限差 %.2f" % (data["summary"]["rag_correct"], ceil_hit / n - data["summary"]["rag_correct"]))
print()
print("RAG 答错但答案就在检索结果里的题（= 模型/prompt 的锅，不是检索的锅）：")
for i, r in enumerate(rows, 1):
    if r["rag_correct"]:
        continue
    idxs = [i2 for i2, _ in retriever.search(r["question"], topk=3)]
    if any(hit(chunks[j], r["answer_key"]) for j in idxs):
        print("  %2d. %s" % (i, r["question"][:70]))
        print("      RAG 答：%s" % (r["rag_answer"][:110].replace("\n", " ") or "(空)"))
