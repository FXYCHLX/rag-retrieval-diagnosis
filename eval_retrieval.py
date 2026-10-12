# -*- coding: utf-8 -*-
"""检索实验：BM25 vs TF-IDF+LSA，算 Recall@1/3/5 与 MRR

判定标准：top-k 中是否有任意一块的原文包含该题的答案关键词（answer_key）。
同时输出每题每方案的首位命中块，便于报告里做案例分析。
"""
import os
import json
import time
from retrieval import load_chunks, index_text, BM25, LSA, Hybrid

WORK = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(WORK, "questions.json")
RESULT = os.path.join(WORK, "experiment_retrieval.json")

K_LIST = [1, 3, 5]


def hit(chunk, key):
    """块级判定：检索到的块正文或其标题路径包含标准答案字符串"""
    k = key.lower()
    return k in chunk["text"].lower() or k in chunk["heading"].lower()


def gold_sections(chunks, key):
    """标准答案所在的小节集合（页面+标题路径），与块大小无关"""
    k = key.lower()
    return {(c["page"], c["heading"]) for c in chunks
            if k in c["text"].lower() or k in c["heading"].lower()}


def section_hit(chunk, gold):
    return (chunk["page"], chunk["heading"]) in gold


def run():
    chunks_file = os.environ.get("CHUNKS_FILE", os.path.join(WORK, "chunks_final.jsonl"))
    chunks = load_chunks(chunks_file)
    print("切片文件:", os.path.basename(chunks_file))
    docs = [index_text(c) for c in chunks]
    questions = json.load(open(QUESTIONS, encoding="utf-8"))
    print("语料块数: %d｜问题数: %d" % (len(chunks), len(questions)))

    results = {}
    for name, build in [("BM25", lambda: BM25(docs)),
                        ("TFIDF+LSA", lambda: LSA(docs)),
                        ("Hybrid+Rerank", lambda: Hybrid(chunks, docs))]:
        t0 = time.time()
        retriever = build()
        build_sec = time.time() - t0
        rows = []
        for q in questions:
            idx_score = retriever.search(q["q"], topk=max(K_LIST))
            gold = gold_sections(chunks, q["answer_key"])
            gold_ids = {c["id"] for c in chunks
                        if (c["page"], c["heading"]) in gold}
            hits = [hit(chunks[i], q["answer_key"]) for i, _ in idx_score]
            sec_hits = [section_hit(chunks[i], gold) for i, _ in idx_score]
            rank = next((r + 1 for r, h in enumerate(hits) if h), None)
            sec_rank = next((r + 1 for r, h in enumerate(sec_hits) if h), None)
            rows.append({
                "question": q["q"],
                "answer_key": q["answer_key"],
                "gold_sections": sorted("%s | %s" % s for s in gold),
                "gold_ids": sorted(gold_ids),
                "section_rank": sec_rank,
                "top1": {
                    "heading": chunks[idx_score[0][0]]["heading"],
                    "score": round(idx_score[0][1], 4),
                    "text": chunks[idx_score[0][0]]["text"][:300],
                },
                "rank": rank,
                "top_ids": [chunks[i]["id"] for i, _ in idx_score],
            })
        rec = {}
        for k in K_LIST:
            rec["recall@%d" % k] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / len(rows), 4)
        mrr = sum((1.0 / r["rank"]) for r in rows if r["rank"]) / len(rows)
        rec["mrr"] = round(mrr, 4)
        # 小节级指标（不受块大小影响）
        for k in K_LIST:
            rec["sec_recall@%d" % k] = round(
                sum(1 for r in rows if r["section_rank"] and r["section_rank"] <= k) / len(rows), 4)
        rec["sec_mrr"] = round(sum((1.0 / r["section_rank"]) for r in rows if r["section_rank"]) / len(rows), 4)
        rec["build_seconds"] = round(build_sec, 2)
        results[name] = {"metrics": rec, "rows": rows}
        print("%-14s 块级 R@1=%.2f @3=%.2f @5=%.2f MRR=%.3f ｜小节级 R@1=%.2f @3=%.2f @5=%.2f MRR=%.3f ｜建索引 %.1fs"
              % (name, rec["recall@1"], rec["recall@3"], rec["recall@5"], rec["mrr"],
                 rec["sec_recall@1"], rec["sec_recall@3"], rec["sec_recall@5"], rec["sec_mrr"], build_sec))

    json.dump(results, open(RESULT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已保存:", RESULT)

    # 逐题对照
    print("\n题号  BM25  LSA   混合+重排   答案关键词")
    for i, q in enumerate(questions, 1):
        rb = results["BM25"]["rows"][i - 1]
        rl = results["TFIDF+LSA"]["rows"][i - 1]
        rh = results["Hybrid+Rerank"]["rows"][i - 1]
        print("%2d   %-5s %-5s %-8s %s" % (i, rb["rank"] or "-", rl["rank"] or "-",
                                           rh["rank"] or "-", q["answer_key"]))


if __name__ == "__main__":
    run()
