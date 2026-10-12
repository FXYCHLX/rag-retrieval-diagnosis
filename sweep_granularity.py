# -*- coding: utf-8 -*-
"""切片粒度对比实验：三种块长 × 三种检索方案"""
import os
import json
from retrieval import load_chunks, index_text, BM25, LSA, Hybrid

WORK = os.path.dirname(os.path.abspath(__file__))
questions = json.load(open(os.path.join(WORK, "questions.json"), encoding="utf-8"))
VARIANTS = [("chunks_400.jsonl", "294字"),
            ("chunks_550.jsonl", "391字"),
            ("chunks_700.jsonl", "486字"),
            ("chunks_900.jsonl", "609字"),
            ("chunks_1200.jsonl", "766字"),
            ("chunks_1600.jsonl", "931字")]


def metrics(retr, chunks):
    ranks = []
    for q in questions:
        k = q["answer_key"].lower()
        rank = None
        for pos, (i, _) in enumerate(retr.search(q["q"], topk=5), 1):
            c = chunks[i]
            if k in c["text"].lower() or k in c["heading"].lower():
                rank = pos
                break
        ranks.append(rank)
    n = len(ranks)
    return {"recall@1": round(sum(1 for x in ranks if x == 1) / n, 4),
            "recall@3": round(sum(1 for x in ranks if x and x <= 3) / n, 4),
            "recall@5": round(sum(1 for x in ranks if x and x <= 5) / n, 4),
            "mrr": round(sum(1.0 / x for x in ranks if x) / n, 4)}


if __name__ == "__main__":
    out = {}
    print("%-14s %-12s %6s %6s %6s %6s" % ("切片", "方案", "R@1", "R@3", "R@5", "MRR"))
    for fname, label in VARIANTS:
        path = os.path.join(WORK, fname)
        if not os.path.exists(path):
            continue
        chunks = load_chunks(path)
        docs = [index_text(c) for c in chunks]
        for name, build in [("BM25", lambda: BM25(docs)),
                            ("TFIDF+LSA", lambda: LSA(docs)),
                            ("Hybrid+Rerank", lambda: Hybrid(chunks, docs, alpha=0.4, w_head=0.1))]:
            m = metrics(build(), chunks)
            out.setdefault(label, {"n_chunks": len(chunks), "avg_len":
                                   round(sum(len(c["text"]) for c in chunks) / len(chunks))})[name] = m
            print("%-14s %-12s %6.2f %6.2f %6.2f %6.3f"
                  % (label, name, m["recall@1"], m["recall@3"], m["recall@5"], m["mrr"]))
    json.dump(out, open(os.path.join(WORK, "experiment_granularity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已保存 experiment_granularity.json")
