# -*- coding: utf-8 -*-
"""混合检索权重参数敏感性分析（alpha / 标题权重）

输出 experiment_grid.json，报告里作为"参数选择"一节的数据来源。
"""
import os
import json
from retrieval import load_chunks, index_text, Hybrid

WORK = os.path.dirname(os.path.abspath(__file__))
chunks = load_chunks()
docs = [index_text(c) for c in chunks]
questions = json.load(open(os.path.join(WORK, "questions.json"), encoding="utf-8"))


def evaluate(alpha, w_head, w_cover=0.25, w_hybrid=0.65):
    if abs(w_hybrid + w_cover + w_head - 1.0) > 1e-9:
        s = w_hybrid + w_cover + w_head
        w_hybrid, w_cover, w_head = w_hybrid / s, w_cover / s, w_head / s
    r = Hybrid(chunks, docs, alpha=alpha, w_hybrid=w_hybrid, w_cover=w_cover, w_head=w_head)
    ranks = []
    for q in questions:
        k = q["answer_key"].lower()
        got = r.search(q["q"], topk=5)
        rank = None
        for pos, (i, _) in enumerate(got, 1):
            c = chunks[i]
            if k in c["text"].lower() or k in c["heading"].lower():
                rank = pos
                break
        ranks.append(rank)
    n = len(ranks)
    return {
        "recall@1": round(sum(1 for x in ranks if x == 1) / n, 4),
        "recall@3": round(sum(1 for x in ranks if x and x <= 3) / n, 4),
        "recall@5": round(sum(1 for x in ranks if x and x <= 5) / n, 4),
        "mrr": round(sum(1.0 / x for x in ranks if x) / n, 4),
    }


if __name__ == "__main__":
    rows = []
    best = None
    for alpha in [0.2, 0.4, 0.5, 0.6, 0.8]:
        for w_head in [0.1, 0.2, 0.3]:
            m = evaluate(alpha, w_head)
            rows.append({"alpha": alpha, "w_head": w_head, **m})
            if best is None or m["mrr"] > best[0]:
                best = (m["mrr"], alpha, w_head, m)
            print("alpha=%.1f  w_head=%.1f  R@1=%.2f R@3=%.2f R@5=%.2f MRR=%.3f"
                  % (alpha, w_head, m["recall@1"], m["recall@3"], m["recall@5"], m["mrr"]))
    print("\n最佳: alpha=%.1f w_head=%.1f MRR=%.3f  R@5=%.2f"
          % (best[1], best[2], best[0], best[3]["recall@5"]))
    json.dump({"grid": rows,
               "best": {"alpha": best[1], "w_head": best[2], **best[3]}},
              open(os.path.join(WORK, "experiment_grid.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已保存 experiment_grid.json")
