# -*- coding: utf-8 -*-
"""诊断：对每题统计含答案的块在全量排序中的最好名次，并打印该块信息"""
import json
from retrieval import load_chunks, index_text, BM25, LSA

chunks = load_chunks()
docs = [index_text(c) for c in chunks]
qs = json.load(open("questions.json", encoding="utf-8"))
bm = BM25(docs)
ls = LSA(docs)

print("%-42s %6s %6s  %s" % ("answer_key", "BM25", "LSA", "gold chunk"))
for q in qs:
    key = q["answer_key"].lower()
    golds = [i for i, c in enumerate(chunks) if key in c["text"].lower()]
    if not golds:
        print("%-42s %6s %6s  (语料里没有)" % (q["answer_key"], "-", "-"))
        continue

    def best_rank(retr):
        order = [i for i, _ in retr.search(q["q"], topk=len(chunks))]
        return min((order.index(g) + 1) for g in golds)

    m = best_rank(bm)
    l = best_rank(ls)
    g = golds[0]
    print("%-42s %6d %6d  [%s] %s | %s" % (q["answer_key"], m, l,
                                           chunks[g]["id"], chunks[g]["heading"][:45],
                                           chunks[g]["text"][:90].replace("\n", " ")))
