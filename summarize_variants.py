# -*- coding: utf-8 -*-
"""汇总四组实验，按「检索命中 / 未命中」分层，看清 RAG 的收益到底从哪来"""
import json
import os
from retrieval import load_chunks, index_text, BM25

WORK = os.path.dirname(os.path.abspath(__file__))
chunks = load_chunks(os.path.join(WORK, "chunks_final.jsonl"))
retriever = BM25([index_text(c) for c in chunks])


def hit(c, key):
    k = key.lower()
    return k in c["text"].lower() or k in c["heading"].lower()


FILES = [
    ("experiment_generation.json", "原始 RAG（top3+严格）"),
    ("experiment_v_topk5.json", "V1（top5+严格）"),
    ("experiment_v_loose.json", "V2（top3+放宽）"),
    ("experiment_v_combo.json", "V3（top5+放宽）"),
]

base = json.load(open(os.path.join(WORK, "experiment_generation.json"), encoding="utf-8"))
b_ok = sum(r["baseline_correct"] for r in base["rows"])
print("直答（baseline）命中 %d/21 = %.3f\n" % (b_ok, b_ok / 21))

print("%-22s %-6s %-7s %-14s %-14s %-7s %-7s" %
      ("配置", "检索上限", "总体", "检索命中组", "检索未命中组", "弃答", "兜底"))
print("-" * 96)
detail = {}
out = {"baseline_correct": b_ok, "n": len(base["rows"]), "variants": []}
for fn, name in FILES:
    d = json.load(open(os.path.join(WORK, fn), encoding="utf-8"))
    k = d["summary"].get("top_k", 3)  # 旧版文件没这个字段，默认 3
    rows = d["rows"]
    n = len(rows)
    ceil = 0
    hg = [0, 0]
    mg = [0, 0]
    for r in rows:
        idxs = [i for i, _ in retriever.search(r["question"], topk=k)]
        ok = any(hit(chunks[j], r["answer_key"]) for j in idxs)
        ceil += ok
        if ok:
            hg[1] += 1
            hg[0] += bool(r["rag_correct"])
        else:
            mg[1] += 1
            mg[0] += bool(r["rag_correct"])
    detail[name] = (d, hg, mg, k)
    out["variants"].append({
        "name": name, "file": fn, "top_k": k,
        "prompt_mode": d["summary"].get("prompt_mode", "strict"),
        "ceiling": round(ceil / n, 4),
        "overall": d["summary"]["rag_correct"],
        "hit_ok": hg[0], "hit_n": hg[1],
        "miss_ok": mg[0], "miss_n": mg[1],
        "notfound": d["summary"].get("rag_notfound", 0),
        "own_knowledge": d["summary"].get("rag_own_knowledge", 0),
    })
    print("%-22s %-6s %-7s %-14s %-14s %-7s %-7s" % (
        name,
        "%.2f" % (ceil / n),
        "%.3f" % d["summary"]["rag_correct"],
        "%d/%d=%.0f%%" % (hg[0], hg[1], 100 * hg[0] / hg[1]) if hg[1] else "-",
        "%d/%d=%.0f%%" % (mg[0], mg[1], 100 * mg[0] / mg[1]) if mg[1] else "-",
        "%.2f" % d["summary"].get("rag_notfound", 0),
        "%.2f" % d["summary"].get("rag_own_knowledge", 0)))

print()
print("=== V2 拆解：0.81 里有多少是靠「退化成直答」拿的 ===")
d, hg, mg, k = detail["V2（top3+放宽）"]
own = [r for r in d["rows"] if r["rag_own_knowledge"]]
no_own = [r for r in d["rows"] if not r["rag_own_knowledge"]]
print("  兜底作答 %d 题，答对 %d 题（%.0f%%）—— 这部分本质是直答，不算 RAG 的功劳"
      % (len(own), sum(r["rag_correct"] for r in own), 100 * sum(r["rag_correct"] for r in own) / len(own)))
print("  基于摘录作答 %d 题，答对 %d 题（%.0f%%）"
      % (len(no_own), sum(r["rag_correct"] for r in no_own), 100 * sum(r["rag_correct"] for r in no_own) / len(no_own)))

print()
print("=== 逐题对照（✓=对）===")
hdr = "%-3s %-5s" % ("#", "直答")
for fn, name in FILES:
    hdr += " %-5s" % name.split("（")[0]
print(hdr)
for i in range(21):
    line = "%-3d %-5s" % (i + 1, "✓" if base["rows"][i]["baseline_correct"] else "✗")
    for fn, name in FILES:
        dd = json.load(open(os.path.join(WORK, fn), encoding="utf-8"))
        line += " %-5s" % ("✓" if dd["rows"][i]["rag_correct"] else "✗")
    print(line)

# ---------- 落盘，供报告脚本读取 ----------
# 直答在相同检索条件下的分层表现（做对照基线）
out["baseline_layers"] = {}
for k in (3, 5):
    hg = [0, 0]
    mg = [0, 0]
    for r in base["rows"]:
        idxs = [i for i, _ in retriever.search(r["question"], topk=k)]
        ok = any(hit(chunks[j], r["answer_key"]) for j in idxs)
        if ok:
            hg[1] += 1
            hg[0] += bool(r["baseline_correct"])
        else:
            mg[1] += 1
            mg[0] += bool(r["baseline_correct"])
    out["baseline_layers"]["top%d" % k] = {
        "hit_ok": hg[0], "hit_n": hg[1], "miss_ok": mg[0], "miss_n": mg[1]}

d2 = detail["V2（top3+放宽）"][0]
own_rows = [r for r in d2["rows"] if r["rag_own_knowledge"]]
no_own_rows = [r for r in d2["rows"] if not r["rag_own_knowledge"]]
out["v2_own"] = {"n": len(own_rows), "ok": sum(r["rag_correct"] for r in own_rows)}
out["v2_no_own"] = {"n": len(no_own_rows), "ok": sum(r["rag_correct"] for r in no_own_rows)}

per_q = []
gens = [json.load(open(os.path.join(WORK, fn), encoding="utf-8")) for fn, _ in FILES]
for i in range(len(base["rows"])):
    per_q.append({
        "question": base["rows"][i]["question"],
        "answer_key": base["rows"][i]["answer_key"],
        "baseline": bool(base["rows"][i]["baseline_correct"]),
        "rag": [bool(g["rows"][i]["rag_correct"]) for g in gens],
    })
out["per_question"] = per_q

json.dump(out, open(os.path.join(WORK, "analysis_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n已写出 analysis_summary.json（%d 组配置）" % len(out["variants"]))
