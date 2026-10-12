# -*- coding: utf-8 -*-
"""RAG 问答原型：BM25 检索 + 提示词组装 + 大模型生成

用法：
    set RAG_API_BASE=https://xxx/v1
    set RAG_API_KEY=sk-xxx
    python rag_pipeline.py --question "How do I change the server port?"     # 单问
    python rag_pipeline.py --experiment                                       # 跑完整对比实验

两组对照：
    baseline —— 不给检索结果，直接问模型
    rag      —— 先检索 top-k 文档块，拼进提示词再问
"""
import os
import re
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

from retrieval import load_chunks, index_text, BM25

WORK = os.path.dirname(os.path.abspath(__file__))


def _load_env_file():
    """支持把 key 写进同目录 .env（每行 KEY=VALUE），这样 key 不必出现在命令行/聊天里。
    用完跑 --clean 删掉，再去控制台吊销 key。"""
    p = os.path.join(WORK, ".env")
    if not os.path.exists(p):
        return False
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return True


ENV_FILE_LOADED = _load_env_file()

API_BASE = os.environ.get("RAG_API_BASE", "").rstrip("/")
API_KEY = os.environ.get("RAG_API_KEY", "")
MODEL = os.environ.get("RAG_MODEL", "deepseek-chat")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
PROMPT_MODE = os.environ.get("RAG_PROMPT_MODE", "strict")

SYSTEM_PROMPT = (
    "You are a technical assistant for the Spring Boot framework. "
    "Answer the user's question directly and concisely. "
    "If you are not sure about the exact configuration property name, say so."
)

# strict：只准照摘录回答，摘录里没有就弃答（保守，防幻觉）
RAG_PROMPT = """Answer the question using ONLY the provided documentation excerpts.
If the excerpts do not contain the answer, reply exactly: NOT_FOUND.
Cite the property name or annotation exactly as it appears in the excerpts.

Question: {question}

Documentation excerpts:
{context}

Answer:"""

# loose：摘录优先，不够时允许用自身知识兜底并打标签（不弃答）
RAG_PROMPT_LOOSE = """Answer the question using the provided documentation excerpts as the primary source.
Cite the property name or annotation exactly as it appears in the excerpts.
If the excerpts do not contain the answer, answer from your own knowledge of Spring Boot
and begin that answer with the tag [OWN_KNOWLEDGE].

Question: {question}

Documentation excerpts:
{context}

Answer:"""


def list_models():
    """列出该 key 可用的模型名（零 token 成本，用来确认 RAG_MODEL 该填什么）"""
    if not API_BASE or not API_KEY:
        sys.exit("缺少 RAG_API_BASE / RAG_API_KEY 环境变量")
    req = urllib.request.Request(API_BASE + "/models",
                                 headers={"Authorization": "Bearer " + API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("取模型列表失败：HTTP %d" % e.code)
        return
    except Exception as e:
        print("取模型列表失败：%s" % e)
        return
    ids = [m.get("id") for m in data.get("data", [])]
    print("可用模型 %d 个：" % len(ids))
    for i in sorted(x for x in ids if x):
        print("  ", i)
    print("\n把要用的那个填给 RAG_MODEL（改 .env 或环境变量）")


USAGE = {"prompt": 0, "completion": 0, "calls": 0}


def chat(messages, temperature=0.0, retries=3):
    """OpenAI 兼容接口的最小实现（不依赖 openai 包）"""
    if not API_BASE or not API_KEY:
        raise RuntimeError("缺少 RAG_API_BASE / RAG_API_KEY 环境变量")
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 400,
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_BASE + "/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + API_KEY,
                         "User-Agent": "rag-eval/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            u = data.get("usage") or {}
            USAGE["prompt"] += u.get("prompt_tokens", 0)
            USAGE["completion"] += u.get("completion_tokens", 0)
            USAGE["calls"] += 1
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print("  重试 %d: %s" % (attempt + 1, e))
            time.sleep(3 * (attempt + 1))


# deepseek-flash 非高峰价（USD / 1M tokens）；高峰时段翻倍
PRICE_IN, PRICE_OUT, RMB = 0.15, 0.60, 7.15


def usage_line():
    p, c = USAGE["prompt"], USAGE["completion"]
    usd = p / 1e6 * PRICE_IN + c / 1e6 * PRICE_OUT
    return ("本次 %d 次调用｜输入 %d tok ｜输出 %d tok ｜ 约 $%.4f ≈ ¥%.3f（非高峰价）"
            % (USAGE["calls"], p, c, usd, usd * RMB))


class RagAssistant:
    def __init__(self, chunks_file=None, top_k=TOP_K):
        path = chunks_file or os.path.join(WORK, "chunks_final.jsonl")
        self.chunks = load_chunks(path)
        docs = [index_text(c) for c in self.chunks]
        self.retriever = BM25(docs)
        self.top_k = top_k

    def retrieve(self, question):
        return [self.chunks[i] for i, _ in self.retriever.search(question, topk=self.top_k)]

    def build_context(self, hits):
        parts = []
        for n, c in enumerate(hits, 1):
            parts.append("[%d] (%s)\n%s" % (n, c["heading"], c["text"]))
        return "\n\n".join(parts)

    def answer_baseline(self, question):
        return chat([{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": question}])

    def answer_rag(self, question):
        hits = self.retrieve(question)
        tpl = RAG_PROMPT_LOOSE if PROMPT_MODE == "loose" else RAG_PROMPT
        prompt = tpl.format(question=question, context=self.build_context(hits))
        return chat([{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}]), hits


def experiment(out_path, rag_only=False):
    questions = json.load(open(os.path.join(WORK, "questions.json"), encoding="utf-8"))
    assistant = RagAssistant()
    print("配置：top_k=%d ｜ prompt=%s ｜ rag_only=%s" % (TOP_K, PROMPT_MODE, rag_only))
    results = []
    for n, q in enumerate(questions, 1):
        print("[%2d/%d] %s" % (n, len(questions), q["q"]), flush=True)
        row = {"question": q["q"], "answer_key": q["answer_key"]}
        if not rag_only:
            try:
                base = assistant.answer_baseline(q["q"])
            except Exception as e:
                print("   baseline 失败:", e)
                base = ""
            row["baseline_answer"] = base
            row["baseline_correct"] = q["answer_key"].lower() in base.lower()
            row["baseline_notfound"] = "NOT_FOUND" in base.upper()

        try:
            rag_ans, hits = assistant.answer_rag(q["q"])
        except Exception as e:
            print("   rag 失败:", e)
            rag_ans, hits = "", []
        row["rag_answer"] = rag_ans
        row["rag_correct"] = q["answer_key"].lower() in rag_ans.lower()
        row["rag_notfound"] = "NOT_FOUND" in rag_ans.upper()
        row["rag_own_knowledge"] = "[OWN_KNOWLEDGE]" in rag_ans.upper()
        row["retrieved"] = [{"heading": h["heading"], "text": h["text"][:200]} for h in hits]
        results.append(row)

    n = len(results)
    summary = {
        "n": n,
        "top_k": TOP_K,
        "prompt_mode": PROMPT_MODE,
        "rag_correct": round(sum(r["rag_correct"] for r in results) / n, 4),
        "rag_notfound": round(sum(r["rag_notfound"] for r in results) / n, 4),
        "rag_own_knowledge": round(sum(r["rag_own_knowledge"] for r in results) / n, 4),
    }
    if not rag_only:
        summary["baseline_correct"] = round(sum(r["baseline_correct"] for r in results) / n, 4)
        summary["baseline_notfound"] = round(sum(r["baseline_notfound"] for r in results) / n, 4)
    json.dump({"summary": summary, "rows": results},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nRAG 命中率 %.2f ｜ 弃答 %.2f ｜ 用自身知识兜底 %.2f"
          % (summary["rag_correct"], summary["rag_notfound"], summary["rag_own_knowledge"]))
    if not rag_only:
        print("直答命中率 %.2f ｜ 直答弃答 %.2f"
              % (summary["baseline_correct"], summary["baseline_notfound"]))
    print("已保存:", out_path)
    print(usage_line())


def clean_key():
    """跑完删掉本地 .env（真删要去控制台吊销 key）"""
    p = os.path.join(WORK, ".env")
    if os.path.exists(p):
        os.remove(p)
        print("已删除本地 key 文件：%s" % p)
        print("⚠️ 这只删了本地副本 —— 去 DeepSeek 控制台把那个 key 点 Delete 才是真的吊销")
    else:
        print("本地没有 .env 文件（key 走了环境变量的话，关掉终端就没了）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", help="单次提问")
    ap.add_argument("--experiment", action="store_true", help="跑完整对比实验")
    ap.add_argument("--rag-only", action="store_true", help="只跑 RAG 组（变体实验省一半调用）")
    ap.add_argument("--ping", action="store_true", help="只用一句话试通接口（省额度）")
    ap.add_argument("--models", action="store_true", help="列出可用模型名（零成本）")
    ap.add_argument("--clean", action="store_true", help="删掉本地 .env 里的 key（跑完收尾用）")
    ap.add_argument("--out", default=os.path.join(WORK, "experiment_generation.json"))
    a = ap.parse_args()
    if a.clean:
        clean_key()
    elif a.models:
        list_models()
    elif a.ping:
        print("base =", API_BASE or "(未设置 RAG_API_BASE)")
        print("model =", MODEL)
        print("key 来源 =", ".env 文件" if ENV_FILE_LOADED else "环境变量")
        if not API_BASE or not API_KEY:
            sys.exit("缺少 RAG_API_BASE / RAG_API_KEY")
        t0 = time.time()
        try:
            txt = chat([{"role": "user", "content": "Reply with the single word: OK"}], retries=1)
        except urllib.error.HTTPError as e:
            code = e.code
            print("失败：HTTP %d" % code)
            if code in (401, 403):
                print("→ 401/403 是 key 本身的问题：贴错了 / 带空格换行 / 已被吊销")
            elif code == 402:
                print("→ 402 = 余额不足，去控制台充值（跑全套只要一毛钱）")
            elif code == 429:
                print("→ 429 = 限流，等一会儿再试")
            else:
                print("→ 可能是模型名不对，下面列出这个 key 能用的模型：\n")
                list_models()
            sys.exit(1)
        except Exception as e:
            sys.exit("失败：%s（先看网络能不能到 %s）" % (e, API_BASE))
        print("耗时 %.1fs｜返回：%s" % (time.time() - t0, txt))
        print("接口通了，可以跑 --experiment")
    elif a.experiment:
        experiment(a.out, rag_only=a.rag_only)
    elif a.question:
        asst = RagAssistant()
        print("=== 直答 ===\n", asst.answer_baseline(a.question))
        ans, hits = asst.answer_rag(a.question)
        print("\n=== RAG ===\n", ans)
        print("\n--- 检索命中 ---")
        for h in hits:
            print("*", h["heading"])
    else:
        ap.print_help()
