# -*- coding: utf-8 -*-
"""两套检索器：BM25（稀疏）与 TF-IDF+LSA（稠密代理），纯本地无依赖外部 API"""
import os
import re
import json
import math

WORK = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(WORK, "chunks.jsonl")

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-@]*")

# 英文停用词（技术文档问答里这些词只起稀释作用）
STOP = set("""a an the is are was were be been being do does did doing have has had having
i you he she it we they me my your his her its our their this that these those
how what which who whom when where why can could shall should will would may might must
of in on at to for from with without by as into about over under between
and or but if then than so such not no nor only just also very more most other some any
use used using want need get got make made set setting""".split())


def tokenize(text, drop_stop=True):
    """英文技术文档：小写化，保留 . - @ 这类出现在配置项里的符号；去掉虚词"""
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    if not drop_stop:
        return toks
    return [t for t in toks if t not in STOP]


def load_chunks(path=CHUNKS):
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))
    return chunks


def index_text(c):
    """标题路径重复三次，提升标题词权重（标题是强语义信号）"""
    h = c["heading"]
    return ("%s %s %s %s" % (h, h, h, c["text"])).strip()


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokens]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf = []
        self.df = {}
        for toks in self.tokens:
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.tokens)

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    @staticmethod
    def term_weight(term):
        """配置项 / 注解 / 连字符词是强信号，加权"""
        return 2.0 if ("." in term or "@" in term or "-" in term) else 1.0

    def search(self, query, topk=10):
        q = tokenize(query)
        scores = [0.0] * self.N
        for term in set(q):
            if term not in self.df:
                continue
            idf = self.idf(term) * self.term_weight(term)
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        order = sorted(range(self.N), key=lambda i: -scores[i])[:topk]
        return [(i, scores[i]) for i in order]


class LSA:
    """TF-IDF + 截断 SVD 降维 + 余弦相似度（稠密检索的本地代理）"""

    def __init__(self, docs, n_components=256):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer

        self.np = np
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                                   token_pattern=None, sublinear_tf=True,
                                   min_df=2, max_features=60000)
        X = self.vec.fit_transform(docs)
        n_components = min(n_components, min(X.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        Z = self.svd.fit_transform(X)
        self.Z = Normalizer().fit_transform(Z)

    def search(self, query, topk=10):
        q = self.vec.transform([query])
        zq = self.svd.transform(q)
        zq = zq / (self.np.linalg.norm(zq) + 1e-9)
        sims = self.Z @ zq[0]
        idx = self.np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in idx]


class Hybrid:
    """BM25 + 稠密分数加权融合，再用词覆盖度做二阶段重排序"""

    def __init__(self, chunks, docs, alpha=0.5, rerank_pool=30,
                 w_hybrid=0.65, w_cover=0.25, w_head=0.10):
        self.chunks = chunks
        self.bm25 = BM25(docs)
        self.lsa = LSA(docs)
        self.alpha = alpha          # BM25 权重（1-alpha 给稠密）
        self.pool = rerank_pool
        self.w_hybrid = w_hybrid
        self.w_cover = w_cover
        self.w_head = w_head
        self.terms = [set(tokenize(t)) for t in docs]
        self.head_terms = [set(tokenize(c["heading"])) for c in chunks]

    @staticmethod
    def _norm(pairs):
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return {i: 1.0 for i, _ in pairs}
        return {i: (s - lo) / (hi - lo) for i, s in pairs}

    def search(self, query, topk=10):
        nb = self._norm(self.bm25.search(query, topk=self.pool))
        nl = self._norm(self.lsa.search(query, topk=self.pool))
        pool = set(nb) | set(nl)
        qterms = set(tokenize(query))
        scored = []
        for i in pool:
            base = self.alpha * nb.get(i, 0.0) + (1 - self.alpha) * nl.get(i, 0.0)
            cover = len(qterms & self.terms[i]) / max(len(qterms), 1)
            head = len(qterms & self.head_terms[i]) / max(len(qterms), 1)
            s = self.w_hybrid * base + self.w_cover * cover + self.w_head * head
            scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:topk]
