# -*- coding: utf-8 -*-
"""生成报告配图：技术脉络图 / 系统架构图 / 实验图表"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

WORK = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(WORK), "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5

BLUE = "#1f4e79"
LBLUE = "#dce6f1"
ORANGE = "#c55a11"
LORANGE = "#fbe4d5"
GREEN = "#375623"
LGREEN = "#e2efda"


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, bold=False, radius=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=%s" % radius,
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1a1a1a", weight="bold" if bold else "normal", linespacing=1.6)


def arrow(ax, p1, p2, color="#404040", style="-|>", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color,
                                 connectionstyle="arc3,rad=%s" % rad))


def fig_pipeline():
    """图1 技术脉络图：朴素 RAG → 高级 RAG → 模块化 RAG"""
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

    stages = [
        (0.25, "朴素 RAG\n(2020)", "向量检索 top-k\n直接拼接生成", LBLUE, BLUE),
        (3.55, "高级 RAG\n(2022-2023)", "混合检索 + 重排序\n查询改写 / 多路召回", LGREEN, GREEN),
        (6.85, "模块化 RAG\n(2024-)", "路由 / 记忆 / 知识图谱\n可编排、可评测", LORANGE, ORANGE),
    ]
    for x, title, desc, fc, ec in stages:
        box(ax, x, 2.6, 2.9, 1.5, title, fc, ec, fs=12, bold=True)
        box(ax, x, 1.35, 2.9, 1.05, desc, "#ffffff", ec, fs=9.5)

    arrow(ax, (3.15, 3.35), (3.55, 3.35), color=BLUE, lw=1.6)
    arrow(ax, (6.45, 3.35), (6.85, 3.35), color=GREEN, lw=1.6)

    ax.text(0.25, 4.65, "RAG 技术演进脉络", fontsize=13.5, weight="bold", color=BLUE)
    ax.text(0.25, 4.2, "检索增强生成从「检索—拼接」走向可编排的模块化系统", fontsize=9.5, color="#555555")

    # 底部：共同瓶颈
    box(ax, 0.25, 0.25, 9.5, 0.85,
        "共同瓶颈：切片粒度 × 检索精度 × 生成对证据的忠实度  —— 也是本文实验的三个着力点",
        "#f2f2f2", "#909090", fs=10)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig1_技术脉络图.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(p)


def fig_arch():
    """图2 系统架构图"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(0.2, 6.25, "系统总体架构", fontsize=13.5, weight="bold", color=BLUE)

    # 离线索引
    ax.add_patch(Rectangle((0.2, 3.75), 9.6, 2.15, facecolor="#fafafa",
                           edgecolor="#bfbfbf", linestyle="--", linewidth=1))
    ax.text(0.45, 5.62, "离线阶段：语料构建与索引", fontsize=10, color="#555555", weight="bold")
    off = ["官方文档\n109 个页面", "正文抽取\n与清洗", "按标题层级\n切片 1281 块", "BM25 / TF-IDF\n倒排索引"]
    for i, t in enumerate(off):
        box(ax, 0.5 + i * 2.35, 4.05, 2.0, 1.15, t, LBLUE, BLUE, fs=9.5)
        if i < 3:
            arrow(ax, (2.5 + i * 2.35, 4.62), (2.85 + i * 2.35, 4.62))

    # 在线问答
    ax.add_patch(Rectangle((0.2, 0.55), 9.6, 2.85, facecolor="#fafafa",
                           edgecolor="#bfbfbf", linestyle="--", linewidth=1))
    ax.text(0.45, 3.12, "在线阶段：问答流程", fontsize=10, color="#555555", weight="bold")
    online = [("用户提问", LORANGE, ORANGE), ("检索器\ntop-k 召回", LORANGE, ORANGE),
              ("提示词组装\n问题 + 证据", LORANGE, ORANGE), ("大模型生成\n带出处回答", LORANGE, ORANGE)]
    for i, (t, fc, ec) in enumerate(online):
        box(ax, 0.5 + i * 2.35, 1.75, 2.0, 1.0, t, fc, ec, fs=9.5)
        if i < 3:
            arrow(ax, (2.5 + i * 2.35, 2.25), (2.85 + i * 2.35, 2.25))
    arrow(ax, (6.5, 3.75), (6.5, 2.78), color="#808080", style="-|>", lw=1.2)
    ax.text(6.65, 3.2, "证据注入", fontsize=8.5, color="#666666")

    box(ax, 0.5, 0.75, 8.85, 0.72,
        "评估：块级 / 小节级 Recall@k、MRR ｜ 生成质量：答案命中率、弃答率、案例分析",
        LGREEN, GREEN, fs=9.5)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig2_系统架构图.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(p)


def fig_retrieval():
    """图3 三种检索方案对比"""
    data = json.load(open(os.path.join(WORK, "experiment_retrieval.json"), encoding="utf-8"))
    methods = ["BM25", "TFIDF+LSA", "Hybrid+Rerank"]
    labels = ["Recall@1", "Recall@3", "Recall@5", "MRR"]
    keys = ["recall@1", "recall@3", "recall@5", "mrr"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    colors = [BLUE, GREEN, ORANGE]

    ax = axes[0]
    x = range(len(labels)); w = 0.26
    for j, m in enumerate(methods):
        vals = [data[m]["metrics"][k] for k in keys]
        ax.bar([i + (j - 1) * w for i in x], vals, w, label=m, color=colors[j])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.015, "%.2f" % v, ha="center", fontsize=7.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("块级指标", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("三种检索方案的检索质量", fontsize=10.5)

    ax = axes[1]
    keys2 = ["sec_recall@1", "sec_recall@3", "sec_recall@5", "sec_mrr"]
    for j, m in enumerate(methods):
        vals = [data[m]["metrics"][k] for k in keys2]
        ax.bar([i + (j - 1) * w for i in x], vals, w, label=m, color=colors[j])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.015, "%.2f" % v, ha="center", fontsize=7.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("小节级指标", fontsize=9.5)
    ax.set_title("剔除切片长度影响后的指标", fontsize=10.5)

    fig.tight_layout()
    p = os.path.join(FIGS, "fig3_检索方案对比.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(p)


def fig_granularity():
    """图4 切片粒度对检索效果的影响"""
    data = json.load(open(os.path.join(WORK, "experiment_granularity.json"), encoding="utf-8"))
    order = ["294字", "391字", "486字", "609字", "766字", "931字"]
    xs = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for m, color, marker in [("BM25", BLUE, "o"), ("TFIDF+LSA", GREEN, "s"),
                             ("Hybrid+Rerank", ORANGE, "^")]:
        ys = [data[o][m]["recall@5"] for o in order]
        ax.plot(xs, ys, marker=marker, color=color, label=m, linewidth=1.6)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, "%.2f" % y, ha="center", fontsize=7.5, color=color)
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=9)
    ax.set_xlabel("切片平均长度", fontsize=9.5)
    ax.set_ylabel("Recall@5", fontsize=9.5)
    ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("切片粒度对检索召回的影响", fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig4_切片粒度影响.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(p)


def fig_generation():
    """图5 生成质量对比：各配置总体命中率 + 按检索是否命中分层"""
    gen_p = os.path.join(WORK, "experiment_generation.json")
    ana_p = os.path.join(WORK, "analysis_summary.json")
    if not (os.path.exists(gen_p) and os.path.exists(ana_p)):
        print("跳过图5：还没跑生成实验")
        return
    ANA = json.load(open(ana_p, encoding="utf-8"))
    V = {v["file"]: v for v in ANA["variants"]}
    n = ANA["n"]
    b_rate = ANA["baseline_correct"] / n

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))

    # 左图：各配置总体命中率
    ax = axes[0]
    names = ["直接提问", "朴素 RAG\ntop-3 严格", "V1\ntop-5 严格", "V2\ntop-3 放宽", "V3\ntop-5 放宽"]
    vals = [b_rate,
            V["experiment_generation.json"]["overall"],
            V["experiment_v_topk5.json"]["overall"],
            V["experiment_v_loose.json"]["overall"],
            V["experiment_v_combo.json"]["overall"]]
    cols = ["#a6a6a6", GREEN, BLUE, ORANGE, "#7030a0"]
    ax.bar(range(len(vals)), vals, 0.58, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.035, "%.2f" % v, ha="center", fontsize=8.5, color="#1a1a1a")
    ax.axhline(b_rate, ls="--", lw=1, color="#888888")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.2)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("答案命中率", fontsize=9.5)
    ax.set_title("各配置的总体命中率", fontsize=10.5)

    # 右图：按检索是否命中分层（朴素 RAG 对 top-3）
    ax = axes[1]
    bl = ANA["baseline_layers"]["top3"]
    o = V["experiment_generation.json"]
    base_v = [bl["hit_ok"] / bl["hit_n"], bl["miss_ok"] / bl["miss_n"]]
    rag_v = [o["hit_ok"] / o["hit_n"], o["miss_ok"] / o["miss_n"]]
    x = range(2); w = 0.32
    ax.bar([i - w / 2 for i in x], base_v, w, label="直接提问", color="#a6a6a6")
    ax.bar([i + w / 2 for i in x], rag_v, w, label="朴素 RAG", color=GREEN)
    for i, (bv, rv) in enumerate(zip(base_v, rag_v)):
        ax.text(i - w / 2, bv + 0.025, "%.0f%%" % (bv * 100), ha="center", fontsize=8.5)
        if rv < 0.03:
            ax.text(i + w / 2, 0.03, "0%\n全部落空", ha="center", fontsize=7.5, color="#c00000")
        else:
            ax.text(i + w / 2, rv + 0.025, "%.0f%%" % (rv * 100), ha="center", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["检索命中\n(%d 题)" % bl["hit_n"], "检索未命中\n(%d 题)" % bl["miss_n"]],
                       fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("答案命中率", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center")
    ax.set_title("按检索是否命中分层", fontsize=10.5)

    fig.tight_layout()
    p = os.path.join(FIGS, "fig5_生成质量对比.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(p)


if __name__ == "__main__":
    fig_pipeline()
    fig_arch()
    fig_retrieval()
    fig_granularity()
    fig_generation()
