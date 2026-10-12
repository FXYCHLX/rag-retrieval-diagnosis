# RAG 检索诊断 · Spring Boot 文档问答

**一个用来回答「RAG 到底哪里不行」的量化实验，不是又一个 RAG demo。**

在 **1281 个文档块 / 21 道真实技术问题** 上，把检索与生成两个环节拆开逐层测，
用可复现的数字定位瓶颈，而不是凭感觉调参。

```
语料：Spring Boot 官方文档 109 个页面　→　1281 个文本块
问题：21 道真实配置类问题（每题有唯一的标准答案标识符）
成本：跑完全套生成实验约 0.1 元（106 次 API 调用）
```

---

## 结论先行：四个可能反直觉的发现

### 1️⃣ 朴素 RAG 输给了「不给检索结果」的直答

| 配置 | 答案命中率 |
|---|---|
| 直答（不给检索结果） | **0.762**（16/21） |
| 原始 RAG（top3 + 严格提示词） | 0.667（14/21） |

**加了检索，反而更差。**

原因不复杂：严格提示词要求「摘录里没有就回答 `NOT_FOUND`」，而 top3 检索会漏掉答案，
模型被约束住，只能弃答——**弃答率 23.8%**。

> 这解释了一个常见现象：很多人给系统加了 RAG，体感变差了，却以为是「模型不行」。

### 2️⃣ 瓶颈在召回，不在生成

把结果按「检索有没有命中正确答案块」切开看：

| 情况 | 题数 | RAG 答对 |
|---|---|---|
| 检索命中（hit） | 17 | **17 / 17（100%）** |
| 检索未命中（miss） | 4 | **0 / 4（0%）** |

**只要块找对了，模型一定能答对；找错了，再好的提示词也救不回来。**

→ 优化重点应该全部压在检索上。这也正是本项目把大半篇幅用在检索评测的原因。

### 3️⃣ 加向量、加混合、加重排，全都打不过朴素 BM25

| 检索方案 | Recall@1 | Recall@3 | Recall@5 | MRR | 建索引耗时 |
|---|---|---|---|---|---|
| **BM25（稀疏）** | **0.476** | **0.714** | **0.810** | **0.577** | **~0.1 s** |
| TF-IDF + LSA（稠密） | 0.381 | 0.571 | 0.667 | 0.484 | ~3.7 s |
| Hybrid + Rerank（混合） | 0.476 | 0.571 | 0.762 | 0.559 | ~1.8 s |

**最朴素的那个赢了，而且快一个数量级。**

> 建索引耗时为单机实测，单次运行存在波动；但 BM25 与稠密方案之间是量级差距，不受波动影响。

在术语密集的**技术文档**场景里（`server.port`、`@ConditionalOnProperty` 这类标识符），
TF-IDF/LSA 这类语义向量会把「长得很像但意思不同」的配置项混在一起，
而 BM25 的词项精确匹配恰好是优势。

> 这不是说向量检索没用，而是说：**别默认「越复杂越好」，先在你自己语料上量一遍。**

### 4️⃣ 文本块不是越小越好，也不是越大越好

用同一套 BM25 检索器，只改切片粒度：

| 目标块长（字符） | 实际块数 | 平均块长 | Recall@5 | 最优 MRR |
|---|---|---|---|---|
| 400 | 4338 | 294 | 0.524 | 0.401 |
| 700 | 2513 | 486 | 0.619 | 0.486 |
| 900 | 1967 | 609 | 0.714 | 0.573 |
| **1200** | **1564** | **766** | **0.762** | **0.520** |
| 1600 | — | — | 0.667 | 0.514 |

**峰值在 ~1200 字符，两头都掉。**

- 块太小（400）：一个配置项的说明被切散，「问 A 答 B」；
- 块太大（1600）：无关内容稀释了关键词密度，打分被拉平。

### 5️⃣ 怎么调参才能超过直答

| 变体 | top_k | 提示词 | 命中率 | 弃答率 |
|---|---|---|---|---|
| V1 | **5** | 严格 | **0.810** | 0.190 |
| V2 | 3 | 放宽（允许 `[OWN_KNOWLEDGE]` 兜底） | **0.810** | 0.000 |
| V3 | 5 | 放宽 | 0.714 | 0.000 |

两条都能到 0.810（超过直答的 0.762），但**代价不同**：

- **V1**：把 top_k 从 3 提到 5，召回率自然上去，弃答率从 23.8% 降到 19.0%。**没有引入新风险。**
- **V2**：放宽提示词让模型用自身知识兜底，弃答归零，但 **23.8% 的答案来自模型自身知识**——
  在技术文档场景这等于**主动引入幻觉**，正确性不再可归因于检索。

> V2 的 5 道「兜底答对」的题里，5 道全对。数字好看，但它已经不是 RAG 了。

---

## 快速开始

**只想验证检索部分（零成本、除抓取外不联网）：**

```bash
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 语料已随仓库提供（chunks_final.jsonl），可直接跑评测
python eval_retrieval.py      # BM25 / LSA / Hybrid 三方案对比
python sweep_granularity.py   # 切片粒度影响
python grid_search.py         # BM25 参数网格搜索
```

**要跑生成对比（需要大模型 API，约 0.1 元）：**

```bash
copy .env.example .env        # 填入你自己的 key
python rag_pipeline.py --ping              # 先试通接口（省额度）
python rag_pipeline.py --experiment        # 直答 vs RAG 全量对比
python rag_pipeline.py --clean             # 跑完删掉本地 key
```

任何 **OpenAI 兼容接口**都能用，只改 `.env` 里的 `RAG_API_BASE` 和 `RAG_MODEL`，代码不用动。

---

## 完整复现流程

每一步的产出都是下一步的输入。

| # | 命令 | 产出 | 需要联网 / API |
|---|---|---|---|
| ① | `python crawl_docs.py` → `python crawl_howto.py` | `corpus/pages/*.html`（109 页） | 联网，约 2 分钟 |
| ② | `python build_chunks.py` | `chunks_final.jsonl`（1281 块） | 否 |
| ③ | `python eval_retrieval.py` | `experiment_retrieval.json` | 否 |
| ④ | `python sweep_granularity.py` | `experiment_granularity.json` | 否 |
| ⑤ | `python grid_search.py` | `experiment_grid.json` | 否 |
| ⑥ | `python rag_pipeline.py --experiment` | `experiment_generation.json` | **要 API** |
| ⑦ | 改 `RAG_TOP_K` / `RAG_PROMPT_MODE` 后 `--rag-only` | 各变体 json | **要 API** |
| ⑧ | `python summarize_variants.py` | `analysis_summary.json` | 否 |
| ⑨ | `python make_figures.py` | `figures/*.png` | 否 |
| ⑩ | `python diag_recall.py` | 逐题召回诊断 | 否 |

> 仓库里已经带了 ③–⑧ 的原始输出（`experiment_*.json`、`analysis_summary.json`），
> **不改代码就能核对本文所有数字**。

---

## 目录结构

```
retrieval.py             三种检索器：BM25 / TF-IDF+LSA / Hybrid+Rerank
build_chunks.py          切片：按标题层级 + 块长上限 + 重叠
eval_retrieval.py        检索评测：块级 + 小节级 Recall@k / MRR
sweep_granularity.py     切片粒度扫描
grid_search.py           BM25 参数网格搜索
diag_recall.py           逐题召回诊断（哪道题为什么没召回）
rag_pipeline.py          RAG 管线：检索 → 提示词组装 → 生成
analyze_generation.py    生成结果分析
summarize_variants.py    变体汇总 → analysis_summary.json
crawl_docs.py            抓取官方文档（reference 目录）
crawl_howto.py           抓取官方文档（how-to 目录）
questions.json           21 道题 + 每题的标准答案标识符
chunks_final.jsonl       切片结果（1281 块）— 检索实验直接读它
experiment_*.json        各实验原始输出（逐题记录，可核对）
analysis_summary.json    汇总统计，本文所有表格的数字来源
requirements.txt         依赖清单
.env.example             API 配置模板
```

---

## 关键实现位置

| 关注点 | 位置 |
|---|---|
| BM25 稀疏检索打分 | `retrieval.py` → `BM25` |
| TF-IDF + LSA 稠密检索 | `retrieval.py` → `LSA` |
| 混合检索 + 重排序 | `retrieval.py` → `Hybrid` |
| 切片（标题层级 + 块长上限 + 重叠） | `build_chunks.py` |
| 评价指标（块级 / 小节级 Recall@k、MRR） | `eval_retrieval.py` |
| 两版提示词（严格 / 放宽） | `rag_pipeline.py` → `RAG_PROMPT` / `RAG_PROMPT_LOOSE` |
| 答案命中判定（标识符字符串匹配） | `summarize_variants.py` |
| 环境变量加载与 key 清理 | `rag_pipeline.py` → `_load_env_file()` / `clean_key()` |

---

## 已知限制（重要）

1. **结论只在「结构化技术文档 + 标识符类问题」这个场景内成立。**
   换到开放域问答、口语化提问、或长文摘要任务，BM25 大概率不再占优——**请在你自己的语料上重测**。
2. **样本量小（21 题）。** 足够看出量级差异（0.52 vs 0.81），但不足以支撑精细排序；
   V1 与 V2 同为 0.810 时，谁更好不该由这个样本断定。
3. **判定方式是「标准答案字符串是否出现在回答里」**，属于**字符串匹配**而非语义判定，
   可能低估改写型正确、高估字面命中。
4. **稠密检索用的是 TF-IDF + LSA（SVD），不是神经嵌入模型。**
   换成 bge / text-embedding 这类真正的语义嵌入，结论可能不同——这也是下一步该补的实验。
5. **未做重排模型的对比。** 当前的 `Hybrid` 里的 rerank 是基于词项覆盖的启发式，不是 cross-encoder。

---

## 两个实际踩过的坑

1. **文档站结构变了就抓不全。**
   抓取脚本依赖 Antora 多页结构（`reference/` + `how-to/` 两个目录）。
   Spring Boot 官方文档改版后 URL 规律会变，需要重新确认，否则静默少抓。
2. **旧模型名会下线。**
   某些模型名报 400/404 时，先跑 `python rag_pipeline.py --models` 看当前可用列表再改配置。

---

## 数据来源与许可

- **语料**：Spring Boot 官方文档（<https://docs.spring.io/spring-boot/>），
  由 `crawl_docs.py` / `crawl_howto.py` 抓取。Spring Boot 采用 **Apache License 2.0**，
  本仓库中的切片数据（`chunks_final.jsonl`）为其派生物，同样遵循该许可。
- **代码**：MIT，见 [LICENSE](LICENSE)。
- **跑生成实验需要你自己的 API key，请勿把 key 提交到仓库**（`.gitignore` 已忽略 `.env`）。
