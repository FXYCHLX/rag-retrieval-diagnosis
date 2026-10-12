# -*- coding: utf-8 -*-
"""把 Spring Boot 文档 HTML 切成带标题路径的文本块（chunk）

策略：按文档顺序扫描标题(h2/h3/h4)与正文块(p/pre/ul/ol/table/div.paragraph)，
维护"当前标题路径"，把正文挂到路径上；相邻同路径小块合并；超长块按句号切并留重叠。
"""
import os
import re
import json
from bs4 import BeautifulSoup

WORK = os.path.dirname(os.path.abspath(__file__))
PAGES = os.path.join(WORK, "corpus", "pages")
OUT = os.path.join(WORK, "chunks.jsonl")

MAX_LEN = 700
MIN_LEN = 80
OVERLAP = 120
SKIP_TAGS = {"nav", "script", "style", "aside", "header", "footer", "button", "svg"}


def clean(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def collect(article):
    """按文档顺序产出 (kind, path_tuple, text)"""
    path = []
    out = []
    for el in article.descendants:
        if getattr(el, "name", None) is None:
            continue
        if el.name in SKIP_TAGS:
            continue
        # 跳过被跳过祖先包裹的元素
        if any(getattr(p, "name", None) in SKIP_TAGS for p in el.parents if p is not article):
            continue
        if el.name in ("h2", "h3", "h4"):
            # 只取标题自身的文本（避免把子元素重复计入）
            text = clean("".join(el.find_all(string=True, recursive=False)) or el.get_text(" ", strip=True))
            if not text:
                continue
            level = int(el.name[1])
            path = path[: level - 2] + [text]
            out.append(("heading", tuple(path), text))
        elif el.name in ("p", "pre", "table"):
            # div.paragraph 里的 p 已经涵盖；避免 table 内 p 重复
            if el.name == "p" and el.find_parent("table") is not None:
                continue
            text = clean(el.get_text(" ", strip=True))
            if not text or len(text) < 20:
                continue
            if el.name == "pre":
                text = "[code] " + text
            out.append((el.name, tuple(path), text))
        elif el.name in ("ul", "ol"):
            if el.find_parent(["ul", "ol"]) is not None:
                continue
            items = [clean(li.get_text(" ", strip=True)) for li in el.find_all("li", recursive=False)]
            text = clean(" ".join(i for i in items if i))
            if not text or len(text) < 20:
                continue
            out.append(("list", tuple(path), text))
    return out


def merge(raw):
    out = []
    buf = None
    for kind, path, text in raw:
        if kind == "heading":
            continue
        if buf and buf["path"] == path and len(buf["text"]) + len(text) + 1 <= MAX_LEN:
            buf["text"] = (buf["text"] + " " + text).strip()
        else:
            if buf:
                out.append(buf)
            buf = {"path": path, "text": text}
    if buf:
        out.append(buf)

    final = []
    for c in out:
        t = c["text"]
        if len(t) <= MAX_LEN:
            if len(t) >= MIN_LEN:
                final.append(c)
            continue
        start = 0
        while start < len(t):
            end = min(start + MAX_LEN, len(t))
            if end < len(t):
                cut = max(t.rfind("。", start, end), t.rfind(". ", start, end))
                if cut > start + MAX_LEN // 2:
                    end = cut + 1
            piece = t[start:end].strip()
            if len(piece) >= MIN_LEN:
                final.append({"path": c["path"], "text": piece})
            if end >= len(t):
                break
            start = max(start + 1, end - OVERLAP)
    return final


def main():
    files = sorted(f for f in os.listdir(PAGES) if f.endswith(".html"))
    all_chunks = []
    for f in files:
        raw = open(os.path.join(PAGES, f), encoding="utf-8", errors="ignore").read()
        soup = BeautifulSoup(raw, "lxml")
        art = soup.find("article") or soup.find("main") or soup.body
        if art is None:
            continue
        h1 = art.find("h1")
        page_title = clean(h1.get_text(" ", strip=True)) if h1 else f[:-5]
        for c in merge(collect(art)):
            path = list(c["path"]) or [page_title]
            if path[0] != page_title:
                path = [page_title] + path
            all_chunks.append({
                "page": f[:-5],
                "title": page_title,
                "heading": " > ".join(path),
                "text": c["text"],
            })

    with open(OUT, "w", encoding="utf-8") as fh:
        for i, c in enumerate(all_chunks):
            c["id"] = "c%05d" % i
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    n = len(all_chunks)
    avg = sum(len(c["text"]) for c in all_chunks) / max(n, 1)
    print("页面数: %d  切块数: %d  平均长度: %.0f 字符" % (len(files), n, avg))
    for c in all_chunks[:2] + all_chunks[n // 2: n // 2 + 2] + all_chunks[-2:]:
        print("- [%s] %s\n    %s" % (c["id"], c["heading"][:80], c["text"][:120]))


if __name__ == "__main__":
    main()
