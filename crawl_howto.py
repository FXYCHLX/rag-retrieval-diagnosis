# -*- coding: utf-8 -*-
"""补抓 Spring Boot how-to 指南（配置类实务问答都在这儿）"""
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://docs.spring.io/spring-boot/"
WORK = os.path.dirname(os.path.abspath(__file__))
PAGES = os.path.join(WORK, "corpus", "pages")
os.makedirs(PAGES, exist_ok=True)
INDEX = os.path.join(WORK, "corpus", "howto-index.html")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read()


if not os.path.exists(INDEX):
    open(INDEX, "wb").write(get(BASE + "how-to/"))
idx = open(INDEX, encoding="utf-8", errors="ignore").read()

links = set()
for h in re.findall(r'href="([^"]+)"', idx):
    if h.startswith(("http", "#", "../", "mailto")):
        continue
    if h.endswith(".html") and "/" not in h:
        links.add("how-to/" + h)
    elif h.endswith("/") and not h.startswith("_"):
        links.add("how-to/" + h)
links = sorted(links)
print("发现 how-to 页:", len(links))


def slug(url):
    s = url.rstrip("/").replace("how-to/", "").replace(".html", "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return "howto_" + (s or "index")


def fetch(url):
    target = os.path.join(PAGES, slug(url) + ".html")
    if os.path.exists(target) and os.path.getsize(target) > 2000:
        return url, os.path.getsize(target), "skip"
    try:
        data = get(BASE + url)
        open(target, "wb").write(data)
        return url, len(data), "ok"
    except Exception as e:
        return url, 0, "FAIL %s" % type(e).__name__


with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (url, size, st) in enumerate(ex.map(fetch, links), 1):
        print("%2d/%d %-6s %7d %s" % (i, len(links), st, size, url), flush=True)
print("done")
