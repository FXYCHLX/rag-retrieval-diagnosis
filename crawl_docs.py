# -*- coding: utf-8 -*-
"""抓取 Spring Boot 官方文档 reference 全部页面，存为 HTML"""
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://docs.spring.io/spring-boot/"
WORK = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(WORK, "corpus")
PAGES = os.path.join(CORPUS, "pages")
os.makedirs(PAGES, exist_ok=True)

index_html = open(os.path.join(CORPUS, "springboot-index.html"), encoding="utf-8", errors="ignore").read()

links = set()
for h in re.findall(r'href="([^"]+)"', index_html):
    if h.startswith("reference/") and h.endswith((".html", "/")):
        links.add(h)
links = sorted(links)
print("发现页面:", len(links))


def slug(url):
    s = url.rstrip("/").replace("reference/", "").replace(".html", "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s or "index"


def fetch(url):
    target = os.path.join(PAGES, slug(url) + ".html")
    if os.path.exists(target) and os.path.getsize(target) > 2000:
        return url, os.path.getsize(target), "skip"
    full = BASE + url
    try:
        req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        open(target, "wb").write(data)
        return url, len(data), "ok"
    except Exception as e:
        return url, 0, "FAIL %s" % type(e).__name__


with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (url, size, st) in enumerate(ex.map(fetch, links), 1):
        print("%3d/%d %-9s %8d  %s" % (i, len(links), st, size, url), flush=True)

print("done")
