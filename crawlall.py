# crawlall.py
# Works with crawl4ai 0.7.x variants (handles BrowserConfig at construction time)
# Python 3.9+; run: python crawlall.py

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import json
import time
import re
from pathlib import Path
from urllib.parse import urlparse, quote_plus

# Crawl4AI imports (use names present in 0.7.x)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

ROOT = "https://giki.edu.pk"
SITEMAP_INDEX = f"{ROOT}/sitemap.xml"
OUT_DIR = Path("data/giki_md")
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = Path("giki_index.jsonl")

# Pruning filter settings
PRUNE_THRESHOLD = 0.45
PRUNE_MIN_WORDS = 6
WORD_COUNT_THRESHOLD = 8
CONCURRENCY = 6

def slugify(url: str) -> str:
    parsed = urlparse(url)
    s = (parsed.netloc + parsed.path).strip("/").lower()
    s = re.sub(r"[^a-z0-9_\-\.]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (quote_plus(s) or "root")[:180]

# --------- fetch sitemap(s) ----------
async def fetch_text(session: aiohttp.ClientSession, url: str, ssl_disable: bool = True) -> str:
    try:
        # Many Windows environments need ssl=False for some sites; set True in prod if certs are OK
        async with session.get(url, ssl=not ssl_disable, timeout=30) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                print(f"[WARN] fetch {url} -> status {resp.status}")
    except Exception as e:
        print(f"[WARN] fetch {url} -> {e}")
    return ""

async def collect_sitemaps_and_pages():
    sitemap_urls = []
    page_urls = []
    async with aiohttp.ClientSession() as session:
        text = await fetch_text(session, SITEMAP_INDEX)
        if not text:
            raise RuntimeError("Failed to fetch sitemap index")
        root = ET.fromstring(text)
        # sitemap index contains <sitemap><loc>
        sitemap_urls = [el.text.strip() for el in root.findall(".//{*}loc") if el.text]
        print(f"[info] sitemap index -> {len(sitemap_urls)} sitemap files found")

        for sm in sitemap_urls:
            txt = await fetch_text(session, sm)
            if not txt:
                continue
            try:
                sroot = ET.fromstring(txt)
                locs = [el.text.strip() for el in sroot.findall(".//{*}loc") if el.text and el.text.startswith(ROOT)]
                print(f"[info] {sm} -> {len(locs)} pages")
                page_urls.extend(locs)
            except Exception as e:
                print(f"[WARN] parse {sm} -> {e}")

    # dedupe preserving order
    seen = set()
    out = []
    for u in page_urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    print(f"[info] total unique pages collected: {len(out)}")
    return out

# --------- crawl and save ----------
async def crawl_and_save(all_urls: list[str]):
    # Build pruning filter and markdown generator (Fit Markdown)
    prune_filter = PruningContentFilter(
        threshold=PRUNE_THRESHOLD,
        threshold_type="dynamic",
        min_word_threshold=PRUNE_MIN_WORDS,
    )
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    # Crawler run config: tune as needed
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=WORD_COUNT_THRESHOLD,
        markdown_generator=md_generator,
        excluded_tags=["nav", "header", "footer", "script", "style"],
        check_robots_txt=False,  # set True to strictly obey robots.txt
    )

    # Create crawler with BrowserConfig at construction time (this avoids passing it to start())
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_cfg)

    # Start crawler (no args)
    await crawler.start()

    try:
        print(f"[start] crawling {len(all_urls)} pages (concurrency={CONCURRENCY})")
        # arun_many is a coroutine that returns results; await it
        results = await crawler.arun_many(urls=all_urls, config=run_cfg, concurrency=CONCURRENCY)

        saved = 0
        with INDEX_FILE.open("a", encoding="utf-8") as idx_f:
            for r in results:
                # r.success, r.url, r.markdown.{raw_markdown, fit_markdown, fit_html}
                if getattr(r, "success", False):
                    url = r.url
                    fit_md = getattr(r.markdown, "fit_markdown", None)
                    raw_md = getattr(r.markdown, "raw_markdown", None)

                    if fit_md and fit_md.strip():
                        fname = OUT_DIR / (slugify(url) + ".md")
                        with fname.open("w", encoding="utf-8") as f:
                            f.write(f"<!-- source: {url} -->\n")
                            f.write(f"<!-- saved_at: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n\n")
                            f.write(fit_md)

                        meta = {
                            "url": url,
                            "file": str(fname),
                            "fit_len": len(fit_md),
                            "raw_len": len(raw_md) if raw_md else None,
                        }
                        idx_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        saved += 1
                        print(f"[saved] {url} -> {fname}")
                    else:
                        print(f"[skip] empty fit_markdown: {url}")
                else:
                    err = getattr(r, "error_message", None)
                    print(f"[fail] {getattr(r, 'url', 'unknown')} -> {err}")

        print(f"[done] saved {saved} pages to {OUT_DIR}. index -> {INDEX_FILE}")
    finally:
        # ensure crawler is closed
        await crawler.close()

# --------- main ----------
async def main():
    urls = await collect_sitemaps_and_pages()
    if not urls:
        print("[err] no URLs found; exiting")
        return
    await crawl_and_save(urls)

if __name__ == "__main__":
    asyncio.run(main())
