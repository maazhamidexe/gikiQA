import asyncio
import requests
import os
import re
from xml.etree import ElementTree
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

SAVE_DIR = "naan"

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def sanitize_filename(url: str) -> str:
    """
    Create a safe filename from a URL.
    Example: https://docs.pydantic.dev/latest/usage/models/ 
    -> latest_usage_models.md
    """
    # Remove schema and replace special chars with underscores
    filename = re.sub(r"https?://", "", url)
    filename = re.sub(r"[^a-zA-Z0-9-_./]", "_", filename)
    filename = filename.strip("/").replace("/", "_")
    if not filename.endswith(".md"):
        filename += ".md"
    return filename

async def crawl_sequential(urls: List[str]):
    print("\n=== Sequential Crawling with Session Reuse ===")

    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        session_id = "session1"
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                print(f"✅ Successfully crawled: {url}")
                markdown_text = result.markdown.raw_markdown

                # Save markdown to file
                filename = sanitize_filename(url)
                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(markdown_text)

                print(f"   → Saved as {filepath} (length {len(markdown_text)})")
            else:
                print(f"❌ Failed: {url} - Error: {result.error_message}")
    finally:
        await crawler.close()

def get_sitemap_urls():
    sitemap_url = "https://srpsolutions.pk/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    urls = get_sitemap_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_sequential(urls)
    else:
        print("No URLs found to crawl")

if __name__ == "__main__":
    asyncio.run(main())
