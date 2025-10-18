"""
Reddit r/giki RAG Data Pipeline:
- Scrapes older posts & comments
- Embeds with OpenAI (text-embedding-3-small)
- Uploads to Pinecone
- Skips already indexed posts
---------------------------------------------
Dependencies:
    pip install requests tqdm openai langchain-openai pinecone-client python-dotenv
"""

import os
import time
import json
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# -------------------- SETUP --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SUBREDDIT = "giki"
POSTS_PER_PAGE = 100  # max Reddit allows
MAX_PAGES = 40        # 10 pages ≈ 1000 posts
SLEEP_BETWEEN = 1.5   # seconds between calls

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GIKI-RAG-Agent/1.0)"}

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)


# -------------------- SCRAPING --------------------
def fetch_posts(subreddit, after=None):
    """Fetch a page of posts from Reddit JSON API."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={POSTS_PER_PAGE}"
    if after:
        url += f"&after={after}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"⚠️ Skipping page (status={resp.status_code})")
        return [], None

    data = resp.json()["data"]
    posts = []
    for item in data["children"]:
        post = item["data"]
        posts.append({
            "id": f"r_{post['id']}",
            "post_id": post["id"],
            "title": post.get("title", ""),
            "selftext": post.get("selftext", ""),
            "url": "https://www.reddit.com" + post.get("permalink", ""),
            "created_utc": post.get("created_utc", 0)
        })
    return posts, data.get("after")


def fetch_comments(post_id):
    """Fetch top-level comments for a Reddit post."""
    url = f"https://www.reddit.com/comments/{post_id}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
        comments_data = data[1]["data"]["children"]
        comments = [
            c["data"].get("body", "")
            for c in comments_data
            if c["kind"] == "t1" and "body" in c["data"]
        ]
        return comments
    except Exception:
        return []


def fetch_existing_ids():
    """Retrieve all IDs already in Pinecone to skip duplicates."""
    existing = set()
    print("🧩 Checking existing vectors in Pinecone...")
    try:
        cursor = None
        while True:
            res = index.list(prefix="", limit=1000, cursor=cursor)
            for v in res.vectors:
                existing.add(v.id)
            if not res.pagination or not res.pagination.get("next"):
                break
            cursor = res.pagination["next"]
    except Exception as e:
        print("⚠️ Error fetching existing IDs:", e)
    print(f"✅ Found {len(existing)} existing vectors.")
    return existing


# -------------------- PIPELINE --------------------
def build_dataset():
    print(f"📡 Scraping r/{SUBREDDIT} posts and comments ...")
    all_posts = []
    after = None
    for page in range(MAX_PAGES):
        posts, after = fetch_posts(SUBREDDIT, after)
        if not posts:
            break
        all_posts.extend(posts)
        print(f"📄 Page {page+1}: {len(posts)} posts fetched")
        time.sleep(SLEEP_BETWEEN)
        if not after:
            break

    dataset = []
    for post in tqdm(all_posts, desc="💬 Fetching comments"):
        comments = fetch_comments(post["post_id"])
        combined_text = (
            f"Title: {post['title']}\n\nBody: {post['selftext']}\n\nComments:\n" +
            "\n".join(comments)
        ).strip()
        dataset.append({
            "id": post["id"],
            "text_preview": combined_text,
            "url": post["url"],
            "source": "r/giki",
            "created_utc": post["created_utc"]
        })
        time.sleep(SLEEP_BETWEEN)

    os.makedirs("data", exist_ok=True)
    with open("data/r_giki_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Total {len(dataset)} posts saved to data/r_giki_dataset.json")
    return dataset


def upload_to_pinecone(dataset, existing_ids):
    print("🚀 Embedding and uploading new items to Pinecone ...")
    new_data = [d for d in dataset if d["id"] not in existing_ids]
    print(f"🆕 New posts to upload: {len(new_data)}")

    batch_size = 32
    for i in tqdm(range(0, len(new_data), batch_size)):
        batch = new_data[i : i + batch_size]
        texts = [item["text_preview"] for item in batch]
        ids = [item["id"] for item in batch]
        metas = [
            {
                "text_preview": item["text_preview"][:1000],
                "source": item["source"],
                "url": item["url"],
                "created_utc": item["created_utc"]
            }
            for item in batch
        ]
        try:
            embeddings = embedder.embed_documents(texts)
            vectors = [
                {"id": ids[j], "values": embeddings[j], "metadata": metas[j]}
                for j in range(len(batch))
            ]
            index.upsert(vectors=vectors)
        except Exception as e:
            print("⚠️ Error uploading batch:", e)
        time.sleep(SLEEP_BETWEEN)
    print("✅ Upload complete.")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    dataset = build_dataset()
    existing_ids = fetch_existing_ids()
    upload_to_pinecone(dataset, existing_ids)
    print("🎯 All Reddit r/giki data (posts + comments) now added to your Pinecone RAG.")
