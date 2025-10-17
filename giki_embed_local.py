#!/usr/bin/env python3
"""
build_vectorstore_optimized.py

Optimized pipeline:
- reads giki_index.jsonl (from your Crawl4AI output)
- cleans markdown, prints a cleaned preview for each file
- chunks text semantically
- batches chunks to Ollama for embeddings
- deduplicates using Pinecone similarity check
- normalizes and upserts vectors to Pinecone in batches
"""

import json
import time
import requests
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()  # load variables from .env


# Read from env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "giki-index")


# ---------- CONFIG ----------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "mxbai-embed-large"

INDEX_FILE = "giki_index.jsonl"
MD_GLOB = "data/giki_md/*.md"


# batching & chunking params
CHUNK_SIZE = 900            # characters, approximate (langchain splits by characters unless token fn used)
CHUNK_OVERLAP = 150
BATCH_EMBED = 32            # number of chunks per Ollama request
UPSERT_BATCH = 100          # number of vectors per Pinecone upsert
DUPLICATE_SIM_THRESHOLD = 0.985  # cosine similarity threshold to consider duplicate
PRINT_PREVIEW_CHARS = 800

# ---------- Dependencies check ----------
from pinecone import Pinecone, ServerlessSpec

# ---------- HELPERS ----------
def clean_markdown(md_text: str) -> str:
    """Cleans markdown produced by Crawl4AI:
       - removes HTML comments, navigation lists, repeated link lists
       - strips URLs and dedups lines
       - uses BeautifulSoup for any residual HTML
    """
    # remove HTML comments
    txt = md_text
    txt = txt.replace("\r\n", "\n")
    txt = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    # drop HTML comments
    import re
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.DOTALL)

    # remove repeated nav-like bullet blocks commonly present in crawl output
    txt = re.sub(r"(\* ?\[[^\]]+\]\([^)]+\)\s*){3,}", "", txt)

    # remove inline markdown links but keep anchor text
    txt = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", txt)

    # remove plaintext urls
    txt = re.sub(r"https?://\S+", "", txt)

    # strip markdown bullets left with nothing useful
    txt = re.sub(r"^\s*[\*\-\+]\s*", "", txt, flags=re.MULTILINE)

    # remove excessive blank lines
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    # use BeautifulSoup to remove stray tags if any HTML lingers
    soup = BeautifulSoup(txt, "html.parser")
    text = soup.get_text("\n")
    # remove duplicate adjacent lines
    lines = [ln.strip() for ln in text.splitlines()]
    # drop empty lines at ends and dedupe repeated lines
    cleaned_lines = []
    last = None
    for ln in lines:
        if ln == last:
            continue
        if ln.strip() == "":
            if not cleaned_lines or cleaned_lines[-1] == "":
                last = ln
                continue
        cleaned_lines.append(ln)
        last = ln
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned

def extract_source_url(md_text: str):
    """Extracts the source url from the Crawl4AI frontmatter comment if present"""
    import re
    m = re.search(r"<!--\s*source:\s*([^\s]+)\s*-->", md_text)
    if m:
        return m.group(1).strip()
    return None

def slugify_from_url(url: str) -> str:
    if not url:
        return "unknown"
    p = urlparse(url)
    safe = (p.netloc + p.path).strip("/").lower().replace("/", "-")
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in safe)
    return safe[:200]

# cosine similarity
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# batch embedding via Ollama
def get_embeddings_batch(chunks: list[str]) -> list[list[float]]:
    """Send a batch of strings to Ollama /api/embed and return list of embeddings."""
    # Ollama accepts "input": [str, str, ...]
    payload = {"model": OLLAMA_MODEL, "input": chunks}
    resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # response might be {"embeddings":[...]} or {"embedding": [...]} for single
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]
    if isinstance(data, dict) and "embedding" in data:
        # single result
        return [data["embedding"]]
    # sometimes Ollama returns {"embedding": [...]} for lists under a different key
    if isinstance(data, list):
        # if it's a list of embeddings
        return data
    raise RuntimeError(f"Unexpected Ollama response shape: {data}")

def normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()

def init_pinecone():
    if not PINECONE_API_KEY:
        raise RuntimeError("Set PINECONE_API_KEY in your .env file")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[pinecone] Creating index '{PINECONE_INDEX_NAME}' (dim=1024, cosine metric)")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # change if needed
                region=PINECONE_ENVIRONMENT or "us-east-1"
            )
        )

    # Connect to the index
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"[pinecone] Connected to index '{PINECONE_INDEX_NAME}'")
    return index

# ---------- Main pipeline ----------
def main():
    print("Starting optimized pipeline...")
    # gather md files via index file (preserves the Crawl4AI order)
    entries = []
    with open(INDEX_FILE, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                entries.append(json.loads(ln))

    print(f"Found {len(entries)} index entries in {INDEX_FILE}")

    # read and clean files, print preview
    file_chunks = []   # tuples: (chunk_text, metadata)
    exact_text_set = set()  # for exact deduplication (hashes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?"]
    )

    for entry in entries:
        md_path = entry.get("file").replace("\\", "/")
        if not os.path.exists(md_path):
            print(f"[warn] file missing: {md_path}")
            continue
        with open(md_path, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = clean_markdown(raw)

        # print cleaned preview so you can inspect quality
        print("\n" + "="*80)
        print(f"FILE: {md_path}")
        print(f"SOURCE URL: {extract_source_url(raw) or entry.get('url')}")
        print(f"--- cleaned preview (first {PRINT_PREVIEW_CHARS} chars) ---")
        print(cleaned[:PRINT_PREVIEW_CHARS])
        print("="*80 + "\n")

        # chunk
        chunks = splitter.split_text(cleaned)
        for i, ch in enumerate(chunks):
            text_norm = ch.strip()
            if len(text_norm) < 50:
                continue
            # simple exact dedupe
            h = hash(text_norm)
            if h in exact_text_set:
                continue
            exact_text_set.add(h)
            meta = {
                "source_url": extract_source_url(raw) or entry.get("url"),
                "file": md_path,
                "chunk_index": i
            }
            file_chunks.append((text_norm, meta))

    print(f"Total semantic chunks to embed after cleaning & exact dedupe: {len(file_chunks)}")

    # Init Pinecone
    idx = init_pinecone()

    # Embedding + upsert loop (batched)
    all_vectors_batch = []
    upsert_count = 0
    total_skipped_duplicates = 0

    # We'll process chunks in batches of BATCH_EMBED
    for batch_start in tqdm(range(0, len(file_chunks), BATCH_EMBED), desc="Embedding batches"):
        batch = file_chunks[batch_start: batch_start + BATCH_EMBED]
        texts = [x[0] for x in batch]
        metas = [x[1] for x in batch]

        # get embeddings from Ollama for the whole batch
        try:
            embeddings = get_embeddings_batch(texts)
        except Exception as e:
            print(f"[error] Ollama embedding failed: {e}")
            # fallback: skip this batch
            continue

        # normalize embeddings and check approximate duplicates against Pinecone
        for i, emb in enumerate(embeddings):
            vec = normalize(emb)
            meta = metas[i]
            chunk_text = texts[i]
            # Query Pinecone to detect near-duplicates (top_k=1)
            try:
                # If index empty, query returns empty matches
                query_resp = idx.query(vector=vec, top_k=1, include_values=False, include_metadata=True)
                matches = query_resp.get("matches", []) or query_resp.get("results", [])
            except Exception as e:
                # In some client versions shape differs; try .query returning dict
                try:
                    query_resp = idx.query(vector=vec, top_k=1, include_values=False, include_metadata=True)
                    matches = query_resp.get("matches", [])
                except Exception:
                    matches = []

            is_dup = False
            if matches:
                # Pinecone returns 'score' where higher is better for cosine in many setups.
                m0 = matches[0]
                score = m0.get("score", None)
                # If score is None, sometimes 'matches' contains nested structures; be conservative
                if score is not None and score >= DUPLICATE_SIM_THRESHOLD:
                    is_dup = True

            if is_dup:
                total_skipped_duplicates += 1
                continue

            # prepare vector for upsert
            vec_id = f"{slugify_from_url(meta.get('source_url'))}--{meta.get('chunk_index')}"
            all_vectors_batch.append({
                "id": vec_id,
                "values": vec,
                "metadata": {
                    "source_url": meta.get("source_url"),
                    "file": meta.get("file"),
                    "chunk_index": meta.get("chunk_index"),
                    "text_preview": chunk_text[:500]
                }
            })

            # upsert in batches
            if len(all_vectors_batch) >= UPSERT_BATCH:
                idx.upsert(vectors=all_vectors_batch)
                upsert_count += len(all_vectors_batch)
                all_vectors_batch = []

        # small delay to avoid hammering (Ollama local is fast; adjust if needed)
        time.sleep(0.05)

    # final flush
    if all_vectors_batch:
        idx.upsert(vectors=all_vectors_batch)
        upsert_count += len(all_vectors_batch)
        all_vectors_batch = []

    print(f"\nDone. Upserted {upsert_count} vectors. Skipped {total_skipped_duplicates} near-duplicates.")
    print("You can now query Pinecone with embeddings produced by Ollama.")

if __name__ == "__main__":
    main()
