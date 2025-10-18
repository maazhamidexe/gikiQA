#!/usr/bin/env python3
"""
build_vectorstore_openai_hierarchical.py

Upgraded pipeline:
- reads giki_index.jsonl (Crawl4AI output)
- cleans markdown
- hierarchical chunking:
    - split by headings into sections
    - split each section into semantic chunks (langchain splitter)
    - optionally group adjacent chunks into 'super-chunks' for broader context
- uses OpenAI text-embedding-3-small for embeddings (cost-efficient, 1536d).
- creates a new Pinecone index automatically (if none provided) with dim=1536
- deduplicates with Pinecone and upserts in batches
"""

import os
import re
import json
import time
import uuid
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Tuple, Dict

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- web-verified model choice ---
# We use OpenAI text-embedding-3-small (cost-effective, 1536-dim embeddings).
# Sources: OpenAI model announcement and docs.
# (See inline citations in assistant response.)
# ----------------------------------------------------

load_dotenv()

# Env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
# If PINECONE_INDEX_NAME not set, we'll create a unique index name automatically.
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()

# Local config
INDEX_FILE = "giki_index.jsonl"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
BATCH_EMBED = int(os.getenv("BATCH_EMBED", 64))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", 100))
DUPLICATE_SIM_THRESHOLD = float(os.getenv("DUPLICATE_SIM_THRESHOLD", 0.985))
PRINT_PREVIEW_CHARS = int(os.getenv("PRINT_PREVIEW_CHARS", 800))

# OpenAI embedding model details
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # text-embedding-3-small uses 1536 dims

# Pinecone import (keeps compatibility with the example style you had)
from pinecone import Pinecone, ServerlessSpec

# ------------------ Helpers ------------------

def clean_markdown(md_text: str) -> str:
    """Clean markdown produced by Crawl4AI and return plain text."""
    txt = md_text.replace("\r\n", "\n")
    txt = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.DOTALL)
    txt = re.sub(r"(\* ?\[[^\]]+\]\([^)]+\)\s*){3,}", "", txt)
    txt = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", txt)
    txt = re.sub(r"https?://\S+", "", txt)
    txt = re.sub(r"^\s*[\*\-\+]\s*", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    soup = BeautifulSoup(txt, "html.parser")
    text = soup.get_text("\n")
    lines = [ln.rstrip() for ln in text.splitlines()]

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

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()

def make_uuid_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"

# ------------------ Hierarchical chunking ------------------

def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text by top-level headings (markdown #, ##, ###).
    Returns list of tuples: (section_heading, section_text)
    The first entry will be the "document" if content before any heading exists.
    """
    pattern = re.compile(r"(?m)^(#{1,6}\s.*)$")
    parts = pattern.split(text)
    # pattern.split yields [pre, heading1, text1, heading2, text2, ...]
    sections = []
    if len(parts) == 1:
        # no headings found: single section
        sections.append(("document", text.strip()))
        return sections
    pre = parts[0].strip()
    if pre:
        sections.append(("document", pre))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        sections.append((heading, body))
    return sections

def hierarchical_chunk(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Produce hierarchical chunks:
      - top-level sections (by heading)
      - within each section, split into semantic chunks using RecursiveCharacterTextSplitter
      - additionally produce 'super-chunks' by grouping adjacent chunk texts to provide broader context
    Returns list of (chunk_text, metadata)
    """
    sections = split_into_sections(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    all_chunks = []
    for sec_idx, (heading, sec_text) in enumerate(sections):
        if not sec_text or len(sec_text.strip()) < 50:
            continue
        sec_id = make_uuid_id("sec-")
        # Split into semantic chunks
        semantic_chunks = splitter.split_text(sec_text)
        # filter & normalize
        semantic_chunks = [c.strip() for c in semantic_chunks if c and len(c.strip()) >= 40]

        # create metadata for each chunk
        for i, ch in enumerate(semantic_chunks):
            meta = {
                "section_id": sec_id,
                "section_heading": heading,
                "section_index": sec_idx,
                "chunk_index_in_section": i,
                "section_preview": sec_text[:600]
            }
            all_chunks.append((ch, meta))

        # create 'super-chunks' to capture broader context by joining adjacent chunk groups
        # we create super-chunks of size 3 (overlapping) to preserve context across neighboring chunks
        group_size = 3
        for gstart in range(0, max(1, len(semantic_chunks)), group_size):
            group_text = " ".join(semantic_chunks[gstart:gstart+group_size])
            if len(group_text.strip()) < 40:
                continue
            super_meta = {
                "section_id": sec_id,
                "section_heading": heading,
                "section_index": sec_idx,
                "superchunk_group_start": gstart,
                "superchunk_group_size": min(group_size, len(semantic_chunks)-gstart),
                "section_preview": sec_text[:600],
                "is_superchunk": True
            }
            all_chunks.append((group_text, super_meta))
    return all_chunks

# ------------------ OpenAI Embeddings ------------------

def openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Batch embeddings using OpenAI Embed API.
    Uses REST call to avoid dependency on 3rd party wrappers — also handles batching.
    Returns list of embedding vectors (floats).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # OpenAI accepts list input in the 'input' field
    payload = {
        "model": OPENAI_EMBED_MODEL,
        "input": texts
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # data["data"] will be list of {"embedding": [...], "index": i, ...}
    embeddings = [item["embedding"] for item in data["data"]]
    return embeddings

# ------------------ Pinecone ------------------

def init_pinecone() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("Set PINECONE_API_KEY in your .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    global PINECONE_INDEX_NAME
    if not PINECONE_INDEX_NAME:
        # Create unique name with timestamp to avoid collisions
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        PINECONE_INDEX_NAME = f"giki-openai-emb-{ts}"
        print(f"[pinecone] PINECONE_INDEX_NAME not set — creating '{PINECONE_INDEX_NAME}'")

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[pinecone] Creating index '{PINECONE_INDEX_NAME}' (dim={EMBED_DIM}, metric=cosine)")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT or "us-east-1")
        )
    else:
        print(f"[pinecone] Using existing index '{PINECONE_INDEX_NAME}'")

    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"[pinecone] Connected to index '{PINECONE_INDEX_NAME}'")
    return index

# ------------------ Main pipeline ------------------

def main():
    print("Starting hierarchical OpenAI -> Pinecone pipeline...")

    # Read index file (Crawl4AI)
    entries = []
    with open(INDEX_FILE, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                entries.append(json.loads(ln))
    print(f"Found {len(entries)} entries in {INDEX_FILE}")

    file_chunks = []  # tuples (text, metadata)
    exact_text_set = set()
    for entry in entries:
        md_path = entry.get("file").replace("\\", "/")
        if not os.path.exists(md_path):
            print(f"[warn] file missing: {md_path}")
            continue
        with open(md_path, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_markdown(raw)
        src_url = extract_source_url(raw) or entry.get("url")
        title = entry.get("title") or (src_url and slugify_from_url(src_url)) or md_path

        print("\n" + "="*80)
        print(f"FILE: {md_path}")
        print(f"SOURCE URL: {src_url}")
        print(f"--- cleaned preview (first {PRINT_PREVIEW_CHARS} chars) ---")
        print(cleaned[:PRINT_PREVIEW_CHARS])
        print("="*80 + "\n")

        # hierarchical chunk
        chunks = hierarchical_chunk(cleaned, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for i, (txt, meta) in enumerate(chunks):
            text_norm = txt.strip()
            if len(text_norm) < 40:
                continue
            h = hashlib.sha256(text_norm.encode("utf-8")).hexdigest()
            if h in exact_text_set:
                continue
            exact_text_set.add(h)

            # enrich metadata
            enriched_meta = {
                "source_url": src_url,
                "file": md_path,
                "title": title,
                "text_len": len(text_norm),
                "text_preview": text_norm[:500],
                **meta
            }
            file_chunks.append((text_norm, enriched_meta))

    print(f"Total hierarchical chunks (after exact dedupe): {len(file_chunks)}")

    # Init Pinecone
    idx = init_pinecone()

    # Embedding + dedupe + upsert
    all_vectors_batch = []
    upsert_count = 0
    total_skipped_duplicates = 0

    for batch_start in tqdm(range(0, len(file_chunks), BATCH_EMBED), desc="Embedding batches"):
        batch = file_chunks[batch_start: batch_start + BATCH_EMBED]
        texts = [x[0] for x in batch]
        metas = [x[1] for x in batch]

        # get embeddings
        try:
            embeddings = openai_embeddings_batch(texts)
        except Exception as e:
            print(f"[error] OpenAI embedding failed for batch starting {batch_start}: {e}")
            # fallback: skip batch
            continue

        # iterate embeddings
        prepared = []
        for i, emb in enumerate(embeddings):
            vec = normalize(emb)
            meta = metas[i]
            chunk_text = texts[i]

            # Query Pinecone for near-duplicates
            is_dup = False
            try:
                query_resp = idx.query(vector=vec, top_k=1, include_values=False, include_metadata=True)
                matches = query_resp.get("matches", []) or query_resp.get("results", [])
            except Exception:
                matches = []

            if matches:
                m0 = matches[0]
                score = m0.get("score", None)
                # Pinecone score for cosine is typically in [0..1] higher = closer
                if score is not None and score >= DUPLICATE_SIM_THRESHOLD:
                    is_dup = True

            if is_dup:
                total_skipped_duplicates += 1
                continue

            # create an id that is unique but stable-ish: use uuid for safety
            vec_id = make_uuid_id(prefix="v-")
            prepared.append({
                "id": vec_id,
                "values": vec,
                "metadata": meta
            })

        # upsert in batches to pinecone
        if prepared:
            # batch upserts to avoid giant payloads
            for i in range(0, len(prepared), UPSERT_BATCH):
                chunk_block = prepared[i:i+UPSERT_BATCH]
                try:
                    idx.upsert(vectors=chunk_block)
                    upsert_count += len(chunk_block)
                except Exception as e:
                    print(f"[error] Upsert failed: {e}")
        # small sleep (politeness for OpenAI / API)
        time.sleep(0.05)

    print(f"\nDone. Upserted {upsert_count} vectors. Skipped {total_skipped_duplicates} near-duplicates.")
    print(f"Pinecone index used: {PINECONE_INDEX_NAME}")
    print("Now you can query Pinecone using embeddings from OpenAI text-embedding-3-small.")

if __name__ == "__main__":
    main()
