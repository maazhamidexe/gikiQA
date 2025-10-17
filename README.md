<h1 align="center">🎓 gikiQA</h1>

<p align="center">
  <em>An AI-powered RAG chatbot tailored for <strong>GIKI</strong> — enabling intelligent Q&A over campus knowledge.</em>  
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-Framework-orange" alt="LangChain" />
  <img src="https://img.shields.io/badge/Pinecone-VectorDB-green" alt="Pinecone" />
  <img src="https://img.shields.io/badge/Crawl4AI-WebScraper-yellow" alt="Crawl4AI" />
  <img src="https://img.shields.io/badge/Ollama-Embeddings-red" alt="Ollama" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688" alt="FastAPI" />
</p>

---

## 🧠 Overview

**gikiQA** is an **AI-powered Retrieval-Augmented Generation (RAG)** chatbot built to understand and answer questions related to **GIKI (Ghulam Ishaq Khan Institute)**.

It scrapes, cleans, and chunks campus data using **Crawl4AI**, creates vector embeddings via **Ollama**, stores them in **Pinecone**, and performs **semantic search** combined with **LLM reasoning** through LangChain.

The end goal — a **FastAPI-based web app** that allows students, faculty, and visitors to query anything about GIKI and get smart, context-aware answers instantly.

---

## 🚀 Features

✨ **Web Crawling:** Crawl any GIKI-related website using Crawl4AI.  
🧩 **Chunking & Cleaning:** Smart text preprocessing for RAG pipelines.  
🧠 **Embeddings via Ollama:** Fast local embeddings with 1024-dim vectors.  
📦 **Vector Storage:** Efficient similarity search using Pinecone.  
💬 **Chat Interface:** LangChain-powered conversational pipeline.  
⚡ **FastAPI Backend:** Coming soon — for real-time deployment and APIs.  
🔐 **Environment Safe:** All keys and configs loaded via `.env`.  

---

## 🧱 Architecture

```mermaid
flowchart TD
    A[🌐 Crawl4AI] --> B[🧹 Text Cleaning & Chunking]
    B --> C[🧠 Ollama Embeddings (1024-dim)]
    C --> D[📦 Pinecone Vector DB]
    D --> E[🤖 LangChain QA Retrieval]
    E --> F[⚡ FastAPI Endpoint]
    F --> G[💬 User Chat Interface]
