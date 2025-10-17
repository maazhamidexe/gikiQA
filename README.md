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

- ✨ Web Crawling with Crawl4AI  
- 🧩 Smart text cleaning and chunking  
- 🧠 Local embeddings (1024-dim) using Ollama  
- 📦 Vector storage with Pinecone  
- 💬 Conversational reasoning via LangChain + OpenAI  
- ⚡ FastAPI backend (coming soon)  
- 🔐 Secure `.env` configuration  

---

## 🧱 Architecture

```mermaid
flowchart TD
    A[Crawl4AI Crawler] --> B[Text Cleaning & Chunking]
    B --> C[Ollama Embeddings (1024-dim)]
    C --> D[Pinecone Vector DB]
    D --> E[LangChain QA Retriever]
    E --> F[FastAPI Backend]
    F --> G[User Chat Interface]
🛠️ Tech Stack
Layer	Technology	Description
Crawler	Crawl4AI	Asynchronous, robust website crawler
Processing	LangChain Text Splitter	Clean & chunk raw data
Embeddings	Ollama / FastEmbed	Generate 1024-dim vectors locally
Vector DB	Pinecone	High-performance vector search
Reasoning	LangChain + OpenAI	Context-aware chatbot
Backend	FastAPI	(Upcoming) Web interface for queries
⚙️ Setup & Installation
1️⃣ Clone the Repository
code
Bash
git clone https://github.com/<your-username>/gikiQA.git
cd gikiQA
2️⃣ Create and Activate Virtual Environment
code
Bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
# source venv/bin/activate
3️⃣ Install Dependencies
code
Bash
pip install -r requirements.txt
4️⃣ Configure Environment Variables
Create a .env file in the root directory:
code
Bash
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX=gikiqa-index
PINECONE_ENVIRONMENT=us-east1-gcp
5️⃣ Run the Crawler & Embedding Pipeline
code
Bash
python crawler_pipeline.py
This will:
Crawl target GIKI pages
Clean and chunk the text
Generate embeddings via Ollama
Upload them to Pinecone
6️⃣ Run the Chatbot
code
Bash
python chatbot.py
Example interaction:
code
Vbnet
You: who is the current rector of giki?
Bot: The current Rector of GIKI is ...
⚡ FastAPI Integration (Coming Soon)
Soon you’ll be able to serve this chatbot via an API endpoint:
code
Python
from fastapi import FastAPI
from chatbot import qa_chain

app = FastAPI()

@app.post("/query")
async def query_giki(question: str):
    response = qa_chain.invoke({"question": question})
    return {"response": response}
📂 Project Structure
code
Bash
gikiQA/
│
├── crawler_pipeline.py      # Crawl + clean + embed pipeline
├── chatbot.py               # LangChain + Pinecone chatbot
├── .env                     # Environment variables
├── requirements.txt         # Dependencies
└── README.md                # You’re reading it!
🌟 Example Query
code
Vbnet
You: Tell me about the hostels in GIKI.
Bot: GIKI has multiple student hostels including...
📅 Upcoming Features
Web UI via FastAPI + React/Svelte
RAG optimization using hybrid search
Contextual re-ranking
PDF ingestion for academic data
Fine-tuned campus-specific model
🤝 Contributing
Contributions are welcome! Feel free to:
Submit issues or feature requests 🧾
Create PRs for improvements 🚀
Share new GIKI data sources 🌍
🧾 License
This project is licensed under the MIT License — see the LICENSE file for details.
