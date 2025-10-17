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

**gikiQA** is a Retrieval-Augmented Generation (RAG) chatbot designed for **Ghulam Ishaq Khan Institute (GIKI)**.  
It automatically **crawls official pages**, **cleans the data**, **embeds it locally**, and enables **natural language querying** via a smart LangChain-powered chatbot.  

---

## 🛠️ Tech Stack

| **Layer**      | **Technology**               | **Description** |
|----------------|------------------------------|-----------------|
| 🕷️ Crawler     | **Crawl4AI**                 | Asynchronous, robust website crawler |
| 🧹 Processing   | **LangChain Text Splitter**  | Clean & chunk raw data |
| 🔢 Embeddings   | **Ollama / FastEmbed**       | Generate 1024-dim vectors locally |
| 🧮 Vector DB    | **Pinecone**                 | High-performance vector search |
| 💬 Reasoning    | **LangChain + OpenAI**       | Context-aware chatbot |
| ⚙️ Backend      | **FastAPI** *(Upcoming)*     | Web interface for queries |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/gikiQA.git
cd gikiQA
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # On Windows
# source venv/bin/activate   # On Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX=gikiqa-index
PINECONE_ENVIRONMENT=us-east1-gcp
```

### 5️⃣ Run the Crawler & Embedding Pipeline
```bash
python crawler_pipeline.py
```

This will:
- ✅ Crawl target GIKI pages
- ✅ Clean and chunk the text
- ✅ Generate embeddings via Ollama
- ✅ Upload them to Pinecone

### 6️⃣ Run the Chatbot
```bash
python chatbot.py
```

**Example Interaction:**
```
You: who is the current rector of giki?
Bot: The current Rector of GIKI is ...
```

---

## ⚡ FastAPI Integration (Coming Soon)

You'll soon be able to serve the chatbot through a REST API:
```python
from fastapi import FastAPI
from chatbot import qa_chain

app = FastAPI()

@app.post("/query")
async def query_giki(question: str):
    response = qa_chain.invoke({"question": question})
    return {"response": response}
```

---

## 📂 Project Structure
```bash
gikiQA/
│
├── crawler_pipeline.py      # Crawl + clean + embed pipeline
├── chatbot.py               # LangChain + Pinecone chatbot
├── .env                     # Environment variables
├── requirements.txt         # Dependencies
└── README.md                # You're reading it!
```

---

## 🌟 Example Query
```
You: Tell me about the hostels in GIKI.
Bot: GIKI has multiple student hostels including...
```

---

## 📅 Upcoming Features

- 🌐 Web UI via FastAPI + React/Svelte
- 🔍 RAG optimization using hybrid search
- 🧭 Contextual re-ranking
- 📚 PDF ingestion for academic data
- 🤖 Fine-tuned campus-specific model

---

## 🤝 Contributing

Contributions are welcome! You can:
- 🧾 Submit issues or feature requests
- 🚀 Create PRs for improvements
- 🌍 Share new GIKI data sources

---

## 🧾 License

This project is licensed under the MIT License — see the LICENSE file for details.
