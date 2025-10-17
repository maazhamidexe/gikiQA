from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ---------- Load environment ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# ---------- Voyage AI Embedding Function ----------
def voyage_embed(texts, model="voyage-3-large"):
    """Embed text(s) using Voyage AI API (1024-dim)."""
    if isinstance(texts, str):
        texts = [texts]

    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": texts,
        "model": model,
        "input_type": "document"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Voyage embedding error: {response.text}")

    data = response.json()
    return [item["embedding"] for item in data.get("data", [])]

# ---------- Embedding Wrapper ----------
class VoyageEmbeddings:
    def embed_query(self, text):
        return voyage_embed(text)[0]
    def embed_documents(self, texts):
        return voyage_embed(texts)

embed_model = VoyageEmbeddings()

# ---------- Pinecone setup ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ---------- Vector Store ----------
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text_preview",  # make sure your Pinecone entries have this field
)

# ---------- LLM setup ----------
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY,
)

# ---------- QA Chain ----------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
)

# ---------- FastAPI ----------
app = FastAPI(title="GIKI Chatbot API", version="1.0")

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """Handles chat queries from frontend."""
    query = request.query.strip()
    if not query:
        return {"error": "Empty query."}
    try:
        result = qa.invoke({"query": query})
        return {"query": query, "answer": result["result"]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "✅ GIKI Chatbot API is running!"}
