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
        "input_type": "document"  # can be "document" or "query"
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

print("✅ Chatbot connected to Pinecone + Voyage AI embeddings.\nType 'exit' to quit.")

# ---------- Chat Loop ----------
while True:
    query = input("\nYou: ").strip()
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break
    try:
        result = qa.invoke({"query": query})
        print(f"Bot: {result['result']}")
    except Exception as e:
        print(f"[Error] {e}")
