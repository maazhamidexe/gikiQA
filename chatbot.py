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

# ---------- Ollama Embedding Function ----------
def ollama_embed(texts):
    """Embed a list of texts using local Ollama model."""
    if isinstance(texts, str):
        texts = [texts]
    response = requests.post(
        "http://localhost:11434/api/embed",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"model": "mxbai-embed-large", "input": texts}),
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama embedding error: {response.text}")
    data = response.json()
    return data["embeddings"]

# ---------- Embedding Wrapper ----------
class OllamaEmbeddings:
    def embed_query(self, text):
        return ollama_embed(text)[0]
    def embed_documents(self, texts):
        return ollama_embed(texts)

embed_model = OllamaEmbeddings()

# ---------- Pinecone setup ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Tell LangChain to use `text_preview` instead of `text`
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text_preview",  # 👈 important fix
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

print("✅ Chatbot connected to Pinecone and ready.\nType 'exit' to quit.")

# ---------- Chat Loop ----------
while True:
    query = input("\nYou: ").strip()
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break
    try:
        # use .invoke() instead of .run() (modern LangChain)
        result = qa.invoke({"query": query})
        print(f"Bot: {result['result']}")
    except Exception as e:
        print(f"[Error] {e}")
