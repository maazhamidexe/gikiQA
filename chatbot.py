import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Load environment ----------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- OpenAI Embedding Model ----------
# ✅ Must match your Pinecone index embedding dimension (1536)
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

# ---------- Pinecone setup ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text_preview",  # Must match what you used during ingestion
)

# ---------- LLM setup ----------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY,
)

# ---------- Custom RAG Prompt ----------
prompt_template = """
You are GIKI's intelligent assistant designed to help students, staff, and visitors
with accurate and friendly answers about the Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI).

Use the retrieved context to answer the question clearly and precisely.
If the answer isn't found in the context, politely say you are not sure and
suggest contacting the relevant department.

--- Context ---
{context}
--- Question ---
{question}

Now, provide your answer as GIKI’s official assistant:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# ---------- QA Chain ----------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
)

print("✅ GIKI Chatbot connected to Pinecone and ready.\nType 'exit' to quit.")

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
