"""
Exploratory / learning script for the RAG PDF chatbot pipeline.
Walks through: load PDF -> chunk -> embed -> store in ChromaDB -> retrieve -> ask Gemini.
See app.py for the final Streamlit app.
"""

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import os

# ---- STEP 1: Load PDF ----
reader = PdfReader("docsample.pdf")
all_text = ""
for page in reader.pages:
    all_text += page.extract_text()
print("✅ Step 1 done - PDF loaded, Length:", len(all_text))

# ---- STEP 2: Chunking ----
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(all_text)
print("✅ Step 2 done - Total chunks:", len(chunks))

# ---- STEP 3: Embeddings ----
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)
print("✅ Step 3 done - All chunks embedded! Shape:", chunk_embeddings.shape)

# ---- STEP 4: Store in ChromaDB ----
client = chromadb.Client()
collection = client.create_collection("pdf_chunks")

collection.add(
    documents=chunks,
    embeddings=[e.tolist() for e in chunk_embeddings],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("✅ Step 4 done - Stored in ChromaDB! Total chunks stored:", collection.count())

# ---- STEP 5: Search by question ----
question = "What are ACID properties?"

results = collection.query(
    query_embeddings=[model.encode(question).tolist()],
    n_results=3
)
print("✅ Step 5 done - Top 3 relevant chunks found!")
for i, doc in enumerate(results["documents"][0], start=1):
    print(f"\n--- Result {i} ---")
    print(doc)

# ---- STEP 6: Get answer from Gemini ----
load_dotenv()
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

retrieved_chunks = results["documents"][0]
context = "\n\n".join(retrieved_chunks)

prompt = f"""You are a helpful assistant.
Using ONLY the context below, answer the question.

Context:
{context}

Question: {question}
Answer:"""

response = client_gemini.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
print("\n✅ Step 6 done - Gemini answered!\n--- ANSWER ---")
print(response.text)