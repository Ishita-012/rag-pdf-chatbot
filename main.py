from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

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
test_sentence = "What is a transaction in a database?"
embedding = model.encode(test_sentence)
print("✅ Step 3 test - Embedding created!")
print("   Number of values:", len(embedding))
print("   First 5 values:", embedding[:5])
# Embed all chunks
chunk_embeddings = model.encode(chunks)
print("✅ All chunks embedded!")
print("   Shape:", chunk_embeddings.shape)

# ---- STEP 4: Store in ChromaDB ----
client = chromadb.Client()
collection = client.create_collection("pdf_chunks")

collection.add(
    documents=chunks,
    embeddings=[e.tolist() for e in chunk_embeddings],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("✅ Step 4 done - Stored in ChromaDB!")
print("   Total chunks stored:", collection.count())

# ---- STEP 5: Search by question ----
question = "What are ACID properties?"

results = collection.query(
    query_embeddings=[model.encode(question).tolist()],
    n_results=3
)

print("\n✅ Step 5 done - Top 3 relevant chunks found!")
print("\n--- Result 1 ---")
print(results["documents"][0][0])
print("\n--- Result 2 ---")
print(results["documents"][0][1])
print("\n--- Result 3 ---")
print(results["documents"][0][2])

# ---- STEP 6: Get answer from Gemini ----
from google import genai
from dotenv import load_dotenv
import os

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

print("\n✅ Step 6 done - Gemini answered!")
print("\n--- ANSWER ---")
print(response.text)

# Temporary - just to check available models
for model in client_gemini.models.list():
    print(model.name)