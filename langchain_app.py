from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# ---- STEP 1: Load PDF ----
loader = PyPDFLoader("docsample.pdf")
documents = loader.load()
print("✅ Step 1 - PDF loaded, pages:", len(documents))

# ---- STEP 2: Split into chunks ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print("✅ Step 2 - Total chunks:", len(chunks))

# ---- STEP 3 & 4: Embed + Store in ChromaDB ----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
print("✅ Step 3 & 4 - Embedded and stored in ChromaDB!")

# ---- STEP 5: Retriever ----
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Step 5 - Retriever ready!")

# ---- STEP 6: Ask a question ----
question = "What are ACID properties?"
relevant_chunks = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in relevant_chunks])

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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

print("\n✅ Step 6 - Gemini answered!")
print("\n--- ANSWER ---")
print(response.text)