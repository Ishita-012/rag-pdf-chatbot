import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# ---- Setup ----
st.title("📄 PDF Chatbot")
st.write("Ask any question about your PDF!")

# ---- Load everything once ----
@st.cache_resource
def setup():
    # Load PDF
    reader = PdfReader("docsample.pdf")
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text()

    # Chunk
    def chunk_text(text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    chunks = chunk_text(all_text)

    # Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks)

    # Store in ChromaDB
    client_chroma = chromadb.Client()
    collection = client_chroma.create_collection("pdf_chunks")
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in chunk_embeddings],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    # Gemini client
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    return collection, model, client_gemini

collection, model, client_gemini = setup()

# ---- Chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
question = st.chat_input("Ask a question about your PDF...")

if question:
    # Show user question
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Search + Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve
            results = collection.query(
                query_embeddings=[model.encode(question).tolist()],
                n_results=3
            )
            context = "\n\n".join(results["documents"][0])

            # Ask Gemini
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

            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})