import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

st.title("📄 PDF Chatbot")
st.write("Upload a PDF and ask any question about it!")

# ---- PDF Upload ----
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

@st.cache_resource
def setup(file_path):
    # Step 1: Load
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Step 3 & 4: Embed + Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

    # Step 5: Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Gemini client
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    return retriever, client_gemini

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("✅ PDF uploaded successfully!")

    retriever, client_gemini = setup(tmp_path)

    # ---- Chat ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve
                relevant_chunks = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in relevant_chunks])

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
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.text
                })
else:
    st.info("👆 Please upload a PDF to get started!")