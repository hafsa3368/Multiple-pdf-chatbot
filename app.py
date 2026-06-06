import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

PROMPT_TEMPLATE = """
You are an intelligent AI research assistant.
Use the provided PDF context to answer the user's question.
Rules:
1. Answer only from the provided context.
2. If information is missing, say: "The uploaded documents do not contain this information."
3. Give detailed and structured answers.
4. Cite important facts from the context.
5. Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Connecting to LLM...")
def get_llm(api_key: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )

def load_pdf_texts(pdf_files):
    texts = []
    for uploaded_file in pdf_files:
        reader = PdfReader(uploaded_file)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if page_texts:
            texts.append("\n".join(page_texts))
    return texts

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - chunk_overlap
    return [chunk for chunk in chunks if chunk]

def create_documents(texts):
    documents = []
    for idx, text in enumerate(texts, start=1):
        for chunk_idx, chunk in enumerate(chunk_text(text), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": f"pdf_{idx}_chunk_{chunk_idx}"},
                )
            )
    return documents

def build_vector_store(documents):
    return FAISS.from_documents(documents, get_embeddings())

def format_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload one or more PDF files and click Process.",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_texts = load_pdf_texts(pdf_docs)
                    if not raw_texts:
                        st.error("No text could be extracted from the uploaded PDFs.")
                    else:
                        documents = create_documents(raw_texts)
                        st.session_state.vector_store = build_vector_store(documents)
                        st.session_state.chat_history = []
                        st.success(f"Done! {len(documents)} chunks indexed.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        if st.session_state.vector_store is None:
            with st.chat_message("assistant"):
                st.info("Please upload and process PDFs first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_store.similarity_search(question, k=3)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    prompt = format_prompt(context=context, question=question)
                    response = get_llm(GROQ_API_KEY).invoke(prompt)
                    answer = response.content
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

main()
