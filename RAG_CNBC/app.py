import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain.prompts import PromptTemplate

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------- LOAD URLS ---------------- #
def load_urls(urls):

    docs = []

    for url in urls:

        try:
            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            res = requests.get(url, headers=headers)

            soup = BeautifulSoup(res.text, "html.parser")

            paragraphs = soup.find_all("p")

            text = " ".join(
                [p.get_text() for p in paragraphs]
            )

            if len(text.strip()) > 100:

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                    )
                )

        except Exception as e:

            st.error(f"Error loading {url}: {e}")

    return docs


# ---------------- ENV ---------------- #
load_dotenv()

# ---------------- UI ---------------- #
st.title("📊 Multi-Source RAG App")

st.sidebar.header("⚙️ Data Source Selection")

source_type = st.sidebar.radio(
    "Choose Data Source",
    ["PDF", "CSV", "URL"]
)

pdf_file = None
csv_file = None
urls = []


# ---------------- PDF ---------------- #
if source_type == "PDF":

    pdf_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )


# ---------------- CSV ---------------- #
elif source_type == "CSV":

    csv_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


# ---------------- URL ---------------- #
elif source_type == "URL":

    use_default = st.sidebar.checkbox(
        "Use sample CNBC URLs",
        value=False
    )

    if use_default:

        urls = [
            "https://www.cnbc.com/2026/04/22/tesla-tsla-q1-2026-earnings-report.html",
            "https://www.cnbc.com/2026/01/28/tesla-tsla-2025-q4-earnings.html"
        ]

    else:

        url_input = st.sidebar.text_area(
            "Enter URLs (one per line)"
        )

        if url_input:

            urls = [
                u.strip()
                for u in url_input.split("\n")
                if u.strip()
            ]


# ---------------- QUERY ---------------- #
query = st.text_input("Ask a question")


# ---------------- RUN ---------------- #
if st.button("Run"):

    # -------- API KEY -------- #
    if not os.getenv("GROQ_API_KEY"):

        st.error("Missing GROQ_API_KEY in .env")
        st.stop()

    # -------- QUERY CHECK -------- #
    if not query:

        st.warning("Please enter a question")
        st.stop()

    docs = []

    # -------- LOAD PDF -------- #
    if source_type == "PDF" and pdf_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(pdf_file.read())

            docs.extend(
                PyPDFLoader(tmp.name).load()
            )

    # -------- LOAD CSV -------- #
    elif source_type == "CSV":

        if csv_file is None:

            st.error("Please upload CSV file")
            st.stop()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as tmp:

            tmp.write(csv_file.read())

            tmp_path = tmp.name

        loader = CSVLoader(file_path=tmp_path)

        csv_docs = loader.load()

        docs.extend(csv_docs)

        # st.write("CSV rows loaded:", len(csv_docs))

    # -------- LOAD URL -------- #
    elif source_type == "URL":

        if not urls:

            st.error(
                "Please enter URLs or enable sample URLs"
            )

            st.stop()

        docs.extend(load_urls(urls))

    # -------- DOC CHECK -------- #
    # st.write("Loaded docs:", len(docs))

    if len(docs) == 0:

        st.error("No documents loaded")
        st.stop()

    # -------- SPLIT -------- #
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # st.write("Chunks:", len(chunks))

    if len(chunks) == 0:

        st.error("No chunks created")
        st.stop()

    # -------- EMBEDDINGS -------- #
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- VECTOR DB -------- #
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    # -------- LLM -------- #
    MODEL_NAME = "llama-3.3-70b-versatile"

    try:

        llm = ChatGroq(model=MODEL_NAME)

    except Exception as e:

        st.error(f"Model error: {e}")
        st.stop()

    # -------- PROMPT -------- #
    template = """
    You are a financial analysis assistant.


    Rules:
    - Give concise answers
    - Avoid repeating points
    - Avoid duplicate text
    - Use bullet points
    - If information is unavailable, say so clearly
    - Include all important regions and products
    - Summarize complete trends from the dataset


    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # -------- QA CHAIN -------- #
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(
            search_kwargs={"k": 6}
        ),
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    # -------- QUERY -------- #
    with st.spinner("Generating answer..."):

        result = qa.invoke(query)

    # -------- OUTPUT -------- #
    st.write("## Answer")

    st.success(result["result"])