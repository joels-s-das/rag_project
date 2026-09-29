import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
load_data → documents created ✅
split_docs → documents updated ✅
create_vectorstore → uses documents ✅
'''

llm = None
vectordb = None


def dotenv_load():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("❌ GROQ_API_KEY not found in .env")
    print("✅ GROQ API Key Loaded")






def initialize_components():
    global llm
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )

def load_data(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        # print(f"✅ Loaded {len(docs)} documents")
        return docs
    except Exception as e:
        print("❌ Error loading URLs:", e)
        return []



vectordb = None


def create_vectorstore(documents):

    global vectordb

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings
    )

    print("✅ Vector DB created")

    return vectordb

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = splitter.split_documents(documents)

    cleaned_docs = [
        d for d in docs
        if d.page_content.strip()
    ]
    return cleaned_docs

from langchain.prompts import PromptTemplate
def get_qa_chain(vectorstore):
    global llm

    if llm is None:
        raise ValueError("❌ LLM not initialized")
    retriever = vectorstore.as_retriever()


    return RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
    )