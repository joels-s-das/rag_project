import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

def dotenv_load():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")

    if key:
        print("✅ GROQ KEY LOADED:", key[:5] + "*****")
    else:
        print("❌ GROQ KEY NOT FOUND")
    print(key[:40])

# dotenv_load()
def load_data(urls):
    loader = WebBaseLoader(urls)
    return loader.load()

urls = [
    "https://www.example.com"
]

documents = load_data(urls)
print("Loaded docs:", len(documents))

def load_data(urls):

    try:
        loader = WebBaseLoader(urls)

        docs = loader.load()

        print(f"Loaded docs: {len(docs)}")

        return docs

    except Exception as e:

        print("Error loading URLs:", e)

        return []
