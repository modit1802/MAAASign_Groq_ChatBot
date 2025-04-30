import os
import requests
import uvicorn
import pickle
import hashlib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import logging

# Load env variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("CHATGROQ_API_KEY")

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Constants
CSV_FILE = "videos.csv"
FAISS_DIR = "faiss_index"
EMBEDDING_PICKLE = "cached_embeddings.pkl"
CSV_HASH_FILE = "csv_hash.txt"

# FastAPI app
app = FastAPI(title="Langchain ISL Server", version="1.0")

# Model for user input
class QuestionRequest(BaseModel):
    query: str

# Load CSV
def load_csv():
    try:
        df = pd.read_csv(CSV_FILE)
        logging.info("CSV loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        raise HTTPException(status_code=500, detail="Error loading CSV")

# Create hash of current CSV
def get_csv_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# Save embeddings to pickle
def save_embeddings(docs, embeddings):
    with open(EMBEDDING_PICKLE, "wb") as f:
        pickle.dump((docs, embeddings), f)

# Load embeddings from pickle
def load_cached_embeddings():
    if os.path.exists(EMBEDDING_PICKLE):
        with open(EMBEDDING_PICKLE, "rb") as f:
            return pickle.load(f)
    return None, None

# Create or reuse embeddings and vectorstore
def get_vectorstore(df):
    new_hash = get_csv_hash(df)

    # Read old hash if exists
    old_hash = None
    if os.path.exists(CSV_HASH_FILE):
        with open(CSV_HASH_FILE, "r") as f:
            old_hash = f.read().strip()

    # If data hasn't changed and vectorstore exists
    if new_hash == old_hash and os.path.exists(f"{FAISS_DIR}/index.faiss"):
        logging.info("Reusing existing vectorstore and embeddings.")
        docs, embeddings = load_cached_embeddings()
        if docs and embeddings:
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True), docs, embeddings

    # If data changed, create everything new
    logging.info("Data changed or no cache found. Creating embeddings and FAISS index.")
    documents = DataFrameLoader(df, page_content_column="Relations").load()

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACE_API_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_DIR)
    save_embeddings(documents, embeddings)

    # Save new CSV hash
    with open(CSV_HASH_FILE, "w") as f:
        f.write(new_hash)

    return vectorstore, documents, embeddings

# Create QA chain
def create_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="gemma2-9b-it"
    )

    prompt_template = """You are an intelligent assistant which provides answers based on Indian Sign Language (ISL). Use the given context to answer the question.
If context is insufficient, provide a helpful and general explanation about ISL.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Fallback web search
def fetch_web_content(query):
    try:
        search_url = f"https://www.google.com/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for g in soup.find_all('div', class_='tF2Cxc'):
            title_tag = g.find('h3')
            link_tag = g.find('a')
            snippet_tag = g.find('span')

            if title_tag and link_tag:
                results.append({
                    'title': title_tag.get_text(),
                    'link': link_tag['href'],
                    'snippet': snippet_tag.get_text() if snippet_tag else "No description"
                })
        return results
    except Exception as e:
        logging.error(f"Web search error: {e}")
        return []

# API Endpoint
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        df = load_csv()
        vectorstore, documents, embeddings = get_vectorstore(df)
        qa_chain = create_chain(vectorstore)

        response = qa_chain.invoke({"query": request.query})
        answer = response.get("result", "")
        source_docs = response.get("source_documents", [])

        result = {
            "answer": answer,
            "source_urls": []
        }

        for doc in source_docs:
            try:
                match = df[df['Relations'].str.strip() == doc.page_content.strip()]
                if not match.empty:
                    result["source_urls"].append(match.iloc[0]['URL'])
            except Exception as e:
                logging.warning(f"Could not match source doc: {e}")

        if "sorry" in answer.lower() or not answer.strip():
            result["web_short_descriptions"] = fetch_web_content(request.query)

        return result

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
