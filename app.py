import os
import requests
import uvicorn
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
import pandas as pd
import logging

# Setup Logging
logging.basicConfig(level=logging.DEBUG)

# Load env variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("CHATGROQ_API_KEY")

# FastAPI app
app = FastAPI(
    title="Langchain server",
    version="1.0",
    description="A simple API server"
)

# CSV Loader
def load_csv():
    try:
        df = pd.read_csv('videos.csv')
        logging.debug("CSV loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        raise HTTPException(status_code=500, detail="Error loading CSV")

# Documents creator
def create_documents(df):
    try:
        loader = DataFrameLoader(df, page_content_column="Relations")
        documents = loader.load()
        logging.debug(f"Documents created: {len(documents)}")
        return documents
    except Exception as e:
        logging.error(f"Error creating documents: {e}")
        raise HTTPException(status_code=500, detail="Error creating documents")

# Embeddings creator
def create_embeddings():
    try:
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACE_API_TOKEN,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Error creating embeddings")

# Vectorstore loader/creator
def get_or_create_vectorstore(documents, embeddings):
    try:
        if os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index/index.pkl"):
            logging.debug("Loading FAISS vectorstore from disk...")
            vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            logging.debug("Creating new FAISS vectorstore...")
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local("faiss_index")
        return vectorstore
    except Exception as e:
        logging.error(f"Error in vectorstore setup: {e}")
        raise HTTPException(status_code=500, detail="Error in vectorstore setup")

# QA chain
def create_chain(vectorstore):
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="gemma2-9b-it"
        )

        prompt_template = """You are an intelligent assistant which provides answer to user on the basis of Indian Sign Language. Answer the following question based on the context provided.
If the context does not provide an answer, generate a helpful response by summarizing what you know about the topic which is based upon Indian Sign Language.
Do not default to saying "Sorry, I don't know." Instead, try to provide a general answer or explanation on ISL.

Context:
{context}

Question:
{question}

Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        logging.error(f"Error creating QA chain: {e}")
        raise HTTPException(status_code=500, detail="Error creating QA chain")

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
        logging.error(f"Error fetching web content: {e}")
        return []

# Request model
class QuestionRequest(BaseModel):
    query: str

# API Endpoint
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        logging.debug("Request received")

        df = load_csv()
        documents = create_documents(df)
        embeddings = create_embeddings()
        vectorstore = get_or_create_vectorstore(documents, embeddings)
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
                logging.warning(f"Couldn't match source doc: {e}")

        if "sorry" in answer.lower() or not answer.strip():
            web_content = fetch_web_content(request.query)
            result["web_short_descriptions"] = web_content if web_content else "No relevant web content found."

        return result

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# Run app
if __name__ == "__app__":
    uvicorn.run(app, host="localhost", port=8000)
