import streamlit as st
import pandas as pd
import os
import requests
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

# API keys from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("CHATGROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")  # Optional

# Step 1: Load the CSV
@st.cache_data
def load_csv():
    df = pd.read_csv('videos.csv')
    return df

# Step 2: Create documents from CSV
@st.cache_data
def create_documents(df):
    loader = DataFrameLoader(df, page_content_column="Relations")
    documents = loader.load()
    return documents

# Step 3: Create Embeddings
@st.cache_resource
def create_embeddings():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACE_API_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

# Step 4: Create Vectorstore
@st.cache_resource
def create_vectorstore(_documents, _embeddings):
    vectorstore = FAISS.from_documents(_documents, _embeddings)
    return vectorstore

# Step 5: Create LLM and Chain with Prompt
@st.cache_resource
def create_chain(_vectorstore):
    # Initialize the Groq-based LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="gemma2-9b-it"
    )

    # Modified prompt to ensure better relevance and prioritization
    prompt_template = """You are an intelligent assistant. Answer the following question based on the context provided.
    If the context does not provide an answer, generate a helpful response by summarizing what you know about the topic. 
    Do not default to saying "Sorry, I don't know." Instead, try to provide a general answer or explanation.

    Context (prioritize this context first):
    {context}

    Question:
    {question}

    Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# Step 6: Fetch web content using a search query with better relevance
def fetch_web_content(query):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the top search results, prioritize by title and snippet
    search_results = []
    for item in soup.find_all('h3'):
        title = item.text
        snippet = item.find_parent().find('span', class_='aCOpRe')
        snippet_text = snippet.text if snippet else "No description available"
        link = item.find_parent()['href']

        # Ranking search results by relevance (title, snippet)
        search_results.append({
            'title': title,
            'snippet': snippet_text,
            'link': link
        })

    return search_results

# Streamlit Frontend
def main():
    st.set_page_config(page_title="RAG Chatbot (CSV based + Web Content)", page_icon="🤖", layout="wide")
    st.title("🤖 Video Relations Chatbot with Web Content")
    st.write("Ask anything related to the video relations and get responses from both CSV and web content!")

    # Load the CSV file, create documents, embeddings, and vector store
    df = load_csv()
    documents = create_documents(df)
    embeddings = create_embeddings()
    vectorstore = create_vectorstore(documents, embeddings)
    qa_chain = create_chain(vectorstore)

    # Ask the user for input
    user_query = st.text_input("Ask your question:")

    if user_query:
        with st.spinner("Thinking... 🤔"):
            # Step 1: First query the vectorstore for relevant content
            response = qa_chain(user_query)

            # Step 2: Check if vectorstore provided a relevant answer
            if response['result'] == "Sorry, I don't know.":
                st.markdown("### 📜 Answer:")
                st.write(response['result'])
                st.markdown("### 🔗 Source URL(s): None found.")
            else:
                # Show the LLM's short description answer from vectorstore
                st.markdown("### 📜 Short Description:")
                st.write(response['result'])

                # Show URLs from the CSV content
                st.markdown("### 🔗 Source URL(s) from CSV:")
                for doc in response["source_documents"]:
                    row_index = df[df['Relations'] == doc.page_content].index[0]
                    url = df.loc[row_index, 'URL']
                    st.write(f"- [{url}]({url})")

            # Step 3: Now query the web for content if no relevant vectorstore result
            if response['result'] == "Sorry, I don't know." or not response['result']:
                web_content = fetch_web_content(user_query)

                if web_content:
                    st.markdown("### 🔗 Relevant Web Search Results:")
                    for result in web_content:
                        st.write(f"- **{result['title']}**: {result['snippet']} [Link]({result['link']})")
                else:
                    st.markdown("### 🔗 No relevant web content found.")

if __name__ == "__main__":
    main()
