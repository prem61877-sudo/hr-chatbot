import streamlit as st
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="GlobalTech HR Bot", page_icon="🏢")
st.title("🏢 GlobalTech HR Assistant")

# 2. Key Guard: Stop if key is missing
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ API Key not found in Streamlit Secrets!")
    st.stop()

# Explicitly set environment variable as a backup
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def load_rag_system():
    if not os.path.exists("data.pdf"):
        return None
        
    # Load and Split
    loader = PyPDFLoader("data.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(loader.load())
    
    # Brain (Embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # AI Model (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=st.secrets["GOOGLE_API_KEY"], # 2025 standard parameter
        temperature=0,
        convert_system_message_to_human=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # Prompt & Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR Assistant. Use the context to answer: {context}"),
        ("human", "{input}")
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Initialize
rag_chain = load_rag_system()

# 3. Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Ask about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the handbook..."):
            response = rag_chain.invoke({"input": prompt_text})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
