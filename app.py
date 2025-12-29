import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Config ---
st.set_page_config(page_title="GlobalTech HR", page_icon="🏢")
st.title("🏢 GlobalTech HR Assistant")

# --- API Key Check ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

@st.cache_resource
def load_rag_system():
    if not os.path.exists("data.pdf"):
        return None
        
    # 1. Load and Chunk PDF
    loader = PyPDFLoader("data.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(loader.load())
    
    # 2. Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
   # 🟢 UPDATED: Dec 2025 Production-ready config
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=1.0, # Recommended for Gemini 3
        convert_system_message_to_human=True, 
        safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        }
    )
    
    # 4. Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("human", "You are an HR Specialist. Using ONLY the provided context, answer the user's question. If the information is not in the context, say 'I cannot find that in the HR handbook.'\n\nContext: {context}\n\nQuestion: {input}")
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# --- Initializing ---
rag_chain = load_rag_system()

if rag_chain is None:
    st.error("Please upload 'data.pdf' to your GitHub repository.")
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about HR..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting HR data..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Check if your Google Cloud project has Gemini 3 Flash enabled.")
