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

# Page Config
st.set_page_config(page_title="GlobalTech HR", page_icon="🏢")
st.title("🏢 GlobalTech HR Assistant")

# Key Check
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_rag_system():
    if not os.path.exists("data.pdf"):
        return None
        
    # 1. Load PDF
    loader = PyPDFLoader("data.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(loader.load())
    
    # 2. Create Vector Brain
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
   llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash", # 🚀 2025 State-of-the-art
        api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0,
        convert_system_message_to_human=True,
        safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        }
    )
    
    # 4. Prompt: Merging instructions into the Human role
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Instructions: You are an HR Assistant. Use the following context to answer the user's question accurately. If the answer isn't in the context, say you don't know.\n\nContext: {context}\n\nQuestion: {input}")
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Initializing
rag_chain = load_rag_system()

if rag_chain is None:
    st.error("File 'data.pdf' not found in the repository!")
    st.stop()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR Handbook..."):
            try:
                response = rag_chain.invoke({"input": prompt_text})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
