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
st.set_page_config(page_title="GlobalTech HR Bot", page_icon="🏢", layout="centered")
st.title("🏢 GlobalTech HR Assistant")
st.markdown("---")

# 2. Key Verification
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing API Key! Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# 3. Cached RAG Brain
@st.cache_resource
def load_rag_system():
    if not os.path.exists("data.pdf"):
        return None
        
    loader = PyPDFLoader("data.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0,
        convert_system_message_to_human=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    system_prompt = "You are an HR Assistant. Use the context to answer: {context}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Initialize the system
rag_chain = load_rag_system()

if rag_chain is None:
    st.error("PDF not found! Please upload 'data.pdf' to your GitHub repository.")
    st.stop()

# 4. Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. User Input Logic
if prompt_text := st.chat_input("Ask about company policies..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR Handbook..."):
            try:
                response = rag_chain.invoke({"input": prompt_text})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Brain Error: {str(e)}")
