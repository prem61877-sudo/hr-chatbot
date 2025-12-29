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

# 1. UI Setup
st.set_page_config(page_title="GlobalTech HR Bot", page_icon="🏢")
st.title("🏢 GlobalTech HR Assistant")

# 2. Get API Key from Streamlit Secrets (Security)
# Note: When testing locally/Colab, you might need to set this manually, 
# but for the final deployment, use st.secrets.
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
except:
    st.warning("Please set your GOOGLE_API_KEY in Streamlit Secrets.")

# 3. Cache the Brain (So it doesn't reload on every click)
@st.cache_resource
def load_rag_system():
    loader = PyPDFLoader("data.pdf")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=st.secrets["GOOGLE_API_KEY"], # Explicitly fetch secret
    temperature=0
)
    
    system_prompt = "You are an HR Assistant. Use the context to answer: {context}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Ensure the data.pdf exists before loading
if os.path.exists("data.pdf"):
    rag_chain = load_rag_system()

    # 4. Chat Interface
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
            response = rag_chain.invoke({"input": prompt_text})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
else:
    st.error("data.pdf not found! Please upload it to the repository.")
