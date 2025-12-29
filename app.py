import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="GlobalTech HR Assistant", page_icon="🏢")
st.title("🏢 GlobalTech AI HR Assistant")

@st.cache_resource
def load_rag_system():
    # Load and split PDF
    loader = PyPDFLoader("data.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    
    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    
    # Gemini 2.5 Flash Config
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0
    )
    return vectorstore, llm

vectorstore, llm = load_rag_system()

# Define System Prompt with Identity and Fallback
system_prompt = (
    "You are the GlobalTech AI HR Assistant. If asked 'who are you' or 'tell me about yourself', "
    "identify as the GlobalTech AI HR Assistant. Use the context to answer. "
    "If the answer is NOT in the context, strictly say: "
    "'This is not written in handbook so you can contact to the Hr manager for further details'\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Build Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
