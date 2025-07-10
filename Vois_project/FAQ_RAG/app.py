# app.py
import streamlit as st
import wikipedia
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


st.set_page_config(page_title="WikiQA 🤖", layout="wide")
st.title("📚 Wikipedia RAG")


with st.sidebar:
    topic_input = st.text_input("Wikipedia article title", value="Sleep and memory")
    load_clicked = st.button("🔄 Load article")

# Persist the currently loaded topic in Session State
if load_clicked or "topic" not in st.session_state:
    st.session_state.topic = topic_input.strip()

topic = st.session_state.topic  # the active article

st.info(f"Current article: **{topic}**")


@st.cache_data(show_spinner=False, max_entries=20)
def fetch_wikipedia(_topic: str):
    """Fetch and split a Wikipedia page into <500-token chunks."""
    try:
        content = wikipedia.page(_topic).content
    except Exception as e:
        raise RuntimeError(f"⚠️ Couldn’t fetch “{_topic}”: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(content)
    if len(chunks) < 2:
        raise RuntimeError("⚠️ Article is too short to build a retriever.")
    return [Document(page_content=ch) for ch in chunks]

@st.cache_resource(show_spinner=False, max_entries=20)
def build_retriever(_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(_docs, embeddings)
    return vectorstore.as_retriever()


try:
    documents = fetch_wikipedia(topic)
    retriever = build_retriever(documents)
except RuntimeError as err:
    st.error(err.args[0])
    st.stop()

# ---------- Prompt + LLM
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the context to answer the user’s question.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ]
)

llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="60ca17c8ea1bd4c7149e011dcbc7146b0b28712e4d9d7c15a926d1df1749ad52",
    temperature=0.3,
)
combine_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_chain)

# ---------- Chat UI
question = st.text_input("Ask a question about the article:")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        response = ""
        msg_placeholder = st.empty()
        for chunk in qa_chain.stream({"input": question}):
            if "answer" in chunk:
                response += chunk["answer"]
                msg_placeholder.markdown(response)
