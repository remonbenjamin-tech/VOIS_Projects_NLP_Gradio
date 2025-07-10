import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer

# ───── NLTK & MODEL SETUP ─────
@st.cache_resource(show_spinner=False)
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
download_nltk()

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")
model = load_model()

# ───── SUMMARIZATION FUNCTION ─────
def summarize(text: str, k: int) -> str:
    sentences = sent_tokenize(text)
    k = min(k, len(sentences))

    stop_words = set(stopwords.words("english"))
    def remove_sw(s: str) -> str:
        return " ".join(w for w in s.lower().split() if w not in stop_words)
    cleaned = [remove_sw(s) for s in sentences]

    embeds = model.encode([s.lower() for s in sentences])

    n = len(sentences)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i, j] = cosine_similarity(
                    embeds[i].reshape(1, -1), embeds[j].reshape(1, -1)
                )[0, 0]

    graph = nx.from_numpy_array(sim)
    ranks = nx.pagerank(graph, max_iter=1000, tol=1e-06)


    ranked = sorted(((ranks[i], s, i) for i, s in enumerate(sentences)),
                    reverse=True)[:k]
    summary = " ".join(s for _, s, _ in sorted(ranked, key=lambda x: x[2]))
    return summary

# ───── STREAMLIT UI ─────
st.title("🧠 Text Summarizer (English)")
st.markdown(
    "Paste an English article or text below, select the number of sentences "
    "you want in the summary, and click **Summarize**."
)

text = st.text_area("Enter your text here:", height=300, key="input_text")

if text:
    max_sent = len(sent_tokenize(text))
    k = st.slider("Number of sentences in summary:", 1, max_sent, 3)
else:
    k = st.slider("Number of sentences in summary:", 1, 10, 3, disabled=True)

if st.button("Summarize", disabled=not text):
    with st.spinner("Summarizing..."):
        summary = summarize(text, k)

    st.subheader("📝 Summary")
    st.write(summary)

    orig_words = len(text.split())
    summ_words = len(summary.split())
    reduction = (1 - summ_words / orig_words) * 100 if orig_words else 0

    st.subheader("📊 Summary Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Length", f"{orig_words} words")
    col2.metric("Summary Length", f"{summ_words} words")
    col3.metric("Reduction", f"{reduction:.1f}%")
