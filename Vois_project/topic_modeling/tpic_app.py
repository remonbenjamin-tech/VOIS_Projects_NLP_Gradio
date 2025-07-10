import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)

# Preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W+', ' ', text.lower())  # Remove punctuation and lowercase
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit UI
st.title("🧠 Topic Modeling Explorer")
uploaded_file = st.file_uploader("Upload your support ticket file (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Show original data
    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Preprocess text
    df['processed'] = df['text'].apply(preprocess)

    # Show processed data
    st.subheader("🔧 Preprocessed Data")

    # Vectorize
    X = vectorizer.fit_transform(df['processed'])

    # Topic modeling
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(X)

    # Show topics
    st.subheader("💡 Top words per topic")
    for idx, topic in enumerate(lda_model.components_):
        st.write(f"**Topic {idx}**: " + ", ".join(
            [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        ))
