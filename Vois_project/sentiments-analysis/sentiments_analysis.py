# sentiment_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Title
st.set_page_config(page_title="📊 Sentiment Dashboard", layout="wide")
st.title("📊 Sentiment Dashboard for Customer Feedback")

# File upload
uploaded_file = st.file_uploader("Upload a CSV with 'date' and 'feedback' columns", type="csv")

# Sentiment analysis function
def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure correct types
    df['date'] = pd.to_datetime(df['date'])
    df['feedback'] = df['feedback'].astype(str)

    # Analyze sentiments
    df['sentiment'] = df['feedback'].apply(analyze_sentiment)

    # Date filtering
    min_date, max_date = df['date'].min(), df['date'].max()
    start_date, end_date = st.date_input("Filter by date", [min_date, max_date])

    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df = df[mask]

    # Show data
    with st.expander("📄 View DataFrame"):
        st.dataframe(df)

    # Plot sentiment over time
    st.subheader("📈 Sentiment Over Time")
    daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(daily_sentiment)

    # Word cloud for each sentiment
    st.subheader("☁️ Word Clouds")
    cols = st.columns(3)
    for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
        text = " ".join(df[df['sentiment'] == sentiment]['feedback'])
        if text:
            wordcloud = WordCloud(background_color='white', width=400, height=300).generate(text)
            with cols[i]:
                st.markdown(f"**{sentiment}**")
                st.image(wordcloud.to_array())

else:
    st.info("Please upload a valid CSV file.")

