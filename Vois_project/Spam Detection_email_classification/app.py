import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
import json

# Load model and tokenizer
model = load_model('emai_classification/spam_classifier.h5')

# Load tokenizer (saved during training)
with open('emai_classification/tokenizer.json') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)


MAX_LEN = 100  # same as used during training

# Streamlit UI
st.title("📩 Spam Detector")
st.markdown("Enter an SMS message and find out if it's **Spam** or **Not Spam**.")

user_input = st.text_area("✉️ Enter your message here:", height=150)

if st.button("Predict"):
    # Preprocess input
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    # Predict
    prediction = model.predict(padded)[0][1]
    label = "🚫 Spam" if prediction > 0.5 else "✅ Not Spam"

    st.subheader(f"Result: {label}")
    st.caption(f"Confidence: {prediction:.2f}")
