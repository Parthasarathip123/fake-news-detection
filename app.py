import streamlit as st
import pickle
import pandas as pd

# Load model
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📰 Fake News Detector")

# User input
news_text = st.text_area("Enter the news article text:")

if st.button("Check News"):
    if news_text:
        prediction = model.predict([news_text])[0]
        result = "✅ Real News" if prediction == 1 else "🚨 Fake News"
        st.subheader(result)
    else:
        st.warning("⚠ Please enter some news text.")