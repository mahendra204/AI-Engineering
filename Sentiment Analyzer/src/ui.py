import streamlit as st
import requests

st.title("Local Sentiment Analyzer")
text = st.text_area("Enter text", "This product is amazing!")

if st.button("Analyze"):
    resp = requests.post("http://127.0.0.1:8000/predict", json={"text": text})
    data = resp.json()
    st.write(f"Label: {data['label']}")
    st.json(data["scores"])
