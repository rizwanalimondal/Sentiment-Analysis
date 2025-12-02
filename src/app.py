import streamlit as st
from predict import predict_sentiment

st.title("Three-Class Sentiment Analysis")
st.write("Positive • Neutral • Negative")

user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if user_input.strip():
        label, probs = predict_sentiment(user_input)

        st.subheader(f"Predicted Sentiment: {label.upper()}")

        st.write("Class Probabilities:")
        for cls, p in probs.items():
            st.write(f"{cls.capitalize()}: {p}")
    else:
        st.warning("Please enter some text.")
