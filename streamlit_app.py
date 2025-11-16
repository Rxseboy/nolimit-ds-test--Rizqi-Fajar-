import streamlit as st
from src.inference import predict
from src.knn_search import load_knn_index, search_similar
from src.embeddings import load_trained_model
from src.dataset import load_balanced_dataset

st.set_page_config(page_title="Tweet Sentiment Classifier", layout="wide")

st.title("Tweet Sentiment Classifier (CardiffNLP RoBERTa)")
st.write("Analyze sentiment and find similar tweets using embeddings + KNN search.")

# Load heavy models one time only
model = load_trained_model()
nn_index = load_knn_index()
train_texts = load_balanced_dataset()["train"]["text"]

# User input
user_text = st.text_area("Enter a tweet for analysis:", height=150)

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # Sentiment prediction
        prediction = predict(user_text)
        st.subheader("Sentiment Prediction")
        st.json(prediction)

        # Similarity Search
        st.subheader("Most Similar Tweets")
        similars = search_similar(
            query=user_text,
            model=model,
            nn=nn_index,
            texts=train_texts,
            k=5,
        )

        for item in similars:
            st.write(f"**• {item['text']}**")
            st.caption(f"Distance: {item['distance']:.4f}")
