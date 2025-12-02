import pickle

# Load saved components
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

def predict_sentiment(text):
    # Vectorize input
    X = vectorizer.transform([text])

    # Predict class (encoded)
    encoded_pred = model.predict(X)[0]

    # Convert back to original label
    decoded_label = encoder.inverse_transform([encoded_pred])[0]

    # Probabilities for each class
    proba = model.predict_proba(X)[0]

    # Map probabilities to correct label order
    class_probs = {
        encoder.inverse_transform([0])[0]: round(proba[0], 3),
        encoder.inverse_transform([1])[0]: round(proba[1], 3),
        encoder.inverse_transform([2])[0]: round(proba[2], 3),
    }

    return decoded_label, class_probs
