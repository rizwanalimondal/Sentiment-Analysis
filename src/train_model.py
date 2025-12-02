import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load prepared dataset
df = pd.read_csv("../data/sentiment_dataset.csv")

X = df['text']
y = df['label']

# Convert labels to numeric
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train 3-class logistic regression
model = LogisticRegression(max_iter=300)
model.fit(X_vec, y_encoded)

# Save model, vectorizer, and label encoder
pickle.dump(model, open("../models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))
pickle.dump(encoder, open("../models/label_encoder.pkl", "wb"))

print("Training complete! Model, Vectorizer & Encoder saved.")
