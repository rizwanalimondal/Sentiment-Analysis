import pandas as pd

# Load raw dataset
df = pd.read_csv("text_emotion.csv")

# Keep only relevant columns
df = df[['sentiment', 'content']]

# Define class groups
positive = ["happiness", "love", "fun", "enthusiasm"]
neutral = ["neutral"]
# Everything else is negative

def map_label(emotion):
    if emotion in positive:
        return "positive"
    elif emotion in neutral:
        return "neutral"
    else:
        return "negative"

# Apply mapping
df['label'] = df['sentiment'].apply(map_label)

# Rename & keep necessary columns
df = df[['content', 'label']]
df.columns = ['text', 'label']

# Save clean dataset
df.to_csv("sentiment_dataset.csv", index=False)

print("Three-class dataset created: sentiment_dataset.csv")
