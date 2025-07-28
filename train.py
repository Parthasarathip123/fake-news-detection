import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load dataset
df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")

df_fake["label"] = 0  # Fake news
df_real["label"] = 1  # Real news

df = pd.concat([df_fake, df_real])  # Combine both datasets
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Create model
model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Save model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")