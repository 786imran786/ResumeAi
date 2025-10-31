# resume_model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("Resume.csv")
X = df['Resume']
y = df['Category']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("resume_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model trained and saved!")
